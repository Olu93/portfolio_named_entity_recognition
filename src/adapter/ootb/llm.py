from urllib import response
from port.entity_extractor import SingleEntityExtractor
from utils.develop import test_extractor
from utils.typings import TextInput
import json
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import Literal, Optional


class Entity(BaseModel):
    text: str = Field(description="The text of the entity")
    type: Literal["PERSON", "ORG", "LOC"] = Field(description="The type of the entity")

class LLMResult(BaseModel):
    persons: Optional[list[Entity]] = Field(description="List of full‑name entities of type PERSON")
    organizations: Optional[list[str]] = Field(description="List of full‑name entities of type ORG")
    locations: Optional[list[str]] = Field(description="List of full‑name entities of type LOC")

class LangChainEntityExtractor(SingleEntityExtractor):
    def __init__(
        self,
        model_name: str = "gpt-4o",
        label: str = "PERSON",
        require_full_name: bool = True,
        temperature: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.label = label
        self.label_map = {
            "PERSON": "persons",
            "ORG": "organizations",
            "LOC": "locations"
        }
        self.require_full_name = require_full_name
        self.parser = PydanticOutputParser(pydantic_object=LLMResult)
        self.format_instructions = self.parser.get_format_instructions()    

        self.prompt = ChatPromptTemplate([
            SystemMessage(content=(
                f"Extract full‑name entities of types {self.label} "
                "from the user text and return JSON according to the following instructions: "
            )),
            SystemMessage(content=self.format_instructions),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def _fit(self, X: TextInput, y: TextInput = None):
        # TODO: Add examples
        return self  # no training

    def _predict(self, X: TextInput):
        out: list[str] = []
        all_input = [{"messages": [HumanMessage(content=text)]} for text in X]
        resp: list[LLMResult] = self.chain.batch(all_input)
        for _,result in zip(X,resp):
            if hasattr(result, self.label_map[self.label]):
                out.append([e.text for e in getattr(result, self.label_map[self.label])])
            else:
                out.append([])
        return out

if __name__ == "__main__":
    test_extractor(
        extractor=LangChainEntityExtractor(),
        extractor_multi=LangChainEntityExtractor(),
        extractor_multi_many=LangChainEntityExtractor()
    )
