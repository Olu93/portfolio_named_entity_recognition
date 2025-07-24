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
        labels: list[str] = ["PERSON"],
        require_full_name: bool = True,
        temperature: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.labels = set(labels)
        self.label_map = {
            "PERSON": "persons",
            "ORG": "organizations",
            "LOC": "locations"
        }
        self.require_full_name = require_full_name
        self.parser = PydanticOutputParser(pydantic_object=LLMResult)
        self.prompt = ChatPromptTemplate([
            SystemMessage(content=(
                f"Extract full‑name entities of types {', '.join(self.labels)} "
                "from the user text and return JSON according to the following instructions: "
                "{schema}"
            )),
            MessagesPlaceholder(variable_name="messages")
        ], parttial_variables={"schema": self.parser.get_format_instructions()})
        self.chain = self.prompt | self.llm | self.parser

    def _fit(self, X: TextInput, y: TextInput = None):
        return self  # no training

    def _predict(self, X: TextInput):
        out: dict[str, list[str]] = {}
        
        resp = self.chain.batch([{"messages": [HumanMessage(content=text)]} for text in X])
        for label in self.labels:
            for text,result in zip(X,resp):
                if hasattr(result, self.label_map[label]):
                    out[self.label_map[label]] = [e.text for e in result[self.label_map[label]]]
                else:
                    out[self.label_map[label]] = []
        return out

if __name__ == "__main__":
    test_extractor(
        extractor=LangChainEntityExtractor(),
        extractor_multi=LangChainEntityExtractor(),
        extractor_multi_many=LangChainEntityExtractor()
    )
