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
import logging

logger = logging.getLogger(__name__)

class Entity(BaseModel):
    text: str = Field(description="The text of the entity")
    type: Literal["PER", "ORG", "LOC"] = Field(description="The type of the entity")

class LLMResult(BaseModel):
    persons: list[str] = Field(default=[], description="List of full‑name entities of type PER which are persons with their full name")
    organizations: list[str] = Field(default=[], description="List of full‑name entities of type ORG which are organizations")
    locations: list[str] = Field(default=[], description="List of full‑name entities of type LOC which are locations")

class LangChainEntityExtractor(SingleEntityExtractor):
    
    MAP = {
        "persons": ["PER"],
        "organizations": ["ORG"],
        "locations": ["LOC"]
    }

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        label: str = "persons",
        require_full_name: bool = True,
        temperature: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(label=label, *args, **kwargs)
        self.labels = self.MAP[label]
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.require_full_name = require_full_name


    def _fit(self, X: TextInput, y: TextInput = None):
        # TODO: Add examples
        examples = [e for i in y for e in i]
        self.prompt = ChatPromptTemplate([
            SystemMessage(content=(
                f"Extract full‑name entities of types {self.labels} where persons is {self.MAP['persons']}, organizations is {self.MAP['organizations']}, and locations is {self.MAP['locations']}"
            )),
            SystemMessage(content=f"Here are some examples: {examples}"),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.chain = self.prompt | self.llm.with_structured_output(LLMResult)
        return self  # no training

    def _predict(self, X: TextInput):
        out: list[str] = []

        for text in X:
            try:
                response: LLMResult = self.chain.invoke({"messages": [HumanMessage(content=text)]})
                out.append(getattr(response, self.label))
            except Exception as e:
                logger.error(f"Error predicting entities for text: {text}")
                logger.error(f"Error: {e}")
                out.append([])
        return out


if __name__ == "__main__":
    test_extractor(
        extractor=LangChainEntityExtractor(),
        extractor_multi=LangChainEntityExtractor(),
        extractor_multi_many=LangChainEntityExtractor()
    )
