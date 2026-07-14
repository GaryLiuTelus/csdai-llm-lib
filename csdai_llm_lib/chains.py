from typing import Callable, Type
from pydantic import BaseModel
from abc import abstractmethod
from dataclasses import dataclass, field
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable, RunnableSequence
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from .utils import get_secret
from .processors import LLMChainBuilder, ChatClient

class NoneHandlingPydanticOutputParser(PydanticOutputParser):
    def parse(self, text: str):
        try:
            return super().parse(text)
        except OutputParserException:
            return None

    def invoke(self, input, config=None, **kwargs):
        try:
            return super().invoke(input, config, **kwargs)
        except OutputParserException:
            return None

class LLMChain:
    def __init__(
            self,
            llm_base_url: str,
            llm_api_key_secret_name: str,
            llm_model_name: str,
            secret_project: str
        ) -> None:
        self._llm_base_url = llm_base_url
        self._llm_api_key_secret_name = llm_api_key_secret_name
        self._llm_model_name = llm_model_name
        self._secret_project = secret_project

    def create_client(self, model_name: str) -> ChatClient:
        return ChatClient(
                base_url=self._llm_base_url,
                api_key=get_secret(
                    self._llm_api_key_secret_name,
                    project=self._secret_project
                ),
                model_name=model_name
            )

    def assembly_llm_chain(
            self, system_message: str,
            prompt_template: str,
            pydantic_obj: type[BaseModel],
            model_name: str = ""
        ) -> RunnableSequence:

        if model_name != "":
            chat_client = self.create_client(model_name)
        else:
            chat_client = self.create_client(self._llm_model_name)
        llm_chain = LLMChainBuilder()

        json_extractor = chat_client.structured_client(pydantic_obj)

        llm_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(system_message),
                    prompt_template
                ]
                )
        llm_chain.add_runnable(llm_prompt)
        llm_chain.add_runnable(json_extractor)
        return llm_chain.chain

    @property
    @abstractmethod
    def full_chains(self) -> Runnable:
        ...

@dataclass
class LLMChainContext:
    llm_chain: Type[LLMChain]
    output_schema: Type[BaseModel]
    init_prompt_template: str
    default_model_name: str
    logger: logging.Logger
    max_seconds_per_record: int = 20
    max_job_timeout: int = 72000 # maximum 20 hours
    extra_context: dict[int, Callable] = field(default_factory=dict)
    output_column_name_suffix: str = ""

def simple_data_extractor(field_name: str) -> Callable:
    return lambda raw_output: getattr(raw_output, field_name)