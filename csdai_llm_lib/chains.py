import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableSequence
from pydantic import BaseModel

from .processors import (
    ChatClient,
    LLMChainBuilder,
    ProgressDisplay,
    require_step_names,
)
from .utils import get_secret


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

class LLMChain(ABC):
    def __init__(
            self,
            llm_base_url: str,
            llm_api_key_secret_name: str,
            llm_model_name: str,
            secret_project: str,
            llm_api_key: str | None = None
        ) -> None:
        """`llm_api_key`, when given, is used as-is instead of reading the secret."""
        self._llm_base_url = llm_base_url
        self._llm_api_key_secret_name = llm_api_key_secret_name
        self._llm_model_name = llm_model_name
        self._secret_project = secret_project
        self._llm_api_key = llm_api_key

    def create_client(self, model_name: str) -> ChatClient:
        return ChatClient(
                base_url=self._llm_base_url,
                api_key=self._llm_api_key or get_secret(
                    self._llm_api_key_secret_name,
                    project=self._secret_project
                ),
                model_name=model_name
            )

    def assembly_llm_chain(
            self, system_message: str,
            prompt_template: str,
            pydantic_obj: type[BaseModel],
            model_name: str = "",
            *,
            name: str,
        ) -> RunnableSequence:
        """Build `prompt | structured llm`; steps are named `<name>_prompt` / `<name>_llm`."""

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
        llm_chain.add_runnable(llm_prompt, f"{name}_prompt")
        llm_chain.add_runnable(json_extractor, f"{name}_llm")
        return llm_chain.chain

    @abstractmethod
    def _build_chains(self) -> Runnable:
        """Assemble the full chain; every step must be named."""

    @property
    def full_chains(self) -> Runnable:
        chain = self._build_chains()
        require_step_names(chain)
        return chain

@dataclass
class LLMChainContext:
    llm_chain: type[LLMChain]
    output_schema: type[BaseModel]
    init_prompt_template: str
    default_model_name: str
    logger: logging.Logger
    max_job_timeout: int = 72000 # maximum 20 hours
    extra_context: dict[int, Callable] = field(default_factory=dict)
    output_column_name_suffix: str = ""
    progress_display: ProgressDisplay = ProgressDisplay.NONE
    record_timeout_seconds: float = 120
    """Hard cap in seconds on one record across all its steps. A record that
    overruns is failed on its own; a chunk's total budget is this value times
    the chunk size."""

def simple_data_extractor(field_name: str) -> Callable:
    return lambda raw_output: getattr(raw_output, field_name)