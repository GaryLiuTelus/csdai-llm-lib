import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from typing import Any, cast

import polars as pl
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables.base import Runnable, RunnableSequence
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from pydantic import ValidationError
from pydantic.types import SecretStr

retryable_errors = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

class ProgressDisplay(StrEnum):
    """How to report record-level progress while a job runs."""

    NONE = "none"
    """No progress bar; periodic counts go to the logger (default)."""
    AGGREGATE = "aggregate"
    """One bar covering every record across all threads."""
    PER_THREAD = "per_thread"
    """One bar per data chunk, stacked by data index."""

@dataclass
class ProcessorOutput:
    completed: pl.DataFrame
    defects: pl.DataFrame
    failed: pl.DataFrame
    early_terminated: dict[int, pl.DataFrame]
@dataclass
class DataframeLLMAnalyzer:
    data: pl.DataFrame
    llm_chain: Runnable
    input_columns: list[str]
    llm_logger: logging.Logger
    extra_context: dict[int, Callable]
    data_index: int = 1


    def get_results(
            self,
            unpacker_funcs: dict[str, Callable[[dict[str, Any]], Any]],
            progress_hook: Callable[[int], Any] | None = None,
        ) -> ProcessorOutput:
        """Run the chain over every row.

        `progress_hook` is called with 1 per finished record — pass a tqdm
        bar's `update` to drive a bar the caller owns. Supplying it suppresses
        the periodic progress logging, so you get one or the other.
        """
        llm_completed, input_defects, llm_failed = [], [], []
        skipped_data: defaultdict[int, list[dict]]= defaultdict(list)
        log_progress = progress_hook is None
        for idx, row in enumerate(self.data.iter_rows(named=True), start=1):
            try:
                invoke_input = {}
                for col in self.input_columns:
                    if row[col] is None:
                        break
                    invoke_input[col] = row[col]
                if len(invoke_input) < len(self.input_columns):
                    input_defects.append(row)
                    continue

                llm_output = None
                if isinstance(self.llm_chain, RunnableSequence):
                    steps = self.llm_chain.steps
                else:
                    steps = [self.llm_chain]
                skipped_step = 0
                for i, c in enumerate(steps, start=1):
                    try:
                        additional_context_func = self.extra_context.get(i)
                        if additional_context_func is not None:
                            invoke_input = additional_context_func(row, invoke_input, llm_output or [])
                        if invoke_input is None:
                            skipped_step = i
                            break
                        llm_output = c.invoke(invoke_input)

                    except OutputParserException as e:
                        self.llm_logger.error("Parsing exception at step %d (data %d, record %d): %s", i, self.data_index, idx, e)
                        llm_output = None
                        break
                    except ValidationError as e:
                        self.llm_logger.error("Pydantic validation error at step %d (data %d, record %d): %s", i, self.data_index, idx, e)
                        llm_output = None
                        break
                    except retryable_errors as e:
                        self.llm_logger.error("Retries exhausted at step %d (data %d, record %d): %s", i, self.data_index, idx, e)
                        llm_output = None
                        break
                    invoke_input = llm_output

                if skipped_step > 0:
                    target_data = skipped_data[skipped_step]
                else:
                    target_data = llm_completed
                if llm_output is not None:
                    if isinstance(llm_output, list):
                        if len(llm_output) == 0:
                            if skipped_step == 0:
                                input_defects.append(row)
                            else:
                                skipped_data[skipped_step].append(row)
                        for lo in llm_output:
                            for col_name, f in unpacker_funcs.items():
                                row[col_name] = f(lo)
                            target_data.append({**row})
                    else:
                        for col_name, f in unpacker_funcs.items():
                            row[col_name] = f(llm_output)
                        target_data.append(row)
                else:
                    llm_failed.append(row)

                if log_progress and idx % 500 == 0:
                    self.llm_logger.info("%d records of data %d has been processed!", idx, self.data_index)
            except Exception as e:  # noqa: BLE001
                # Last-resort safety net: never let one record abort the batch.
                self.llm_logger.error("Unexpected error (data %d, record %d):\n%s", self.data_index, idx, e)
                llm_failed.append(row)
                continue
            finally:
                # In `finally` so records that `continue` early still count.
                if progress_hook is not None:
                    progress_hook(1)
        return ProcessorOutput(
            pl.DataFrame(llm_completed),
            pl.DataFrame(input_defects),
            pl.DataFrame(llm_failed),
            {
                step: pl.DataFrame(early_terminated_data)
                for step, early_terminated_data in skipped_data.items()
            }
        )



class LLMChainBuilder:
    def __init__(self) -> None:
        self._chains: list[Runnable] = []

    def add_runnable(self, new: Runnable):
        self._chains.append(new)

    @property
    def chain(self)-> RunnableSequence:
        if len(self._chains) == 0:
            raise ValueError("need at least one Runnable to build a chain")
        return cast(RunnableSequence, reduce(lambda a, b: a | b, self._chains))

class ChatClient:
    def __init__(
            self, base_url: str,
            api_key: str,
            model_name: str,
            max_retries: int = 3,
            **kwargs
        ) -> None:
        self._base_client = ChatOpenAI(
            base_url=base_url,
            api_key=SecretStr(api_key),
            model=model_name,
            timeout=60,
            model_kwargs=kwargs
        )

        self._max_retries = max_retries

    def structured_client(self, pydantic_obj: type) -> Runnable:
        return self._base_client.with_structured_output(pydantic_obj).with_retry(
            stop_after_attempt=self._max_retries,
            retry_if_exception_type=retryable_errors,
            wait_exponential_jitter=True,
            exponential_jitter_params={
                "initial": 2.0,
                "jitter": 1.0
            }
        )
