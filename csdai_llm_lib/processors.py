import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from typing import Any, cast
from uuid import UUID

import polars as pl
from langchain_core.callbacks import BaseCallbackHandler
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

def step_name(step: Runnable) -> str:
    """Name of a chain step: its `run_name` config if set, else the Runnable's own name."""
    cfg = getattr(step, "config", None) or {}
    return cfg.get("run_name") or step.get_name()


def require_step_names(chain: Runnable) -> None:
    """Raise if any step of `chain` has neither a `run_name` config nor an explicit name."""
    steps = chain.steps if isinstance(chain, RunnableSequence) else [chain]
    unnamed = [
        i
        for i, s in enumerate(steps, start=1)
        if not (getattr(s, "config", None) or {}).get("run_name") and s.name is None
    ]
    if unnamed:
        raise ValueError(
            f"Unnamed chain steps at positions {unnamed}; "
            "use assembly_llm_chain(name=...), RunnableLambda(name=...), "
            "RunnableEach(bound=..., name=...) or .with_config(run_name=...)"
        )


def invoke_with_timeout(
    runnable: Runnable, payload: Any, timeout: float | None, config: dict | None = None
) -> Any:
    """`runnable.invoke(payload)`, raising `TimeoutError` if it takes longer than `timeout` seconds.

    A call that overruns keeps running in a daemon thread until its own client
    timeout fires; its result is discarded. `timeout=None` calls it directly.
    """
    if timeout is None:
        return runnable.invoke(payload, config=config)
    if timeout <= 0:
        raise TimeoutError("no time left for this record")
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = runnable.invoke(payload, config=config)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller below
            box["error"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"step did not finish within {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["result"]


class StepErrorTracker(BaseCallbackHandler):
    """Remember which named sub-run failed while a (possibly nested) step runs.

    Use one tracker per `invoke` and pass it as `config={"callbacks": [tracker]}`.
    Runs are keyed by `run_id` because a mapped step executes its items in a
    thread pool, so "the last step that started" would be ambiguous. An error
    propagates outward from the step that raised it, so the first error event
    is the innermost step (`parent_run_id` is not reliable for this).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._names: dict[UUID, str] = {}
        self._errored: list[UUID] = []

    def _start(self, run_id: UUID, name: str | None) -> None:
        with self._lock:
            self._names[run_id] = name or "<unnamed>"

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, name=None, **kwargs):
        self._start(run_id, name)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, name=None, **kwargs):
        self._start(run_id, name)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, name=None, **kwargs):
        self._start(run_id, name)

    def _error(self, run_id: UUID) -> None:
        with self._lock:
            self._errored.append(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._error(run_id)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._error(run_id)

    def failed_step(self) -> str | None:
        """Name of the innermost failed run, or None if nothing failed."""
        with self._lock:
            return self._names.get(self._errored[0]) if self._errored else None


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
    record_timeout: float | None = None
    """Hard cap in seconds on one record across all its steps; an overrun record is failed."""


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
                record_start = time.monotonic()
                if isinstance(self.llm_chain, RunnableSequence):
                    steps = self.llm_chain.steps
                else:
                    steps = [self.llm_chain]
                skipped_step = 0
                for i, c in enumerate(steps, start=1):
                    tracker = StepErrorTracker()
                    try:
                        additional_context_func = self.extra_context.get(i)
                        if additional_context_func is not None:
                            invoke_input = additional_context_func(row, invoke_input, llm_output or [])
                        if invoke_input is None:
                            skipped_step = i
                            self.llm_logger.debug(
                                "Skipped before step %d [%s] (data %d, record %d): extra_context returned None (previous output: %s)",
                                i, step_name(c), self.data_index, idx,
                                f"{len(llm_output)} item(s)" if isinstance(llm_output, list) else type(llm_output).__name__,
                            )
                            break
                        llm_output = invoke_with_timeout(
                            c,
                            invoke_input,
                            None
                            if self.record_timeout is None
                            else self.record_timeout - (time.monotonic() - record_start),
                            config={"callbacks": [tracker]},
                        )

                    except OutputParserException as e:
                        self.llm_logger.error("Parsing exception at step %d [%s] (data %d, record %d): %s", i, tracker.failed_step() or step_name(c), self.data_index, idx, e)
                        llm_output = None
                        break
                    except ValidationError as e:
                        self.llm_logger.error("Pydantic validation error at step %d [%s] (data %d, record %d): %s", i, tracker.failed_step() or step_name(c), self.data_index, idx, e)
                        llm_output = None
                        break
                    except TimeoutError as e:
                        self.llm_logger.error(
                            "Record timed out at step %d [%s] (data %d, record %d, limit %ss): %s",
                            i, tracker.failed_step() or step_name(c), self.data_index, idx, self.record_timeout, e,
                        )
                        llm_output = None
                        break
                    except retryable_errors as e:
                        self.llm_logger.error(
                            "Retries exhausted at step %d [%s] (data %d, record %d): %s: %s (cause: %r)",
                            i, tracker.failed_step() or step_name(c), self.data_index, idx, type(e).__name__, e, e.__cause__,
                        )
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
        step_labels = (
            {i: step_name(c) for i, c in enumerate(self.llm_chain.steps, start=1)}
            if isinstance(self.llm_chain, RunnableSequence)
            else {1: step_name(self.llm_chain)}
        )
        for skipped_at, skipped_rows in skipped_data.items():
            self.llm_logger.info(
                "data %d: %d record(s) ended early before step %d [%s]",
                self.data_index, len(skipped_rows), skipped_at, step_labels.get(skipped_at, "?"),
            )
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

    def add_runnable(self, new: Runnable, name: str | None = None):
        self._chains.append(new.with_config(run_name=name) if name else new)

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
            # The OpenAI client's own retries multiply with `with_retry`
            # (up to 3 x 3 attempts); keep a single, visible retry layer.
            max_retries=0,
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
