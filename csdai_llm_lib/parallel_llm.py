import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from string import Formatter
from threading import Lock, Thread
from typing import Any

import polars as pl
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .base_env import get_base_settings
from .chains import LLMChainContext, simple_data_extractor
from .processors import DataframeLLMAnalyzer, ProcessorOutput, ProgressDisplay
from .utils import get_thread_parameters, readable_time


def assembly_dataframe(dataframes: list[pl.DataFrame]) -> pl.DataFrame:
    # In case type mismatch due to NULL columns
    # Get expected types from all dataframes
    expected_types = {}
    for df in dataframes:
        if len(df) == 0:
            continue
        for col, dtype in df.schema.items():
            existing_dtype = expected_types.get(col)
            if existing_dtype is None or ("null" in str(existing_dtype).lower() and existing_dtype != dtype):
                expected_types[col] = dtype

    if len(expected_types) == 0:
        return pl.DataFrame()
    # Cast and concatenate
    result: pl.DataFrame = pl.concat(
        df.with_columns(
            *[
                pl.col(col).cast(expected_types[col], strict=False)
                for col in df.columns
                if df[col].dtype != expected_types[col]
            ]
        )
        for df in dataframes
        if len(df) > 0
    )
    return result

class AggregateProgress:
    """One bar for the whole job: records drive the bar, chunks drive the postfix.

    Rendered as e.g.
    `processing:  45%|####      | 4500/10000 [12:31<15:02, 5.99record/s, 3/8 threads done]`
    """

    def __init__(self, total_records: int, total_chunks: int) -> None:
        self._bar = tqdm(total=total_records, desc="processing", unit="record")
        self._total_chunks = total_chunks
        self._done_chunks = 0
        # tqdm serialises its own counter; this guards ours.
        self._lock = Lock()
        self._render()

    def record_done(self, n: int = 1) -> None:
        self._bar.update(n)

    def chunk_done(self) -> None:
        with self._lock:
            self._done_chunks += 1
            self._render()

    def _render(self) -> None:
        self._bar.set_postfix_str(f"{self._done_chunks}/{self._total_chunks} threads done", refresh=True)

    def close(self) -> None:
        self._bar.close()


class PerThreadProgress:
    """One live bar per data chunk, plus a summary block printed when the job ends.

    Bars are built with `leave=False` and their final frame is reprinted by hand,
    because tqdm cannot both keep a positioned bar on screen and clean up after
    it. `close()` with `leave=True` renders the frame at `pos=0` — the anchor
    line, not the bar's own line — then scrolls one line, and never erases the
    line the bar had been painting. Closing chunk bars as their workers finish
    therefore strands the last in-progress frame of every bar still running:
    nothing repaints those lines, so later log records land on them from column 0
    with no erase-to-end-of-line and the tail of the stale bar survives to the
    right of the message.

    `leave=False` takes the branch that does erase, so a finishing chunk hands its
    line back cleanly. `finish()` snapshots the frame first — tqdm recomputes
    `elapsed` on every render, so a frame captured later would report the whole
    job's runtime instead of that chunk's — and `close()` writes the snapshots out
    in data-index order, filling the lines the bars just vacated.
    """

    def __init__(self) -> None:
        self._bars: dict[int, tqdm] = {}
        self._summaries: dict[int, str] = {}
        # Workers register and finish bars; the main thread drains both dicts.
        self._lock = Lock()

    def bar(self, idx: int, total: int) -> tqdm:
        # A per-chunk bar can only be built where the chunk's size is known.
        chunk_bar = tqdm(
            total=total, desc=f"data {idx:0>2}", unit="record", position=idx - 1, leave=False
        )
        with self._lock:
            self._bars[idx] = chunk_bar
        return chunk_bar

    def finish(self, idx: int) -> None:
        """Freeze one chunk's final frame and give its screen line back."""
        with self._lock:
            chunk_bar = self._bars.pop(idx, None)
            if chunk_bar is None:
                return
            self._summaries[idx] = str(chunk_bar)
        chunk_bar.close()

    def close(self) -> None:
        with self._lock:
            # Anything still open timed out; snapshot it where it stopped.
            leftover = sorted(self._bars.items())
            for idx, chunk_bar in leftover:
                self._summaries[idx] = str(chunk_bar)
            self._bars.clear()
            summaries = [self._summaries[idx] for idx in sorted(self._summaries)]
            self._summaries.clear()
        for _, chunk_bar in leftover:
            chunk_bar.close()
        if summaries:
            # Erasing a positioned bar leaves the cursor part-way along the line
            # the `position=0` bar vacated (tqdm only rewinds for `pos == 0`), so
            # the block starts with `\r` to reclaim that line from column 0.
            tqdm.write("\r" + "\n".join(summaries))


def multi_thread_llm(
        data_list: Iterable[pl.DataFrame],
        llm_analysis_function: Callable[[int, pl.DataFrame], ProcessorOutput],
        llm_logger: logging.Logger,
        max_sec_per_record: int = 20,
        max_timeout: int = 72000 # no more than 20 hours
    ) -> ProcessorOutput:

    def wrapper_func(
            func: Callable[[int, pl.DataFrame], ProcessorOutput],
            data: pl.DataFrame,
            idx: int,
            return_dict: dict
        ) -> None:
        result = func(idx, data)
        return_dict[idx] = result

    max_data_size = 0
    return_dict: dict[int, ProcessorOutput] = {}
    threads: list[Thread] = []

    for i, data in enumerate(data_list, start=1):
        max_data_size = max(max_data_size, len(data))
        thread = Thread(
            target=wrapper_func,
            args=(llm_analysis_function, data, i, return_dict),
            daemon=True
        )
        threads.append(thread)
        thread.start()

    timeout_sec = min(max_data_size * max_sec_per_record, max_timeout)

    deadline = time.time() + timeout_sec

    for i, thread in enumerate(threads, start=1):
        remaining = max(0.0, deadline - time.time())
        thread.join(timeout=remaining)
        if thread.is_alive():
            llm_logger.error("Thread %d did not finish within timeout (%d seconds), skipping", i, timeout_sec)

    processed_data = list(return_dict.values())
    if len(processed_data) == 0:
        return ProcessorOutput(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), {})
    completed, defects, failed = [], [], []
    early_terminated: defaultdict[int, list[pl.DataFrame]] = defaultdict(list)
    for processed in processed_data:
        completed.append(processed.completed)
        defects.append(processed.defects)
        failed.append(processed.failed)
        for s, d in processed.early_terminated.items():
            early_terminated[s].append(d)

    return ProcessorOutput(
        assembly_dataframe(completed),
        assembly_dataframe(defects),
        assembly_dataframe(failed),
        {s: assembly_dataframe(d) for s, d in early_terminated.items()}
    )


def llm_processor(
        llm_context: LLMChainContext,
        input_columns: list[str],
        output_column_name_suffix: str = "",
        aggregate: AggregateProgress | None = None,
        per_thread: PerThreadProgress | None = None,
    ) -> Callable[[int, pl.DataFrame], ProcessorOutput]:
    record_hook: Callable[[int], Any] | None = aggregate.record_done if aggregate is not None else None

    def info_extractor(
            idx: int,
            input_data: pl.DataFrame,
        ) -> ProcessorOutput:
        settings = get_base_settings()
        sales_opp_analyzer = llm_context.llm_chain(
            llm_base_url=settings.llm_base_url,
            llm_api_key_secret_name=settings.llm_api_key_secret_name,
            llm_model_name=llm_context.default_model_name,
            secret_project=settings.project_id
        )


        llm_processors = DataframeLLMAnalyzer(
            input_data,
            sales_opp_analyzer.full_chains,
            input_columns=input_columns,
            llm_logger=llm_context.logger,
            extra_context=llm_context.extra_context,
            data_index=idx
        )

        chunk_bar = per_thread.bar(idx, len(input_data)) if per_thread is not None else None
        try:
            processed = llm_processors.get_results(
                {
                    f"{f}{output_column_name_suffix}": simple_data_extractor(f)
                    for f in llm_context.output_schema.model_fields
                },
                progress_hook=chunk_bar.update if chunk_bar is not None else record_hook,
            )
        finally:
            # Frees the bar's line; the frame it snapshots is replayed in the
            # summary block that `PerThreadProgress.close()` prints.
            if per_thread is not None:
                per_thread.finish(idx)

        # Only reached when the chunk finished; a raising chunk stays uncounted,
        # so the postfix keeps showing it as outstanding.
        if aggregate is not None:
            aggregate.chunk_done()
        return processed
    return info_extractor

def llm_job_assignment(input_data: pl.DataFrame, llm_context: LLMChainContext) -> ProcessorOutput:
    data_size = len(input_data)
    thread_chunk_size, thread_count = get_thread_parameters(data_size)
    # `thread_chunk_size` floors, so any remainder spills into one extra chunk.
    chunk_count = thread_count + 1 if data_size > thread_chunk_size * thread_count else thread_count
    llm_context.logger.info(
        "Total %d records are split into %d chunks, each chunk contain maximum %d records.",
        data_size,
        chunk_count,
        thread_chunk_size
    )

    input_columns = [
        field
        for _, field, _, _ in Formatter().parse(llm_context.init_prompt_template)
        if field is not None and field != "format_instructions"
    ]

    progress_display = llm_context.progress_display
    aggregate = (
        AggregateProgress(data_size, chunk_count)
        if progress_display is ProgressDisplay.AGGREGATE
        else None
    )
    per_thread = PerThreadProgress() if progress_display is ProgressDisplay.PER_THREAD else None
    # Bars write to stderr, the logger's console handler to stdout, and neither
    # knows the other's cursor state. Routing records through `tqdm.write()`
    # clears the bars, prints, then redraws them.
    log_redirect = (
        logging_redirect_tqdm(loggers=[llm_context.logger], tqdm_class=tqdm)
        if aggregate is not None or per_thread is not None
        else nullcontext()
    )

    start = time.perf_counter()
    with log_redirect:
        try:
            results = multi_thread_llm(
                input_data.iter_slices(thread_chunk_size),
                llm_processor(
                    llm_context,
                    input_columns,
                    llm_context.output_column_name_suffix,
                    aggregate=aggregate,
                    per_thread=per_thread,
                ),
                max_sec_per_record=llm_context.max_seconds_per_record,
                max_timeout=llm_context.max_job_timeout,
                llm_logger=llm_context.logger
            )
        finally:
            if aggregate is not None:
                aggregate.close()
            if per_thread is not None:
                per_thread.close()
    llm_context.logger.info(
        "%d records are processed in %s",
        data_size,
        readable_time(time.perf_counter() - start)
    )
    return results

