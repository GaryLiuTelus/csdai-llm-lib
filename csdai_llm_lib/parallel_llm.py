import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from string import Formatter
from threading import Thread

import polars as pl

from .base_env import get_base_settings
from .chains import LLMChainContext, simple_data_extractor
from .processors import DataframeLLMAnalyzer, ProcessorOutput
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
        llm_logger.info("processing data %d", i)
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


def llm_processor(llm_context: LLMChainContext, input_columns: list[str], output_column_name_suffix: str = "") -> Callable[[int, pl.DataFrame], ProcessorOutput]:

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

        processed = llm_processors.get_results(
            {
                f"{f}{output_column_name_suffix}": simple_data_extractor(f)
                for f in llm_context.output_schema.model_fields.keys()
            }
        )

        llm_context.logger.info("data %d has been processed successfully", idx)
        return processed
    return info_extractor

def llm_job_assignment(input_data: pl.DataFrame, llm_context: LLMChainContext) -> ProcessorOutput:
    data_size = len(input_data)
    thread_chunk_size, thread_count = get_thread_parameters(data_size)
    llm_context.logger.info(
        "Total %d records are split into %d chunks, each chunk contain maximum %d records.",
        data_size,
        thread_count + 1 if data_size > thread_chunk_size * thread_count else thread_count,
        thread_chunk_size
    )

    input_columns = [
        field
        for _, field, _, _ in Formatter().parse(llm_context.init_prompt_template)
        if field is not None and field != "format_instructions"
    ]

    start = time.perf_counter()
    results = multi_thread_llm(
        input_data.iter_slices(thread_chunk_size),
        llm_processor(llm_context, input_columns, llm_context.output_column_name_suffix),
        max_sec_per_record=llm_context.max_seconds_per_record,
        max_timeout=llm_context.max_job_timeout,
        llm_logger=llm_context.logger
    )
    llm_context.logger.info(
        "%d records are processed in %s",
        data_size,
        readable_time(time.perf_counter() - start)
    )
    return results

