import io
import logging
import os
import time
from collections.abc import Callable, Iterator
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, cast

import polars as pl
from google.api_core.exceptions import NotFound
from google.cloud import bigquery, secretmanager
from google.cloud.bigquery import WriteDisposition

from .base_env import get_base_settings
from .processors import ProcessorOutput

P = ParamSpec("P")
DataProcessor = Callable[Concatenate[pl.DataFrame, P], ProcessorOutput]


def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=get_base_settings().project_id)


def get_available_table_name(
    base_table_name: str, client: bigquery.Client | None = None
) -> str:
    client = client or get_bq_client()
    i = 0
    while True:
        suffix = f"_{i}" if i > 0 else ""
        candidate = f"{base_table_name}{suffix}"
        try:
            client.get_table(candidate)
        except NotFound:
            return candidate
        i += 1


def bq2pl(query: str, client: bigquery.Client) -> pl.DataFrame:
    rows = client.query(query).result()
    return cast(pl.DataFrame, pl.from_arrow(rows.to_arrow()))


def pl2bq(
    data: pl.DataFrame,
    target_table: str,
    client: bigquery.Client,
    write_disposition: str = bigquery.WriteDisposition.WRITE_TRUNCATE,
) -> None:
    with io.BytesIO() as stream:
        data.write_parquet(stream)
        stream.seek(0)

        parquet_options = bigquery.ParquetOptions()
        parquet_options.enable_list_inference = True

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            parquet_options=parquet_options,
            write_disposition=write_disposition,
        )

        job = client.load_table_from_file(stream, target_table, job_config=job_config)
    job.result()


def get_thread_parameters(data_size: int) -> tuple[int, int]:
    cpu_counts = min(data_size, os.cpu_count() or 8)
    thread_count = min(data_size, cpu_counts * 6, 20)
    thread_chunk_size = data_size // thread_count

    return thread_chunk_size, thread_count


def get_secret(secret_id: str, project: str, version: str | None = None) -> str:
    secret_client = secretmanager.SecretManagerServiceClient()
    version = version or "latest"
    response = secret_client.access_secret_version(
        request={"name": f"projects/{project}/secrets/{secret_id}/versions/{version}"}
    )
    return response.payload.data.decode()


def read_data(
    query_path: str,
    base_file_path: str = "",
    logger: logging.Logger | None = None,
    **kwargs,
) -> pl.DataFrame:
    logger = logger or logging.getLogger(__name__)
    script_path = Path(query_path)
    base_path = Path(base_file_path)
    if not script_path.is_absolute():
        if base_file_path == "":
            raise ValueError(
                f"script file path '{query_path}' is a relative path, a base_file_path should be given"
            )
        script_path = (base_path.parent / script_path).resolve()
    if script_path.exists() and script_path.is_file():
        with open(script_path, "r", encoding="utf-8") as f:
            script = f.read()
            if len(kwargs) > 0:
                script = script.format(**kwargs)
            data = bq2pl(script, get_bq_client())
            if logger is not None:
                logger.info(
                    "Data has been successfully read into Polars DataFrame (%d records)",
                    len(data),
                )
            if len(data) == 0:
                raise ValueError("data is empty!")
            return data
    else:
        raise ValueError(f"{script_path} cannot be found or it is not a file")


def _rechunk(result: Any, chunk_size: int) -> Iterator[pl.DataFrame]:
    """Re-slice a BigQuery result, streamed as Arrow batches, into `chunk_size` rows."""
    buffer: pl.DataFrame | None = None
    for batch in result.to_arrow_iterable():
        part = cast(pl.DataFrame, pl.from_arrow(batch))
        buffer = part if buffer is None else pl.concat([buffer, part])
        while len(buffer) >= chunk_size:
            yield buffer.head(chunk_size)
            buffer = buffer.slice(chunk_size)
    if buffer is not None and len(buffer) > 0:
        yield buffer


def read_data_chunks(
    query_path: str,
    chunk_size: int,
    base_file_path: str = "",
    client: bigquery.Client | None = None,
    logger: logging.Logger | None = None,
    max_rows: int | None = None,
    **query_params: Any,
) -> Iterator[pl.DataFrame]:
    """Run a SQL script file and yield its result as DataFrames of `chunk_size` rows.

    The chunked counterpart of `read_data`: the result is streamed from BigQuery,
    so only about one chunk is held in memory at a time. The query runs once, on
    the first `next()`; tables written while chunks are processed (e.g. a
    process history the query anti-joins) do not change the remaining chunks.

    Args:
        query_path: SQL script path; if relative, resolved against `base_file_path`'s directory.
        base_file_path: File a relative `query_path` is resolved against (usually `__file__`).
        chunk_size: Rows per yielded DataFrame (the last one may be smaller).
        client: BigQuery client; defaults to one for the base settings' project.
        logger: Logger for progress messages.
        max_rows: Stop after this many rows in total (e.g. for test runs).
        **query_params: Values for the `str.format` placeholders in the script.

    Raises:
        ValueError: If the query returns no rows.
    """
    logger = logger or logging.getLogger(__name__)
    script_path = Path(query_path)
    if not script_path.is_absolute():
        if base_file_path == "":
            raise FileNotFoundError(
                f"'{query_path}' is a relative path, you need to provide `base_file_path`"
            )
        script_path = Path(base_file_path).parent / script_path
    script = script_path.read_text(encoding="utf-8")
    result = (client or get_bq_client()).query(script.format(**query_params)).result()
    logger.info("Query finished: %s rows to process", result.total_rows)

    yielded = 0
    for chunk in _rechunk(result, chunk_size):
        if max_rows is not None:
            chunk = chunk.head(max_rows - yielded)
        yielded += len(chunk)
        yield chunk
        if max_rows is not None and yielded >= max_rows:
            return
    if yielded == 0:
        raise ValueError("data is empty!")


def pl_append_bigquery(
    final_data: pl.DataFrame, table_name: str, logger: logging.Logger | None = None
) -> None:
    pl2bq(
        final_data,
        table_name,
        get_bq_client(),
        write_disposition=WriteDisposition.WRITE_APPEND,
    )

    if logger is not None:
        logger.info(
            "Data has been successfully write into '%s' (%d records)",
            table_name,
            len(final_data),
        )


def readable_time(tot_seconds: float) -> str:
    hours, remaining = divmod(tot_seconds, 3600)
    minutes, seconds = divmod(remaining, 60)

    result = []

    if hours > 0:
        result.append(f"{hours:.0f}")

    if minutes > 0:
        result.append(f"{minutes:.0f}")

    if seconds > 0:
        result.append(f"{seconds:.2f}")

    return ":".join(result)


def data_processing_timer(
    logger: logging.Logger | None = None,
) -> Callable[[DataProcessor[P]], DataProcessor[P]]:
    def func_wrapper(func: DataProcessor[P]) -> DataProcessor[P]:
        @wraps(func)
        def decorated(
            input_data: pl.DataFrame, *args: P.args, **kwargs: P.kwargs
        ) -> ProcessorOutput:
            start = time.time()
            output_data = func(input_data, *args, **kwargs)
            if logger is not None:
                elapsed_time = time.time() - start
                failed_count = len(output_data.failed)
                logger.log(
                    logging.WARNING if failed_count > 0 else logging.INFO,
                    """%s
        total %d records
        %d has defects
        %d failed
        early_terminated: (%s)
        output %d
        takes %s (%.2f s/records)""",
                    f"Data has been processed by LLM, but {failed_count} record(s) failed."
                    if failed_count > 0
                    else "Data has been processed successfully by LLM.",
                    len(input_data),
                    len(output_data.defects),
                    failed_count,
                    "None"
                    if len(output_data.early_terminated) == 0
                    else ", ".join(
                        [
                            f"terminate step: {step} [{output_data.early_terminated_names.get(step, '?')}], {len(d)} records"
                            for step, d in output_data.early_terminated.items()
                        ]
                    ),
                    len(output_data.completed),
                    readable_time(elapsed_time),
                    elapsed_time / len(input_data),
                )
            return output_data

        return decorated

    return func_wrapper
