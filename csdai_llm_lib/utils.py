import io
import os
import time
import logging
from pathlib import Path
from typing import cast, Optional, Callable
from functools import wraps

import polars as pl
from google.cloud import bigquery, secretmanager
from google.cloud.bigquery import WriteDisposition
from google.api_core.exceptions import NotFound

from .base_env import get_base_settings
from .processors import ProcessorOutput

DataProcessor = Callable[[pl.DataFrame], ProcessorOutput]

def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=get_base_settings().project_id)
def get_available_table_name(base_table_name: str, client: bigquery.Client | None = None) -> str:
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
        write_disposition: str = bigquery.WriteDisposition.WRITE_TRUNCATE
    ) -> None:
    with io.BytesIO() as stream:
        data.write_parquet(stream)
        stream.seek(0)

        parquet_options = bigquery.ParquetOptions()
        parquet_options.enable_list_inference = True

        job_config = bigquery.LoadJobConfig(
            source_format = bigquery.SourceFormat.PARQUET,
            parquet_options=parquet_options,
            write_disposition=write_disposition
        )

        job = client.load_table_from_file(
            stream,
            target_table,
            job_config=job_config
        )
    job.result()

def get_thread_parameters(data_size: int) -> tuple[int, int]:
    cpu_counts = min(data_size, os.cpu_count() or 8)
    thread_count = min(data_size, cpu_counts * 6, 20)
    thread_chunk_size = data_size // thread_count

    return thread_chunk_size, thread_count

def get_secret(secret_id: str, project: str, version: Optional[str] = None) -> str:
    secret_client = secretmanager.SecretManagerServiceClient()
    version = version or "latest"
    response = secret_client.access_secret_version(
        request={
            "name": f"projects/{project}/secrets/{secret_id}/versions/{version}"
        }
    )
    return response.payload.data.decode()

def read_data(query_path: str, base_file_path: str = "", logger: Optional[logging.Logger]=None, **kwargs) -> pl.DataFrame:
    logger = logger or logging.getLogger(__file__)
    script_path = Path(query_path)
    base_path = Path(base_file_path)
    if not script_path.is_absolute():
        if base_file_path == "":
            raise ValueError(f"script file path '{query_path}' is a relative path, a base_file_path should be given")
        script_path = (base_path.parent / script_path).resolve()
    if script_path.exists() and script_path.is_file():
        with open(script_path, "r", encoding="utf-8") as f:
            script = f.read()
            if len(kwargs) > 0:
                script = script.format(**kwargs)
            data = bq2pl(
                script,
                get_bq_client()
            )
            if logger is not None:
                logger.info(
                    "Data has been successfully read into Polars DataFrame (%d records)",
                    len(data)
                )
            if len(data) == 0:
                raise ValueError("data is empty!")
            return data
    else:
        raise ValueError(f"{script_path} cannot be found or it is not a file")
def pl_append_bigquery(final_data: pl.DataFrame, table_name: str, logger:logging.Logger | None = None) -> None:
    pl2bq(
        final_data,
        table_name,
        get_bq_client(),
        write_disposition=WriteDisposition.WRITE_APPEND
    )

    if logger is not None:
        logger.info(
                "Data has been successfully write into '%s' (%d records)",
                table_name,
                len(final_data)
            )
def readable_time(tot_seconds: float) -> str:
    hours, remaining = divmod(tot_seconds, 3600)
    minutes, seconds = divmod(remaining, 60)

    result = ""
    if hours > 0:
        result += f"{hours:.0f} hour(s)"
    if minutes > 0:
        result += f" {minutes:.0f} minute(s)"
    if seconds > 0:
        result += f" {seconds:.2f} second(s)"
    return result.strip()
def data_processing_timer(logger: logging.Logger | None = None) -> Callable[[DataProcessor], DataProcessor]:
    def func_wrapper(func: DataProcessor) -> DataProcessor:
        @wraps(func)
        def decorated(input_data: pl.DataFrame) -> ProcessorOutput:
            start = time.time()
            output_data = func(input_data)
            if logger is not None:
                logger.info(
                        """Data has been processed successfully by LLM.
    total %d records
    %d has defects
    %d failed parsing
    early_terminated: (%s)
    output %d
    takes %s""",
                        len(input_data),
                        len(output_data.defects),
                        len(output_data.failed),
                        "None" if len(output_data.early_terminated) == 0 else ", ".join([f"terminate step: {step}, {len(d)} records" for step, d in output_data.early_terminated.items()]),
                        len(output_data.completed),
                        readable_time(time.time() - start)
                    )
            return output_data
        return decorated
    return func_wrapper
