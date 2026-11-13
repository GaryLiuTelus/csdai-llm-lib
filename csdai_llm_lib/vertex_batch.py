"""Vertex AI Gemini batch prediction through BigQuery tables.

A batch job reads its requests from a BigQuery input table and writes its
responses to a BigQuery output table. This module builds the request bodies,
loads them, submits and polls the job, and reads back and parses the responses.

Requires the `vertex` extra (`google-genai`).

Typical use::

    config = VertexBatchConfig(model="gemini-2.5-flash", dataset="my_batch_ds",
                               job_name="my_pipeline", job_id=JOB_ID)
    check_dataset(bq, bq.project, config.dataset, "northamerica-northeast1")
    temp_tables: list[str] = []
    results = run_structured_batch(
        client, bq,
        {f"r{i}": (system_prompt, user_prompt) for i, user_prompt in ...},
        MyOutputModel, config, label="step1", logger=logger, temp_tables=temp_tables,
    )
    ...  # store results
    drop_tables(bq, temp_tables, logger)
"""

import json
import logging
import re
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from google import genai
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.genai import types
from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

TERMINAL_STATES = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
    }
)
SUCCESS_STATES = frozenset({"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"})


@dataclass(frozen=True)
class VertexBatchConfig:
    """Settings of the batch jobs a pipeline submits.

    Attributes:
        model: Gemini model name, e.g. "gemini-2.5-flash".
        dataset: BigQuery dataset (in the BigQuery client's project) that holds
            the temporary input/output tables. Its location must be supported by
            Vertex AI batch prediction. It must already exist; see `check_dataset`.
        job_name: Prefix of the batch job display name.
        job_id: Run identifier, part of the temporary table names.
        poll_seconds: Seconds between job status checks.
        max_wait_seconds: The job is cancelled and `TimeoutError` raised after this.
        table_expiry: Expiry set on the temporary tables, so tables that are not
            dropped (e.g. after a failure) clean up on their own.
    """

    model: str
    dataset: str
    job_name: str = "llm_batch"
    job_id: str = "test"
    poll_seconds: int = 30
    max_wait_seconds: int = 24 * 3600
    table_expiry: timedelta = timedelta(days=7)


def build_request(
    system_prompt: str, user_prompt: str, schema: type[BaseModel] | None = None
) -> dict:
    """Build one Gemini batch request body (the JSON stored in the input table).

    Args:
        system_prompt: System instruction for the model.
        user_prompt: Fully formatted user message.
        schema: Pydantic model whose JSON schema constrains the JSON response;
            None for a free-text response.

    Returns:
        A GenerateContent request dict using the REST (camelCase) field names.
    """
    request: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    if schema is not None:
        request["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema.model_json_schema(),
        }
    return request


def response_text(response: Any) -> str | None:
    """Concatenate the non-thought text parts of the first candidate.

    Args:
        response: A GenerateContent response, as a dict or a JSON string.

    Returns:
        The text, or None if the response has no text part or is malformed.
    """
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return None
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return None
    texts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
    return "".join(texts) if texts else None


def parse_output(text: str | None, schema: type[M]) -> M | None:
    """Validate model output text against `schema`.

    Strips optional Markdown code fences before parsing.

    Returns:
        The validated model instance, or None if `text` is None or invalid.
    """
    if text is None:
        return None
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip()).strip()
    try:
        return schema.model_validate_json(cleaned)
    except ValidationError:
        return None


def check_dataset(
    bq: bigquery.Client, project: str, dataset: str, location: str | None = None
) -> None:
    """Check that the BigQuery dataset for batch tables exists (it is never created).

    Args:
        location: If given, the dataset must also be in this location.

    Raises:
        ValueError: The dataset does not exist, or is in another location.
    """
    dataset_id = f"{project}.{dataset}"
    try:
        ds = bq.get_dataset(dataset_id)
    except NotFound:
        raise ValueError(
            f"BigQuery dataset {dataset_id} for the batch tables does not exist"
        ) from None
    if location is not None and (ds.location or "").lower() != location.lower():
        raise ValueError(
            f"BigQuery dataset {dataset_id} is in {ds.location}, expected {location}"
        )


def set_expiry(bq: bigquery.Client, table_id: str, expiry: timedelta) -> None:
    """Set the table to expire `expiry` from now."""
    table = bq.get_table(table_id)
    table.expires = datetime.now(UTC) + expiry
    bq.update_table(table, ["expires"])


def drop_tables(
    bq: bigquery.Client, table_ids: list[str], logger: logging.Logger
) -> None:
    """Delete the temporary batch tables; a failure only warns, expiry is the backstop."""
    for table_id in table_ids:
        try:
            bq.delete_table(table_id, not_found_ok=True)
            logger.info("deleted temporary table %s", table_id)
        except Exception:
            logger.warning(
                "could not delete temporary table %s; it expires on its own",
                table_id,
                exc_info=True,
            )


def _load_requests(
    bq: bigquery.Client, requests: Mapping[str, dict], table_id: str
) -> None:
    """Stream the requests through a temporary JSONL file into `table_id`."""
    with tempfile.TemporaryFile() as jsonl:
        for k, v in requests.items():
            jsonl.write(json.dumps({"key": k, "request": v}).encode() + b"\n")
        jsonl.seek(0)
        bq.load_table_from_file(
            jsonl,
            table_id,
            job_config=bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                schema=[
                    bigquery.SchemaField("key", "STRING"),
                    bigquery.SchemaField("request", "JSON"),
                ],
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            ),
        ).result()


def run_batch(
    client: genai.Client,
    bq: bigquery.Client,
    requests: Mapping[str, dict],
    config: VertexBatchConfig,
    *,
    label: str,
    logger: logging.Logger,
    temp_tables: list[str],
) -> dict[str, str | None]:
    """Run one batch job; return the response text per request key (None = failed).

    The input and output tables are appended to `temp_tables` for the caller to
    drop (see `drop_tables`) once the results are safely stored; keep them if
    the run fails, for debugging.

    Args:
        client: Vertex AI genai client (`genai.Client(vertexai=True, ...)`).
        bq: BigQuery client; the temporary tables go to its project.
        requests: Request bodies (see `build_request`) by unique key.
        config: Model, dataset and polling settings.
        label: Short name of this step, part of the table and job names.
        logger: Logger for progress messages.
        temp_tables: Collects the temporary tables created.

    Raises:
        TimeoutError: The job ran longer than `config.max_wait_seconds` (it is cancelled).
        RuntimeError: The job ended in a state other than (partially) succeeded.
    """
    if not requests:
        return {}
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    job_id = re.sub(r"\W", "_", config.job_id)
    run_name = f"{job_id}_{label}_{stamp}"
    in_table = f"{bq.project}.{config.dataset}.batch_in_{run_name}"
    out_table = f"{bq.project}.{config.dataset}.batch_out_{run_name}"

    _load_requests(bq, requests, in_table)
    set_expiry(bq, in_table, config.table_expiry)
    temp_tables.append(in_table)
    logger.info("[%s] loaded %d requests into %s", label, len(requests), in_table)

    job = client.batches.create(
        model=config.model,
        src=f"bq://{in_table}",
        config=types.CreateBatchJobConfig(
            display_name=f"{config.job_name}-{run_name}"[:120],
            dest=f"bq://{out_table}",
        ),
    )
    assert job.name is not None
    logger.info("[%s] submitted batch job %s", label, job.name)

    started = time.monotonic()
    while job.state is None or job.state.name not in TERMINAL_STATES:
        if time.monotonic() - started > config.max_wait_seconds:
            client.batches.cancel(name=job.name)
            raise TimeoutError(f"batch job {job.name} exceeded max wait; cancelled")
        time.sleep(config.poll_seconds)
        job = client.batches.get(name=job.name)
        assert job.name is not None
    state = job.state.name  # type: ignore[union-attr]
    logger.info(
        "[%s] batch job %s finished: %s after %.0fs",
        label,
        job.name,
        state,
        time.monotonic() - started,
    )
    if state not in SUCCESS_STATES:
        raise RuntimeError(f"batch job {job.name} ended as {state}: {job.error}")

    dest_uri = (
        job.dest.bigquery_uri if job.dest and job.dest.bigquery_uri else out_table
    ).removeprefix("bq://")
    set_expiry(bq, dest_uri, config.table_expiry)
    temp_tables.append(dest_uri)
    results: dict[str, str | None] = dict.fromkeys(requests)
    for row in bq.query(f"SELECT * FROM `{dest_uri}`").result():
        if row.get("status"):
            logger.warning(
                "[%s] request %s failed: %s", label, row["key"], row["status"]
            )
            continue
        results[row["key"]] = response_text(row["response"])
    return results


def run_structured_batch(
    client: genai.Client,
    bq: bigquery.Client,
    prompts: Mapping[str, tuple[str, str]],
    schema: type[M],
    config: VertexBatchConfig,
    *,
    label: str,
    logger: logging.Logger,
    temp_tables: list[str],
) -> dict[str, M | None]:
    """Run one batch job of JSON-schema requests and parse each response.

    Args:
        prompts: (system prompt, user prompt) by unique key.
        schema: Pydantic model of the expected response.
        Others: as in `run_batch`.

    Returns:
        The parsed response per key; None if the request failed or its output
        did not validate against `schema`.
    """
    requests = {k: build_request(s, u, schema) for k, (s, u) in prompts.items()}
    texts = run_batch(
        client,
        bq,
        requests,
        config,
        label=label,
        logger=logger,
        temp_tables=temp_tables,
    )
    return {k: parse_output(t, schema) for k, t in texts.items()}


def generate_once(client: genai.Client, model: str, request: dict) -> str | None:
    """Send one batch request body as a synchronous call and return its text.

    Useful as a smoke test of the exact request shape before submitting a batch.
    """
    config = dict(request.get("generationConfig", {}))
    if "systemInstruction" in request:
        config["systemInstruction"] = request["systemInstruction"]
    resp = client.models.generate_content(
        model=model,
        contents=request["contents"],
        config=types.GenerateContentConfig.model_validate(config),
    )
    return resp.text
