# csdai-llm-lib

CSD AI team LLM library.

## Vertex AI Gemini batch prediction (`csdai_llm_lib.vertex_batch`)

Install with the `vertex` extra:

```toml
dependencies = ["csdai-llm-lib[vertex]"]

[tool.uv.sources]
csdai-llm-lib = { git = "https://github.com/GaryLiuTelus/csdai-llm-lib.git", tag = "v1.12.0" }
```

```python
from csdai_llm_lib.utils import read_data_chunks
from csdai_llm_lib.vertex_batch import (
    VertexBatchConfig, drop_tables, check_dataset, run_structured_batch,
)

config = VertexBatchConfig(model="gemini-2.5-flash", dataset="my_batch_ds",
                           job_name="my_pipeline", job_id=JOB_ID)
check_dataset(bq, bq.project, config.dataset, "northamerica-northeast1")

for chunk in read_data_chunks("query.sql", __file__, 5000, client=bq, logger=logger):
    temp_tables: list[str] = []
    results = run_structured_batch(          # {key: MyOutput | None}
        genai_client, bq,
        {f"r{i}": (system_prompt, template.format(**row))
         for i, row in enumerate(chunk.iter_rows(named=True))},
        MyOutput, config, label="step1", logger=logger, temp_tables=temp_tables,
    )
    ...                                      # store results
    drop_tables(bq, temp_tables, logger)     # kept on failure, for debugging
```

- `genai_client` is `genai.Client(vertexai=True, project=..., location=...)`.
- A `None` result means the request failed or its output did not match the schema; retry it in a later run.
- Temporary input/output tables expire after `config.table_expiry` (7 days) even if not dropped.
- Lower-level pieces: `build_request`, `run_batch` (raw text per key), `parse_output`, and
  `generate_once` (sends one batch request body synchronously, as a smoke test).
