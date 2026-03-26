# Pipeline

The FISSEQ data pipeline exposes a small CLI via the entry point
``fisseq-data-pipeline``. Subcommands are provided by
[Python Fire](https://github.com/google/python-fire):

- `run` — Production, single-pass run: clean, normalize, and write outputs.
- `configure` — Write a default configuration file.

## Quick start

```bash
# Run the full pipeline
fisseq-data-pipeline run \
  --input_data_path data.parquet \
  --config config.yaml \
  --output_dir out
```

## Write a default config to the current directory

```bash
fisseq-data-pipeline configure

# Write to a custom location
fisseq-data-pipeline configure --output_path path/to/config.yaml
```

## Logging

Log level is controlled via the `FISSEQ_PIPELINE_LOG_LEVEL` environment
variable (default: `info`).

```bash
FISSEQ_PIPELINE_LOG_LEVEL=debug fisseq-data-pipeline run \
  --input_data_path data.parquet
```

## Command Interface

---

### Run

::: fisseq_data_pipeline.pipeline.run

---

### Configure

::: fisseq_data_pipeline.pipeline.configure

---
