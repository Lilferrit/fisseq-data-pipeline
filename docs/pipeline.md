# Pipeline

The FISSEQ data pipeline exposes a small CLI via the entry point
``fisseq-data-pipeline``. Subcommands are provided by
[Python Fire](https://github.com/google/python-fire):

- `validate` — Train/validate on a stratified split and write outputs.
- `run` — Production, single-pass run (not yet implemented).
- `configure` — Write a default configuration file.

## Quick start

```bash
# validate with explicit config and output directory
fisseq-data-pipeline validate \
  --input_data_path data.parquet \
  --config config.yaml \
  --output_dir out \
  --test_size 0.2 \
  --write_train_results true
```

## Write a default config to the current directory

```bash
fisseq-data-pipeline configure
```

## Logging

```bash
FISSEQ_PIPELINE_LOG_LEVEL=debug fisseq-data-pipeline validate \
  --input_data_path data.parquet
```

## Command Interface

---

### Validate

::: fisseq_data_pipeline.pipeline.validate

---

### Run

::: fisseq_data_pipeline.pipeline.run
options:
    show_signature: true
    show_signature_annotations: true
    show_source: true

---

### Configure

::: fisseq_data_pipeline.pipeline.configure
options:
    show_signature: true
    show_signature_annotations: true
    show_source: true

---

## Auxiliary functions

This functions are not exposed to the command line, and are for internal use only.

---

::: fisseq_data_pipeline.pipeline.setup_logging

---

::: fisseq_data_pipeline.pipeline.main

---
