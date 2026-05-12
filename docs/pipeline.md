# Running the pipeline

The FISSEQ pipeline consists of two sequential Hydra entry points. Each can also be run independently.

---

## Step 1 — Normalize

Fit z-score normalization statistics on WT control cells and apply them to the full dataset:

```bash
python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    input_file=data/cells.parquet
```

Output: `out/cells.parquet` with z-scored feature columns and an added `meta_is_control` boolean column.

---

## Step 2 — Aggregate

Aggregate cell-level data to per-variant statistics, then normalize to synonymous variant baseline:

```bash
python -m fisseq_data_pipeline.aggregate \
    output_dir=./out \
    input_file=out/cells.parquet
```

Output: `out/cells.parquet` with one row per non-synonymous variant. Feature columns are aggregate statistics (e.g. mean, KS, AUROC) z-scored to synonymous variants.

---

## Using a custom label or control column

Override `control_sample_query` in the normalize step and `label_column` in the aggregate step:

```bash
python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    input_file=data/cells.parquet \
    control_sample_query="meta_treatment = 'DMSO'"

python -m fisseq_data_pipeline.aggregate \
    output_dir=./out \
    input_file=out/cells.parquet \
    label_column=meta_treatment
```

---

## Named output roots

Use `output_root` to prefix all output files from a run:

```bash
python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    output_root=run1 \
    input_file=data/cells.parquet

# Produces: out/run1.cells.parquet, out/run1.normalizer.parquet, out/run1.normalize.log
```

---

## Logging

Logs are written to both stdout and a `.log` file in `output_dir`. Control verbosity with `log_level`:

```bash
python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    input_file=data/cells.parquet \
    log_level=debug
```

---

For all available config options see [Configuration](./configuration.md).
