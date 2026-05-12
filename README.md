# FISSEQ Data Pipeline

The **FISSEQ Data Pipeline** provides a reproducible, configurable workflow for processing **FISSEQ cell profiling data**.
It handles z-score normalization on control samples and per-variant feature aggregation, making it straightforward to compare variants across experiments.

---

## Features

- **Normalization**: Z-score normalization fit on WT control cells and applied across the full dataset. Statistics serialized as Parquet for later reuse.
- **Aggregation**: Per-variant summary statistics using native Polars expressions (mean, median, MAD, std) or reference-distribution comparisons (EMD, KS, Q-Q correlation, AUROC). Synonymous variants are used as the aggregation-level normalization baseline.
- **Hydra-driven**: Each step is a standalone Hydra entry point. Any config field can be overridden on the command line.

---

## Installation

This package is experimental and not hosted on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/Lilferrit/fisseq-data-pipeline.git
```

Or clone and install locally:

```bash
git clone https://github.com/Lilferrit/fisseq-data-pipeline.git
cd fisseq-data-pipeline
pip install -e .
```

---

## Quick start

### Step 1 — Normalize cell-level data

Fit z-score statistics on WT control cells and apply to the full dataset:

```bash
fisseq_normalize \
    output_dir=./out \
    input_file=data/cells.parquet
```

Output: `out/cells.parquet` with normalized feature columns and an added `meta_is_control` boolean column.

### Step 2 — Aggregate to per-variant statistics

Summarize cell-level data to one row per variant, then normalize to synonymous baseline:

```bash
fisseq_aggregate \
    output_dir=./out \
    input_file=out/cells.parquet
```

Output: `out/cells.parquet` with one row per non-synonymous variant and aggregate feature statistics z-scored to synonymous variants.

---

## Configuration

Both entry points share a common set of base fields (`output_dir`, `output_root`, `log_level`) and each adds its own:

### `normalize`

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to input parquet file. |
| `control_sample_query` | `"meta_aa_changes = 'WT'"` | SQL WHERE clause identifying control rows. |
| `save_normalizer` | `true` | Write fitted normalizer to `normalizer.parquet`. |

### `aggregate`

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to cell-level normalized parquet file. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `aggregator` | `"multi"` | Aggregation method (see table below). |
| `save_normalizer` | `true` | Write synonymous-baseline normalizer to `normalizer.parquet`. |

### Aggregators

| Value | Description |
| ----- | ----------- |
| `mean` | Per-variant feature mean |
| `median` | Per-variant feature median |
| `MAD` | Per-variant median absolute deviation |
| `std` | Per-variant standard deviation |
| `EMD` | Earth Mover's Distance vs. control distribution |
| `KS` | Kolmogorov-Smirnov statistic vs. control distribution |
| `QQ` | Q-Q Pearson correlation vs. control distribution |
| `AUROC` | AUROC vs. control distribution |
| `multi` | All of the above except EMD (default) |

---

## Typical workflow

```bash
# Normalize
fisseq_normalize \
    output_dir=./out \
    input_file=data/cells.parquet

# Aggregate (default: all non-EMD statistics)
fisseq_aggregate \
    output_dir=./out \
    input_file=out/cells.parquet

# Aggregate with a single method
fisseq_aggregate \
    output_dir=./out \
    input_file=out/cells.parquet \
    aggregator=KS
```

### Custom control query or label column

```bash
fisseq_normalize \
    output_dir=./out \
    input_file=data/cells.parquet \
    control_sample_query="meta_treatment = 'DMSO'"

fisseq_aggregate \
    output_dir=./out \
    input_file=out/cells.parquet \
    label_column=meta_treatment
```

### Named output roots

Use `output_root` to prefix all output files from a run (useful when running multiple configurations into the same directory):

```bash
fisseq_normalize \
    output_dir=./out \
    output_root=run1 \
    input_file=data/cells.parquet

# Produces: out/run1.cells.parquet, out/run1.normalizer.parquet, out/run1.normalize.log
```

---

## Logging

Log level is controlled via the `log_level` config field (default: `info`). Logs are written to both stdout and a `.log` file in `output_dir`.

```bash
fisseq_normalize \
    output_dir=./out \
    input_file=data/cells.parquet \
    log_level=debug
```

---

## Documentation

Full API documentation: [https://lilferrit.github.io/fisseq-data-pipeline/](https://lilferrit.github.io/fisseq-data-pipeline)

## License

[MIT](LICENSE.txt)
