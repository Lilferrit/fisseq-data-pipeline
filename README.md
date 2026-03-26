# FISSEQ Data Pipeline

The **FISSEQ Data Pipeline** provides a reproducible, configurable workflow for processing **FISSEQ cell profiling data**.
It handles data cleaning, normalization, and per-(batch, label) aggregation, making it easier to analyze experiments across batches and biological conditions.

---

## Features

- **Data cleaning**: Remove all-non-finite columns and rows with any non-finite feature value.
- **Normalization**: Batch-wise z-score normalization fit on control samples, with serialized statistics for later reuse.
- **Aggregation**: Per-(batch, label) summary statistics using native Polars expressions (mean, median, MAD, std) or reference-distribution comparisons (EMD, KS, Q-Q correlation, AUROC).
- **Config-driven**: YAML configuration specifies feature selection, control sample queries, and metadata fields.

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

## Configuration

The pipeline is configured with a YAML file. To generate a default config in your working directory:

```bash
fisseq-data-pipeline configure

# Or write it to a specific location
fisseq-data-pipeline configure --output_path path/to/config.yaml
```

The default configuration looks like:

```yaml
# Regex or list of column names to select feature columns.
# The default matches CellProfiler outputs (uppercase start, contains underscore).
feature_cols: "^[A-Z][A-Za-z0-9]*_.*"

# SQL-like WHERE clause to identify control samples.
control_sample_query: "variantClass = 'WT'"

# Column containing batch identifiers (e.g. well, experiment, or run).
batch_col_name: "tile_experiment_well"

# Column containing biological labels (e.g. variant or treatment).
label_col_name: "aaChanges"
```

`feature_cols` can be a **regex string** to match column names, or an **explicit list** of column names:

```yaml
# Regex (selects all columns starting with an uppercase letter followed by underscore)
feature_cols: "^[A-Z][A-Za-z0-9]*_.*"

# Explicit list
feature_cols:
  - Intensity_MeanIntensity_DAPI
  - Intensity_StdIntensity_DAPI
  - Texture_Correlation_DAPI
```

---

## Pipeline CLI

The main pipeline entry point is `fisseq-data-pipeline`. It runs data cleaning and normalization in a single pass and writes outputs to disk.

### `run`

```bash
fisseq-data-pipeline run \
  --input_data_path data.parquet \
  --config config.yaml \
  --output_dir results/
```

| Argument | Description | Default |
| --- | --- | --- |
| `--input_data_path` | Path to input Parquet file | required |
| `--config` | Path to YAML config, or omit to use defaults | `None` |
| `--output_dir` | Directory to write outputs | current directory |
| `--eager_db_loading` | Load full dataset into memory up front | `False` |

**Outputs** written to `output_dir`:

| File | Description |
| --- | --- |
| `data-cleaned.parquet` | Feature matrix after cleaning |
| `normalized.parquet` | Z-score normalized feature matrix |
| `normalizer.pkl` | Serialized `Normalizer` object for reuse |

### `configure`

```bash
# Write config.yaml to the current directory
fisseq-data-pipeline configure

# Write to a custom path
fisseq-data-pipeline configure --output_path experiments/exp1/config.yaml
```

---

## Aggregation CLI

After normalization, compute per-(batch, label) summary statistics using the aggregation module.

### `compute`

Reads a normalized Parquet file and writes `aggregated.parquet` containing only the aggregate feature columns (one row per (batch, label) group).

```bash
python -m fisseq_data_pipeline.aggregate compute \
  --norm_df results/normalized.parquet \
  --out_dir results/ \
  --aggregator mean
```

| Argument | Description |
| --- | --- |
| `--norm_df` | Path to normalized Parquet file |
| `--out_dir` | Directory to write outputs |
| `--aggregator` | Aggregation method (see table below) |

Available aggregators:

| Value | Description | Type |
| --- | --- | --- |
| `mean` | Per-group feature mean | Native |
| `median` | Per-group feature median | Native |
| `MAD` | Per-group median absolute deviation | Native |
| `std` | Per-group feature standard deviation | Native |
| `EMD` | Earth Mover's Distance vs. control distribution | Reference |
| `KS` | Kolmogorov-Smirnov statistic vs. control distribution | Reference |
| `QQ` | Q-Q Pearson correlation vs. control distribution | Reference |
| `AUROC` | Area under the ROC curve vs. control distribution | Reference |

Reference-based aggregators (`EMD`, `KS`, `QQ`, `AUROC`) compare each (batch, label) group against the control rows for the same batch.

### `normalize`

Normalize an aggregate DataFrame to synonymous-variant rows, fitting and applying a normalizer. Outputs `normalized.parquet` and `normalizer.pkl`.

```bash
python -m fisseq_data_pipeline.aggregate normalize \
  --agg_df results/aggregated.parquet \
  --out_dir results/
```

---

## Logging

Log level is controlled via the `FISSEQ_PIPELINE_LOG_LEVEL` environment variable. Accepted values: `debug`, `info`, `warning`, `error`, `critical`. Default: `info`.

```bash
FISSEQ_PIPELINE_LOG_LEVEL=debug fisseq-data-pipeline run \
  --input_data_path data.parquet
```

Log files are written to the output directory with a timestamped filename: `fisseq-data-pipeline-YYYYMMDD:HHMMSS.log`.

---

## Typical workflow

```bash
# 1. Generate a config
fisseq-data-pipeline configure --output_path config.yaml

# 2. Edit config.yaml to match your dataset's column names

# 3. Run the main pipeline
fisseq-data-pipeline run \
  --input_data_path data.parquet \
  --config config.yaml \
  --output_dir results/

# 4. Compute aggregation statistics
python -m fisseq_data_pipeline.aggregate compute \
  --norm_df results/normalized.parquet \
  --out_dir results/ \
  --aggregator EMD

# 5. Normalize aggregates to synonymous variants
python -m fisseq_data_pipeline.aggregate normalize \
  --agg_df results/aggregated.parquet \
  --out_dir results/
```

---

## Documentation

Full API documentation: [https://lilferrit.github.io/fisseq-data-pipeline/](https://lilferrit.github.io/fisseq-data-pipeline)

## License

[MIT](LICENSE.txt)
