# FISSEQ Data Pipeline


The **FISSEQ Data Pipeline** is a command-line tool designed to perform normalization and harmonization of cell-level data extracted from **FISSEQ** experiments.

## Usage


The CLI is powered by [Python Fire](https://github.com/google/python-fire). The entry points are `validate`, `run`, and `configure`.

## Installation

This package in its current state should be considered experimental, and is thus not hosted on PyPI.
However, the package may be installed directly from Github using the command:

```bash
pip install git+https://github.com/Lilferrit/fisseq-data-pipeline.git
```

### 1. Validate with Cross-Validation


```bash
python -m fisseq_data_pipeline validate \
--input_data_path /path/to/cells.parquet \
--config /path/to/config.yaml \
--output_dir ./results/ \
--n_folds 5
```


**Outputs (per fold):**
- `unmodified.fold_00001.parquet`
- `normalized.fold_00001.parquet`
- `harmonized.fold_00001.parquet`
- `normalizer.fold_00001.pkl`
- `harmonizer.fold_00001.pkl`


### 2. Run on Full Dataset


```bash
python -m fisseq_data_pipeline run \
--input_data_path /path/to/cells.parquet \
--config /path/to/config.yaml \
--output_dir ./results/
```


**Outputs:**
- `normalized.parquet`
- `harmonized.parquet`
- `normalizer.pkl`
- `harmonizer.pkl`


### 3. Configure

Get a template configuration file:


```bash
python -m fisseq_data_pipeline configure --output_path ./config.yaml
```

## Configuration


The pipeline expects a YAML configuration file with at least the following keys:


```yaml
batch_col_name: batch
label_col_name: label
feature_cols: ^f_ # regex or explicit list of features
control_sample_query: "batch = 'control'"
```

## Logging


Logs are written both to console and to a timestamped log file in the chosen output directory:


```
fisseq-data-pipeline-YYYYMMDD:HHMMSS.log
```


The log level defaults to `info`, but can be set via the `FISSEQ_PIPELINE_LOG_LEVEL` environmental variable.

```bash
export FISSEQ_PIPELINE_LOG_LEVEL=debug
```