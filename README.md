# FISSEQ Data Pipeline

The **FISSEQ Data Pipeline** provides a reproducible, configurable workflow for processing **FISSEQ cell profiling data**.
It handles data cleaning, normalization, harmonization, and stratified evaluation, making it easier to analyze experiments across batches and biological conditions.

## Features

### Command-line interface (CLI)

Access the pipeline with a single entry point:

```bash
fisseq-data-pipeline [validate|run|configure]
```

- **Data cleaning**: Remove problematic data from the dataset.
- **Normalization**: Z-score normalization with reusable statistics.
- **Harmonization**: Batch-effect correction using ComBat via [neuroHarmonize](https://github.com/rpomponio/neuroHarmonize).
- **Config-driven**: YAML configuration specifies feature selection, control sample queries, and metadata fields.
- **Reproducible train/test splits**:Stratified by label and batch, controlled via a random seed.

## Installation

Clone the repository and install dependencies:

This package in its current state should be considered experimental, and is thus not hosted on PyPI.
However, the package may be installed directly from Github using the command:

```bash
pip install git+https://github.com/Lilferrit/fisseq-data-pipeline.git
```

You may also clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/fisseq-data-pipeline.git
cd fisseq-data-pipeline
pip install -e .
```

## Quick start

Write a default configuration file:

```bash
fisseq-data-pipeline configure
```

Run validation on your dataset:

```bash
fisseq-data-pipeline validate \
  --input_data_path data.parquet \
  --config config.yaml \
  --output_dir results
```

Adjust logging level via environment variable:

```bash
FISSEQ_PIPELINE_LOG_LEVEL=debug fisseq-data-pipeline validate \
  --input_data_path data.parquet
```

## Documentation

## License

[MIT](LICENSE.txt)
