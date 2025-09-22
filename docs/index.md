# FISSEQ Data Pipeline

Welcome to the **FISSEQ Data Pipeline** documentation.  
This project provides a reproducible, configurable pipeline for processing FISSEQ cell profiling data, including cleaning, normalization, harmonization, and stratified evaluation.

## Features

### Command-line interface (CLI)

Access the pipeline with a single entry point:

```bash
  fisseq-data-pipeline [validate|run|configure]
```

For more details on command line usage see [Pipeline](./pipeline.md).

### Data cleaning

Remove invalid rows/columns and rare label–batch pairs.
See [Filter](./filter.md).

### Normalization

Compute z-score normalization statistics on control samples and apply them across the dataset.
See [Normalize](./normalize.md).

### Harmonization

Apply ComBat batch correction via `neuroHarmonize.`
See [Harmonize](./harmonize.md).

## Installation

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

### Running the Pipeline

After installation the pipeline can be run from the command line.
For more details see [Pipeline](./pipeline.md).

### Configuration

The pipeline may be configured using a yaml configuration file.
For more details see [Configuration](./configuration.md).
