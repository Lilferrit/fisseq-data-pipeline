# FISSEQ Data Pipeline

Welcome to the **FISSEQ Data Pipeline** documentation.

## Features

### Command-line interface (CLI)

Access the pipeline with a single entry point:

```bash
  fisseq-data-pipeline [validate|run|configure]
```

For more details on command line usage see [Pipeline](./pipeline.md).

### Data cleaning

Remove invalid rows/columns and rare label–batch pairs, namely columns that contain all NaN values, followed by rows that contain any remaining NaN values.

See [Filter](./filter.md).

### Normalization

Compute z-score normalization statistics on control samples and apply them across the dataset.
By default z-score normalization is fit only to control samples.
Fitting only to control samples ensures that biological variation is captured even in the case where biological covariants are largely disjoint across batches.

See [Normalize](./normalize.md).

### Harmonization

The harmonization step applies batch correction using the `neuroHarmonize` library.
Similar to the normalization stage, by default the harmonizer's batch parameters are fit just the control (wild type) samples in the input data, and these parameters are then used to apply batch correction to all samples.

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
