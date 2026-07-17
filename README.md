# FISSEQ Data Pipeline

A Nextflow + Python workflow for processing single-cell CellProfiler morphological
profiling data from FISSEQ (Fluorescence In-Situ Sequencing) experiments.

## Overview

Each cell carries a genetic variant label; the pipeline measures how each
variant's cell population differs from wildtype (WT) controls using
morphological features. The high-level shape:

```text
cell-level features -> QC filtering -> normalization -> batch-effect checks
    -> one-vs-WT classification -> bootstrap feature selection
    -> batch correction -> PERMANOVA batch-effect testing
```

## Quick start

Install the environment ([uv](https://docs.astral.sh/uv/)-managed):

```bash
git clone https://github.com/Lilferrit/fisseq-data-pipeline.git
cd fisseq-data-pipeline
uv sync --group dev
```

Or install the package straight from GitHub with `pip` (no clone needed):

```bash
pip install git+https://github.com/Lilferrit/fisseq-data-pipeline.git
```

Run the full pipeline end to end with [Nextflow](https://www.nextflow.io/) (≥ 23.10):

```bash
nextflow run Lilferrit/fisseq-data-pipeline -c your.config --input_dir /path/to/experiment
```

## Documentation

Full documentation — architecture, Nextflow workflow reference, CLI/config
options, and an end-to-end walkthrough — is at
**[lilferrit.github.io/fisseq-data-pipeline](https://lilferrit.github.io/fisseq-data-pipeline)**.

## License

[MIT](LICENSE.txt)
