# FISSEQ Data Pipeline

Welcome to the **FISSEQ Data Pipeline** documentation.

## Overview

The pipeline processes cell-level FISSEQ imaging feature matrices through two sequential steps:

1. **Normalization** — Fit z-score statistics on wild-type (WT) control cells and apply them across the full dataset.
2. **Aggregation** — Summarize cell-level features into per-variant statistics, then normalize to a synonymous variant baseline.

Each step is a standalone [Hydra](https://hydra.cc) entry point and can be run independently or in sequence.

## Quick start

### Step 1 — Normalize cell-level data

```bash
python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    input_file=data/cells.parquet
```

### Step 2 — Aggregate to per-variant statistics

```bash
python -m fisseq_data_pipeline.aggregate \
    output_dir=./out \
    input_file=out/cells.parquet
```

For a full walkthrough see [Running the pipeline](./pipeline.md).

## Installation

This package is not yet hosted on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/Lilferrit/fisseq-data-pipeline.git
```

Or clone and install locally:

```bash
git clone https://github.com/Lilferrit/fisseq-data-pipeline.git
cd fisseq-data-pipeline
pip install -e .
```

## Modules

- [Normalize](./normalize.md) — Z-score normalization fit on control samples.
- [Aggregate](./aggregate.md) — Per-variant feature aggregation with synonymous baseline normalization.
- [Configuration](./configuration.md) — All config fields for each entry point.
- [Running the pipeline](./pipeline.md) — End-to-end walkthrough.
- [Utilities](./utils.md) — Shared logging setup.
