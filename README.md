# FISSEQ Data Pipeline

A Nextflow + Python workflow for processing single-cell CellProfiler morphological
profiling data from FISSEQ (Fluorescence In-Situ Sequencing) experiments.

## Overview

Each cell carries a genetic variant label; the pipeline measures how each
variant's cell population differs from wildtype (WT) controls using
morphological features. The high-level shape:

```text
raw CellProfiler features -> QC filtering -> WT normalization
    -> one-vs-WT cross-validated scoring -> cross-experiment score aggregation
    -> bootstrap feature selection -> cross-experiment feature selection
```

Every parameter lives in `params.yaml`, including the single `random_seed` that
drives every stochastic step, and the `experiments:` list that declares what to
run over.

## Quick start

Install the environment ([uv](https://docs.astral.sh/uv/)-managed):

```bash
git clone https://github.com/Lilferrit/fisseq-data-pipeline.git
cd fisseq-data-pipeline
uv sync --group dev
```

Run the full pipeline end to end with [Nextflow](https://www.nextflow.io/) (≥ 26.04).
`-params-file` is required — `nextflow.config` holds no parameter defaults:

```bash
nextflow run Lilferrit/fisseq-data-pipeline \
    -params-file params.yaml \
    --pipeline_dir /path/to/experiment
```

Every process runs in a published container image
(`ghcr.io/lilferrit/fisseq-data-pipeline`); add `-profile local` to run against a
local uv environment instead.

## Documentation

See the **[Quickstart Guide](https://lilferrit.github.io/fisseq-data-pipeline/quickstart/)**
for a full walkthrough of getting your first run going, including cluster/SGE
setup.

Full documentation — architecture, Nextflow workflow reference, CLI/config
options, and an end-to-end walkthrough — is at
**[lilferrit.github.io/fisseq-data-pipeline](https://lilferrit.github.io/fisseq-data-pipeline)**.

## License

[MIT](LICENSE.txt)
