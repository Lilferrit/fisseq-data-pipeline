# FISSEQ Data Pipeline

A Nextflow + Python workflow for processing single-cell CellProfiler morphological
profiling data from FISSEQ (Fluorescence In-Situ Sequencing) experiments. Each cell
carries a genetic variant label; the pipeline measures how each variant's cell
population differs from wildtype (WT) controls using morphological features.

## Where to start

- **[Quickstart](quickstart.md)** — the fastest path from a fresh checkout to
  a first pipeline run, including setting up a cluster config.
- **[Architecture](architecture.md)** — the full pipeline DAG, what each stage
  produces/consumes, and the key shared abstractions (`Normalizer`,
  `BaseAggregator`, `BatchCorrector`).
- **[Installation](installation.md)** — environment setup, including cluster/HPC
  configuration.
- **[Nextflow Workflow](nextflow.md)** — the Nextflow processes, how they're wired
  together, and how to run the pipeline (profiles).
- **[Configuration](configuration.md)** — every parameter, the `experiments:`
  schema, the single `random_seed`, and global channels.
- **CLI Reference** — one page per Python entry point (input, QC filter,
  normalize, aggregate, feature selection, global feature selection, OvWT,
  global OvWT), each with its config fields and a runnable example.
- **API Reference** — full function/class-level documentation for every module,
  generated from source docstrings.
- **[Walkthrough](walkthrough.md)** — a complete end-to-end run, from raw
  CellProfiler output to final feature-selected results.

## Pipeline at a glance

```text
params.yaml (experiments: [...])  ──►  INPUT  ──►  input/<batch_stem>.parquet
     │
     ▼
QC_FILTER   (per experiment)
     │
     ▼
NORMALIZE   (per experiment)          z-score fit on WT control cells
     │
     ├──► OVWT_BATCHWISE               (per experiment — params.run_ovwt)
     │       └──► GLOBAL_OVWT          (once per active global channel)
     │
     └──► Feature selection            (per experiment — params.run_feature_selection)
             └──► GLOBAL_FEATURE_SELECT (once per active global channel)
```

See [Architecture](architecture.md) for the full diagram and stage-by-stage detail.

For a repo overview and quick start, see the
[README](https://github.com/Lilferrit/fisseq-data-pipeline).
