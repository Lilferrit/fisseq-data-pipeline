# FISSEQ Data Pipeline

A Nextflow + Python workflow for processing single-cell CellProfiler morphological
profiling data from FISSEQ (Fluorescence In-Situ Sequencing) experiments. Each cell
carries a genetic variant label; the pipeline measures how each variant's cell
population differs from wildtype (WT) controls using morphological features.

## Where to start

- **[Architecture](architecture.md)** — the full pipeline DAG, what each stage
  produces/consumes, and the key shared abstractions (`Normalizer`,
  `BaseAggregator`, `BatchCorrector`).
- **[Installation](installation.md)** — environment setup, including cluster/HPC
  configuration.
- **[Nextflow Workflow](nextflow.md)** — the Nextflow processes, how they're wired
  together, and how to run the pipeline (params, profiles).
- **CLI Reference** — one page per Python entry point (QC filter, normalize,
  aggregate, feature selection, batch correction, ANOVA, OvWT, OvWT cell
  scores, batch-vs-batch), each with its config fields and a runnable example.
- **API Reference** — full function/class-level documentation for every module,
  generated from source docstrings.
- **[Walkthrough](walkthrough.md)** — a complete end-to-end run, from raw
  CellProfiler output to final feature-selected results.

## Pipeline at a glance

```text
input/*.parquet  (one file per batch)
     │
     ▼
QC_FILTER   (per batch)
     │
     ├──► BATCHVSBATCH (pre)        (global, optional — params.global)
     ▼
NORMALIZE   (per batch)
     │
     ├──► BATCHVSBATCH (post)       (global, optional — params.global)
     ├──► OVWT_BATCHWISE             (per batch)
     ├──► OVWT_GLOBAL                (global, optional — params.global)
     ├──► Feature selection          (batchwise always; global optional)
     └──► ANOVA (normalized)         (global — always runs)

QC_FILTER ──► BATCH_CORRECT_FIT ──► BATCH_CORRECT_TRANSFORM ──► ANOVA (batch-corrected)  (always runs)
```

See [Architecture](architecture.md) for the full diagram and stage-by-stage detail.

For a repo overview and quick start, see the
[README](https://github.com/Lilferrit/fisseq-data-pipeline).
