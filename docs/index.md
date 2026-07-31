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
- **[Configuration](configuration.md)** — every parameter, the `pipeline_dir`
  layout, per-batch YAML overrides, and global groups.
- **CLI Reference** — one page per Python entry point (QC filter, normalize,
  aggregate, feature selection, batch correction, ANOVA, OvWT, OvWT cell
  scores, batch-vs-batch, wildtype-vs-wildtype), each with its config fields
  and a runnable example.
- **API Reference** — full function/class-level documentation for every module,
  generated from source docstrings.
- **[Walkthrough](walkthrough.md)** — a complete end-to-end run, from raw
  CellProfiler output to final feature-selected results.

## Pipeline at a glance

```text
configs/*.yaml  (mandatory, one file per batch) ──► INPUT ──► input/*.parquet
     │
     ▼
QC_FILTER   (per batch)
     │
     ├──► BATCHVSBATCH (pre)        (once per active group — params.global_groups)
     ▼
NORMALIZE   (per batch)
     │
     ├──► BATCHVSBATCH (post)       (once per active group)
     ├──► OVWT_BATCHWISE             (per batch)
     ├──► OVWT_GLOBAL                (once per active group)
     ├──► WTVWT_BATCHWISE            (per batch, optional — params.run_wtvwt)
     ├──► Feature selection          (batchwise always; global sub-branch once per active group)
     └──► ANOVA (normalized)         (global — always runs)

QC_FILTER ──► BATCH_CORRECT_FIT ──► BATCH_CORRECT_TRANSFORM ──► ANOVA (batch-corrected)  (always runs)
```

See [Architecture](architecture.md) for the full diagram and stage-by-stage detail.

For a repo overview and quick start, see the
[README](https://github.com/Lilferrit/fisseq-data-pipeline).
