# Walkthrough: running the pipeline end to end

This walks through a full run of the default `fisseq` workflow, from raw
CellProfiler output to final feature-selected results.

## 1. Install

```bash
git clone https://github.com/Lilferrit/fisseq-data-pipeline.git
cd fisseq-data-pipeline
uv sync --group dev
```

See [Installation](installation.md) for details, including cluster/HPC setup.

## 2. Lay out your input data

Every batch is declared by a YAML config file under `<pipeline_dir>/configs/`
naming its raw CellProfiler feature matrix (or matrices) — there is no mode
where the pipeline scans a directory of pre-staged Parquet files directly.
See [Configuration](configuration.md#pipeline-directory-layout) for the full
layout and [CLI Reference: Input](cli/input.md#config_path-yaml-schema) for
the config schema:

```text
<pipeline_dir>/
  configs/
    batch1.yaml   # input_paths: [/path/to/batch1_raw.parquet]
    batch2.yaml   # input_paths: [/path/to/batch2_raw.parquet]
    ...
```

## 3. Run the pipeline

```bash
nextflow run . --pipeline_dir /path/to/experiment
```

This runs the default `FisseqPipeline`, which chains every stage described in
[Architecture](architecture.md):

1. `INPUT` — converts each batch's config into an `input/*.parquet` file
   (always runs, once per config).
2. `QC_FILTER` — edit-distance, barcode-count, and variant-barcode-count filtering
   (per batch).
3. `BATCHVSBATCH` (pre) — batch-effect check on QC-filtered cells (once per
   active group in `params.global_groups`; none by default).
4. `NORMALIZE` — z-score normalization fit on WT control cells (per batch).
5. `BATCHVSBATCH` (post) — batch-effect check on normalized cells (once per
   active group).
6. `OVWT_BATCHWISE` / `OVWT_GLOBAL` — one-vs-wildtype XGBoost classification
   (`OVWT_GLOBAL` once per active group).
7. `WTVWT_BATCHWISE` — wildtype-only pairwise barcode classification (per
   batch, if `params.run_wtvwt`).
8. Bootstrap feature selection (batchwise always; global sub-branch once per
   active group) — see
   [Nextflow Workflow](nextflow.md#feature-selection-channel-wiring) for the
   six-stage breakdown.
9. `BATCH_CORRECT_FIT` / `BATCH_CORRECT_TRANSFORM` — centroid batch correction
   (always runs, over every batch).
10. `ANOVA` — batch-effect assessment, run once on normalized cells and once
    on batch-corrected cells (always runs).

Override any [parameter](configuration.md#parameters) on the command line,
e.g. to adjust QC thresholds. Global processes (`OVWT_GLOBAL`, the global
feature-selection branch, and `BATCHVSBATCH`) don't run at all unless you tag
batches into a group and activate it — see
[Configuration: Global groups](configuration.md#global-groups):

```bash
nextflow run . \
    --pipeline_dir /path/to/experiment \
    --barcode_count_threshold 15
```

To run on a cluster, supply your own config:

```bash
nextflow run . -c your.config -profile sge --pipeline_dir /path/to/experiment
```

If a run is interrupted, resume from the last completed task:

```bash
nextflow run . --pipeline_dir /path/to/experiment -resume
```

## 4. Inspect the results

All outputs land under `<pipeline_dir>`, alongside `configs/`/`input/` — see
[Architecture: Output layout](architecture.md#output-layout) for the full tree.
The two results most analyses care about:

- `<pipeline_dir>/feature_select_batchwise/<batch>/output.parquet` (and, if a
  global group is active, `global/<group>/feature_select/output.parquet`) —
  final per-variant, feature-selected profiles.
- `<pipeline_dir>/anova/anova.parquet` and
  `<pipeline_dir>/batch_correction/anova/anova.parquet` — per-feature
  batch-effect ANOVA results, before and after batch correction.

## 5. Running individual steps

Every Nextflow process wraps a standalone `python -m fisseq_data_pipeline.<module>`
invocation. To debug or rerun one stage manually, invoke it directly — see the
[CLI Reference](cli/qcfilter.md) for each tool's config fields:

```bash
uv run python -m fisseq_data_pipeline.qcfilter \
    output_dir=./out \
    'cell_files=[data/plate1.parquet]' \
    bc_threshold=10

uv run python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    input_file=out/filtered_cells.parquet
```
