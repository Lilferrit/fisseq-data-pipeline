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

Each batch's CellProfiler feature matrix (with barcode/variant annotations) goes
in its own Parquet file under `<input_dir>/input/`:

```text
<input_dir>/
  input/
    batch1.parquet
    batch2.parquet
    ...
```

## 3. Run the pipeline

```bash
nextflow run . --input_dir /path/to/experiment
```

This runs the default `FisseqPipeline`, which chains every stage described in
[Architecture](architecture.md):

1. `QC_FILTER` — edit-distance, barcode-count, and variant-barcode-count filtering
   (per batch).
2. `BATCHVSBATCH` (pre) — batch-effect check on QC-filtered cells (global).
3. `NORMALIZE` — z-score normalization fit on WT control cells (per batch).
4. `BATCHVSBATCH` (post) — batch-effect check on normalized cells (global).
5. `OVWT_BATCHWISE` / `OVWT_GLOBAL` — one-vs-wildtype XGBoost classification.
6. Bootstrap feature selection (batchwise and, if `params.global`, global) —
   see [Nextflow Workflow](nextflow.md#feature-selection-channel-wiring) for the
   six-stage breakdown.
7. `BATCH_CORRECT_FIT` / `BATCH_CORRECT_TRANSFORM` — centroid batch correction.
8. `PERMANOVA` — batch-effect assessment, run once on normalized cells and once
   on batch-corrected cells.

Override any [parameter](nextflow.md#parameters) on the command line, e.g. to
adjust QC thresholds or skip the global feature-selection branch:

```bash
nextflow run . \
    --input_dir /path/to/experiment \
    --bc_threshold 15 \
    --global false
```

To run on a cluster, supply your own config:

```bash
nextflow run . -c your.config -profile sge --input_dir /path/to/experiment
```

If a run is interrupted, resume from the last completed task:

```bash
nextflow run . --input_dir /path/to/experiment -resume
```

## 4. Inspect the results

All outputs land under `<input_dir>`, alongside `input/` — see
[Architecture: Output layout](architecture.md#output-layout) for the full tree.
The two results most analyses care about:

- `<input_dir>/feature_select_batchwise/<batch>/output.parquet` (and
  `feature_select_global/output.parquet`, if `params.global`) — final
  per-variant, feature-selected profiles.
- `<input_dir>/permanova/permanova.parquet` and
  `<input_dir>/batch_correction/permanova/permanova.parquet` — per-variant
  batch-effect PERMANOVA results, before and after batch correction.

## 5. Running individual steps

Every Nextflow process wraps a standalone `fisseq-*` CLI tool. To debug or rerun
one stage manually, invoke it directly — see the [CLI Reference](cli/qcfilter.md)
for each tool's config fields:

```bash
uv run fisseq-qc-filter \
    output_dir=./out \
    'cell_files=[data/plate1.parquet]' \
    bc_threshold=10

uv run fisseq-normalize \
    output_dir=./out \
    input_file=out/filtered_cells.parquet
```
