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

## 2. Declare your experiments

Experiments are declared as a list under `experiments:` in `params.yaml`. Each
entry names its raw CellProfiler feature matrix (or matrices) via `input_paths`;
there is no mode where the pipeline scans a directory of pre-staged Parquet
files. See [Configuration](configuration.md#declaring-experiments) for the full
schema and [CLI Reference: Input](cli/input.md) for what INPUT does with them.

```yaml
# params.yaml
experiments:
  - batch_stem: batch1
    input_paths: [/path/to/batch1_raw.parquet]
  - batch_stem: batch2
    input_paths: [/path/to/batch2_raw.parquet]
```

`<pipeline_dir>` is just the output root — it needs no pre-existing contents.

## 3. Run the pipeline

```bash
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml
```

This chains every stage described in [Architecture](architecture.md):

1. `INPUT` — merges each experiment's `input_paths` into one
   `input/<batch_stem>.parquet` (always runs, once per experiment).
2. `QC_FILTER` — edit-distance, barcode-count and variant-barcode-count
   filtering, per experiment. Also assigns `meta_cell_index`, the stable
   per-cell identity everything downstream depends on for reproducibility.
3. `NORMALIZE` — z-score normalization fit on WT control cells, per experiment.
4. `OVWT_BATCHWISE` — k-fold cross-validated one-vs-wildtype scoring, per
   experiment (`params.run_ovwt`). Each variant gets a pooled AUROC and a
   median-of-per-barcode AUROC, and every cell gets one out-of-fold score.
5. `GLOBAL_OVWT` — per-experiment synonymous z-score of those AUROCs, then the
   cross-experiment median (once per active global channel).
6. Bootstrap feature selection (`params.run_feature_selection`) — see
   [Nextflow Workflow](nextflow.md#processes) for the stage breakdown.
7. `GLOBAL_FEATURE_SELECT` — cross-experiment feature selection reusing the
   batchwise artifacts (once per active global channel).

Override any [parameter](configuration.md#parameter-reference) on the command
line. The two global stages don't run at all unless you tag experiments into a
channel and activate it — see
[Configuration: Global channels](configuration.md#global-channels):

```bash
nextflow run . \
    --pipeline_dir /path/to/experiment \
    -params-file params.yaml \
    --barcode_count_threshold 15
```

To run on a cluster, supply your own config:

```bash
nextflow run . -c your.config -profile sge \
    --pipeline_dir /path/to/experiment -params-file params.yaml
```

If a run is interrupted, resume from the last completed task:

```bash
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml -resume
```

## 4. Inspect the results

All outputs land under `<pipeline_dir>` — see
[Architecture: Output layout](architecture.md#output-layout) for the full tree.
The results most analyses care about:

- `<pipeline_dir>/feature_select_batchwise/<batch_stem>/output.parquet` (and, if
  a global channel is active, `global/<channel>/feature_select/aggregate.parquet`)
  — final per-variant, feature-selected profiles.
- `<pipeline_dir>/ovwt_batchwise/<batch_stem>/results.parquet` — per-variant
  `auroc_pooled` and `auroc_median_barcode`.
- `<pipeline_dir>/global/<channel>/ovwt_distinguishability/global_scores.parquet`
  (only if a global channel is active) — synonymous-corrected, cross-experiment
  median distinguishability per variant. This is the headline result for a
  multi-experiment run.

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
