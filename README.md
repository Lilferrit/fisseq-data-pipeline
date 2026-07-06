# FISSEQ Data Pipeline

The **FISSEQ Data Pipeline** provides a reproducible, configurable workflow for processing **FISSEQ cell profiling data**.
It handles QC filtering, z-score normalization, per-variant feature aggregation, feature selection, batch-effect validation, and one-vs-wildtype variant classification.

---

## Pipeline overview

```
input/*.parquet  (one file per batch)
      │
      ▼
QC_FILTER  (per batch)     ← edit distance, barcode count, variant barcode count filters
      │
      ▼
NORMALIZE  (per batch)     ← z-score normalization fit on WT control cells
      │
      ├──▶ PERMANOVA               (global — waits for all batches)
      ├──▶ OVWT_BATCHWISE          (per batch)
      ├──▶ OVWT_GLOBAL             (global — waits for all batches)
      ├──▶ FEATURE_SELECT_BATCHWISE (per batch)
      └──▶ FEATURE_SELECT_GLOBAL   (global — waits for all batches)
```

---

## Installation

This package is experimental and not hosted on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/Lilferrit/fisseq-data-pipeline.git
```

Or clone and install locally:

```bash
git clone https://github.com/Lilferrit/fisseq-data-pipeline.git
cd fisseq-data-pipeline
pip install -e .
```

---

## Nextflow pipeline

### Prerequisites

- [Nextflow](https://www.nextflow.io/) ≥ 23.10
- A Python environment with `fisseq-data-pipeline` installed (see [Installation](#installation))

### Quickstart

Run directly from GitHub — no cloning required:

```bash
nextflow run Lilferrit/fisseq-data-pipeline \
    -c your.config \
    --input_dir /path/to/experiment
```

Use `-r` to pin to a branch or release tag. Nextflow caches the pulled revision in `~/.nextflow/assets`; pass `-latest` to force a refresh:

```bash
# Track the main branch
nextflow run Lilferrit/fisseq-data-pipeline \
    -r main \
    -c your.config \
    --input_dir /path/to/experiment

# Pin to a specific release tag
nextflow run Lilferrit/fisseq-data-pipeline \
    -r v1.2.0 \
    -c your.config \
    --input_dir /path/to/experiment
```

Or from a local clone:

```bash
nextflow run . --input_dir /path/to/experiment
```

### Choosing a workflow

The pipeline ships two workflows, selected with `--workflow`:

| Workflow | Steps | Use when |
| -------- | ----- | -------- |
| `fisseq` (default) | QC → Normalize → PERMANOVA (WT & SYN) → OvWT (batchwise & global) → Feature select (batchwise & global) | Full end-to-end analysis |
| `ovwt` | QC → OvWT batchwise → OvWT cell scores batchwise | OvWT classification only, no normalization or feature selection |

**Run the full fisseq pipeline** (default — `--workflow fisseq` may be omitted):

```bash
nextflow run Lilferrit/fisseq-data-pipeline \
    -c your.config \
    --input_dir /path/to/experiment
```

**Run the OvWT-only pipeline:**

```bash
nextflow run Lilferrit/fisseq-data-pipeline \
    -c your.config \
    --workflow ovwt \
    --input_dir /path/to/experiment
```

### Input directory layout

Place one `.parquet` file per batch inside an `input/` subdirectory of your experiment root:

```
<input_dir>/
  input/
    batch1.parquet
    batch2.parquet
    ...
```

### Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--input_dir` | **required** | Root directory containing `input/*.parquet` batch files. |
| `--bc_threshold` | `10` | Minimum cells per barcode (QC filter). |
| `--variant_bc_threshold` | `4` | Minimum distinct barcodes per variant (QC filter). |
| `--edit_distance_threshold` | `1` | Maximum allowed edit distance (QC filter). |
| `--minimum_correlation` | `0.5` | Minimum pseudo-replicate Pearson *r* for feature selection. |
| `--permanova_n_permutations` | `999` | Label permutations for the per-variant PERMANOVA p-value. |
| `--ovwt_min_cells` | `250` | Minimum cells required per variant for OvWT classification. |
| `--downsample_wt` | `5000` | Downsample wildtype cells to this count for OvWT classification. |
| `--aggregator` | `"multi"` | Feature aggregation method (see `fisseq-feature-select` docs). |
| `--workflow` | `"fisseq"` | Which workflow to run: `"fisseq"` (full pipeline) or `"ovwt"` (OvWT-only). |

### Environment configuration

The repo does not ship a `nextflow.config`. Executor and environment setup are entirely user-provided via a config file passed with `-c`:

```bash
nextflow run . -c your.config --input_dir /path/to/experiment
```

Copy `example.config` from the repo root as a starting point and uncomment/fill in one of the `beforeScript` options:

```groovy
// Option A: activate a pre-existing venv (recommended for shared clusters)
beforeScript = 'source /path/to/your/venv/bin/activate'

// Option B: install from GitHub on each run (simpler, slower)
beforeScript = 'uv pip install git+https://github.com/your-org/fisseq-data-pipeline.git@main --system'
```

### Output layout

All outputs are written back into `<input_dir>` alongside the `input/` folder:

```text
<input_dir>/
  qc_filter/<batch>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  normalization/
    cells/<batch>.parquet
    normalizers/<batch>.normalizer.parquet
  permanova/
    permanova.parquet
  ovwt_batchwise/<batch>/
    results.parquet
    models.pkl
  ovwt_global/
    results.parquet
    models.pkl
  feature_select_batchwise/<batch>/
    <batch>.parquet
    feature_correlations.parquet
  feature_select_global/
    global.parquet
    feature_correlations.parquet
```

### Resuming a run

Nextflow caches completed tasks. Resume from the last successful step after an interruption:

```bash
nextflow run . --input_dir /path/to/experiment -resume
```

---

## CLI tools

### `fisseq-qc-filter`

Filter raw cell-level data before any downstream processing. Accepts one or more input files (CSV or Parquet) and applies three sequential filters:

1. **Edit distance** — drops cells with `editDistance > edit_distance_threshold`.
2. **Barcode cell count** — drops barcodes represented by fewer than `bc_threshold` cells.
3. **Variant barcode count** — drops variants supported by fewer than `variant_bc_threshold` distinct barcodes.

Input column names (`upBarcode`, `aaChanges`, `editDistance`) are configurable in case they differ across datasets.

**Output files** (written to `output_dir`):
- `filtered_cells.parquet` — cells passing all three filters
- `barcode_counts.parquet` — per-barcode cell counts and pass/fail flags
- `variants_per_barcode.parquet` — per-variant barcode counts and pass/fail flags

```bash
fisseq-qc-filter \
    output_dir=./out \
    'cell_files=[data/plate1.parquet, data/plate2.parquet]' \
    bc_threshold=10 \
    variant_bc_threshold=4 \
    edit_distance_threshold=1
```

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cell_files` | **required** | Path or list of paths to raw cell files (CSV or Parquet). |
| `bc_threshold` | `10` | Minimum cells required per barcode. |
| `variant_bc_threshold` | `4` | Minimum distinct barcodes required per variant. |
| `edit_distance_threshold` | `1` | Maximum allowed edit distance. |
| `barcode_col_name` | `"upBarcode"` | Input column name for cell barcodes. |
| `aa_changes_col_name` | `"aaChanges"` | Input column name for amino-acid change labels. |
| `edit_distance_col_name` | `"editDistance"` | Input column name for edit distances. |
| `label_column` | `"meta_aa_changes"` | Output column name for the variant label. |

---

### `fisseq-normalize`

Fit per-feature z-score statistics on control (WT) cells and apply them across the full dataset. The normalizer (means and standard deviations) can be saved as a Parquet file for later reuse or auditing. Features with zero variance are stored as `null` in the output.

Control rows are identified via a SQL WHERE clause (`control_sample_query`) evaluated against the input frame, making it easy to adapt to non-standard control labels without code changes.

**Output files**:
- `{output_dir}/{input_stem}.parquet` — normalized cell data with an added `meta_is_control` boolean column
- `{output_dir}/normalizer.parquet` — fitted normalizer (when `save_normalizer=true`)

```bash
fisseq-normalize \
    output_dir=./out \
    input_file=out/filtered_cells.parquet
```

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to QC-filtered cell parquet. |
| `control_sample_query` | `"meta_aa_changes = 'WT'"` | SQL WHERE clause identifying control rows used to fit the normalizer. |
| `save_normalizer` | `true` | Write the fitted normalizer to `normalizer.parquet`. |

---

### `fisseq-aggregate`

Aggregate cell-level data to one row per variant using a chosen aggregation method. After aggregation, variant profiles are z-scored relative to the synonymous variant baseline (using the same `Normalizer` class as `fisseq-normalize`). Supports multiple aggregation strategies including distribution-comparison methods that compare each variant's cell population against the WT distribution.

**Output files**:
- `{output_dir}/{input_stem}.parquet` — per-variant aggregate, normalized to synonymous baseline
- `{output_dir}/normalizer.parquet` — synonymous-baseline normalizer (when `save_normalizer=true`)

```bash
fisseq-aggregate \
    output_dir=./out \
    input_file=out/filtered_cells.parquet \
    aggregator=multi
```

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to cell-level normalized parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `aggregator` | `"multi"` | Aggregation method (see table below). |
| `save_normalizer` | `true` | Write the synonymous-baseline normalizer. |
| `block_list_file` | `null` | Parquet with `feature` and `feature_ok` columns; blocked features are skipped. |
| `compute_impact_score` | `true` | Append an impact score column derived from variant classification. |

#### Aggregators

| Value | Description |
| ----- | ----------- |
| `mean` | Per-variant feature mean |
| `median` | Per-variant feature median |
| `MAD` | Per-variant median absolute deviation |
| `std` | Per-variant standard deviation |
| `EMD` | Earth Mover's Distance vs. WT distribution |
| `KS` | Kolmogorov-Smirnov statistic vs. WT distribution |
| `QQ` | Q-Q Pearson correlation vs. WT distribution |
| `AUROC` | AUROC vs. WT distribution |
| `multi` | All of the above except EMD (default) |

---

### `fisseq-feature-select`

Filter features in two stages before producing the final aggregate:

1. **Pseudo-replicate reproducibility** — cells are split into two equal halves stratified by variant label. Each half is independently aggregated and per-feature Pearson *r* between the two halves is computed. Features below `minimum_correlation` are added to a block list.
2. **pycytominer selection** — the full dataset is aggregated with the block list applied, then `pycytominer.feature_select` removes low-variance, blocklisted, and redundant (highly correlated) features.

**Output files**:
- `{output_dir}/feature_correlations.parquet` — per-feature *r*, *r²*, *p*-value, and `feature_ok` flag
- `{output_dir}/{input_stem}.parquet` — final feature-selected per-variant aggregate

```bash
fisseq-feature-select \
    output_dir=./out \
    input_file=out/filtered_cells.parquet \
    minimum_correlation=0.5
```

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to cell-level normalized parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `aggregator` | `"multi"` | Aggregation method for pseudo-replicate splitting and final aggregation. |
| `minimum_correlation` | `0.5` | Features with pseudo-replicate *r* below this threshold are blocked. |
| `random_state` | `42` | Seed for the stratified pseudo-replicate split. |

---

### `fisseq-permanova`

Assess batch effects in cell-level data using a per-variant one-way PERMANOVA on cosine distances. `input_file` is treated as a glob pattern; each matching file becomes one batch (labeled by its filename stem). For each variant seen in more than one batch, all unique pairwise cosine distances between its cells are computed (including cross-batch pairs) and used to derive a pseudo-F statistic via the standard sum-of-squares decomposition (Anderson 2001). An optional permutation test (shuffling batch labels while holding distances fixed) yields a p-value per variant.

**Output files**:
- `{output_dir}/permanova.parquet` — one row per eligible variant, with per-variant metadata (from `get_aggregate_meta_data`), `f_statistic`, and `p_value` (`null` when `n_permutations=0`)

```bash
fisseq-permanova \
    output_dir=./out \
    'input_file=data/batches/*.parquet' \
    n_permutations=999
```

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern matching one or more batch parquet files. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `n_permutations` | `999` | Number of label permutations for the p-value. `0` skips the permutation test (`p_value` is `null`). |
| `seed` | `42` | Base random seed for label permutation; variant *i* (sorted order) uses `seed + i`. |

---

### `fisseq-ovwt`

Train a separate XGBoost binary classifier for each non-wildtype variant, treating the task as "this variant vs. wildtype." An 80/10/10 train/test/val split (stratified by label) is shared across all variants. Wildtype cells can be downsampled to the size of the largest variant group to reduce class imbalance. Results (per-variant AUROC and accuracy on train/val/test splits) and all trained models are serialized to disk.

**Output files**:
- `{output_dir}/results.csv` — per-variant `train_auroc`, `val_auroc`, `test_auroc`, `train_accuracy`, `val_accuracy`, `test_accuracy`, plus any per-variant metadata columns
- `{output_dir}/models.pkl` — dictionary of trained `xgb.Booster` objects keyed by variant label
- `{output_dir}/{train,test,val}.parquet` — data splits (only when `save_splits=true`)

```bash
fisseq-ovwt \
    output_dir=./out \
    input_file=out/features.parquet \
    min_cells=250 \
    downsample_wt=true
```

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to feature-selected per-variant or cell-level parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `wt_label` | `"WT"` | Label string for wildtype cells. |
| `min_cells` | `250` | Drop variants with fewer than this many cells (`null` disables). |
| `downsample_wt` | `true` | Downsample WT to the size of the largest variant group before splitting. |
| `random_state` | `42` | Seed for splits and WT downsampling. |
| `feature_cols` | `null` | Explicit list of feature column names; auto-detected if `null`. |
| `save_splits` | `false` | Write train/test/val splits to parquet files. |
| `xgboost.num_boost_round` | `100` | Maximum boosting rounds. |
| `xgboost.early_stopping_rounds` | `5` | Stop early if eval metric does not improve. |
| `xgboost.weigh_samples` | `true` | Use balanced sample weights to handle class imbalance. |
| `xgboost.params.max_depth` | `3` | Maximum tree depth. |
| `xgboost.params.subsample` | `0.5` | Fraction of rows sampled per tree. |

---

### `fisseq-batch-vs-batch` *(invoke as module — no registered entry point yet)*

Detect per-variant batch effects using a multiclass XGBoost classifier. For each variant, a single model is trained to predict the batch label from cell morphology features. From its predicted probabilities, a one-vs-rest ROC AUC and Mann-Whitney U p-value are extracted for every (variant, batch) pair, quantifying how distinguishable each batch is from the others within that variant.

The global 80/10/10 train/test/val split is stratified on a composite (variant, batch) key so every variant's rows in each split span all of its batches. One model is trained per variant (not per (variant, batch) pair).

**Output files**:

- `{output_dir}/results.parquet` — one row per (variant, batch) pair with columns `variant`, `batch`, `auroc`, `mw_pvalue`, `n_batch_cells`, `n_cells`

```bash
uv run python -m fisseq_data_pipeline.batchvsbatch \
    output_dir=./out \
    input_file=out/features.parquet \
    batch_column=meta_batch
```

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to feature-selected or normalized cell-level parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `batch_column` | `"meta_batch"` | Column identifying batch labels. |
| `min_cells` | `50` | Skip variants with fewer than this many cells total. |
| `min_batches` | `2` | Skip variants appearing in fewer than this many unique batches. |
| `random_state` | `42` | Seed for the stratified split. |
| `feature_cols` | `null` | Explicit list of feature column names; auto-detected if `null`. |
| `xgboost.num_boost_round` | `100` | Maximum boosting rounds. |
| `xgboost.early_stopping_rounds` | `5` | Stop early if eval metric does not improve. |
| `xgboost.weigh_samples` | `true` | Use balanced sample weights. |
| `xgboost.params.max_depth` | `3` | Maximum tree depth. |
| `xgboost.params.subsample` | `0.5` | Fraction of rows sampled per tree. |

---

## Common options

All entry points share a base set of fields:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, all output files are prefixed `{output_root}.{name}` instead of being placed in `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`). |

### Named output roots

`output_root` prefixes every output file, which is useful when running multiple configurations into the same directory:

```bash
fisseq-normalize \
    output_dir=./out \
    output_root=run1 \
    input_file=data/cells.parquet
# Produces: out/run1.cells.parquet, out/run1.normalizer.parquet, out/run1.normalize.log
```

---

## Logging

Logs are written to both stdout and a `.log` file in `output_dir`:

```bash
fisseq-normalize \
    output_dir=./out \
    input_file=data/cells.parquet \
    log_level=debug
```

---

## Documentation

Full API documentation: [https://lilferrit.github.io/fisseq-data-pipeline/](https://lilferrit.github.io/fisseq-data-pipeline)

## License

[MIT](LICENSE.txt)
