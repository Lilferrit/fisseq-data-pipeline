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
      ├──▶ PERMANOVA_WT            (global — waits for all batches)
      ├──▶ PERMANOVA_SYN           (global — waits for all batches)
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
nextflow run your-org/fisseq-data-pipeline \
    -c your.config \
    --input_dir /path/to/experiment
```

Or from a local clone:

```bash
nextflow run . --input_dir /path/to/experiment
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
| `--permanova_n_bootstraps` | `200` | Bootstrap iterations for PERMANOVA. |
| `--permanova_sample_size` | `1000` | Rows per PERMANOVA bootstrap sample. |
| `--ovwt_min_cells` | `250` | Minimum cells required per variant for OvWT classification. |
| `--downsample_wt` | `5000` | Downsample wildtype cells to this count for OvWT classification. |
| `--aggregator` | `"multi"` | Feature aggregation method (see `fisseq-feature-select` docs). |

### Environment configuration

`nextflow.config` in this repo contains only executor/cluster settings. Environment setup — activating a virtualenv or installing the package — belongs in a separate config file that you pass via `-c`:

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
    wildtype/permanova.parquet
    synonymous/permanova.parquet
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

Assess batch effects in cell-level data using bootstrapped PERMANOVA on cosine distance matrices. `input_file` is treated as a glob pattern; each matching file becomes one batch (labeled by its filename stem). For each bootstrap sample, both an observed F-statistic and a label-shuffled null F-statistic are computed from the same distance matrix, making it straightforward to compare signal vs. noise across runs.

Can optionally filter to a single variant class (e.g. WT-only) before computing, which isolates technical batch variation from biological signal.

**Output files**:
- `{output_dir}/permanova.parquet` — `n_bootstraps` rows with `f_value` and `f_value_shuffled` columns

```bash
fisseq-permanova \
    output_dir=./out \
    'input_file=data/batches/*.parquet' \
    variant_class_filter=WT \
    n_bootstraps=200 \
    sample_size=1000
```

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern matching one or more batch parquet files. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `variant_class_filter` | `"WT"` | Restrict to this variant class before PERMANOVA (`null` uses all rows). |
| `n_bootstraps` | `200` | Number of bootstrap samples. |
| `sample_size` | `1000` | Rows per bootstrap sample. |
| `seed` | `42` | Base random seed; bootstrap *i* uses `seed + i`. |
| `parallel` | `false` | Run bootstraps in parallel via joblib. |
| `n_jobs` | `-1` | Parallel worker count (`-1` = all cores). Used only when `parallel=true`. |

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
