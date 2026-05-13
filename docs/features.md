# Feature Selection

The `fisseq_data_pipeline.features` module selects reproducible and informative features from a per-variant aggregate. It first estimates feature reliability via pseudo-replicate Pearson correlations, uses that estimate to build a block list, re-aggregates the full dataset skipping blocked features, then applies pycytominer filters to remove low-variance and redundant features.

---

## Overview

### Pipeline

```
cell-level parquet
        │
        ▼
pseudo_replicate_correlation()
  ├── stratified 50/50 cell split
  ├── aggregate each half independently
  └── Pearson r per feature (reproducibility proxy)
        │
        ▼  r < minimum_correlation → blocked
feature_correlations.parquet  (feature, r, r², p_value, feature_ok)
        │
        ▼
aggregate()  (full dataset, block list applied)
        │
        ▼
pyc_feature_select()
  ├── variance_threshold
  ├── blocklist  (pycytominer built-in)
  └── correlation_threshold
        │
        ▼
output parquet  (one row per variant, selected features only)
```

### Key functions

| Function | Role |
| -------- | ---- |
| `pseudo_replicate_correlation` | Stratified 50/50 cell split → per-feature Pearson *r* |
| `compute_feature_correlations` | Joins two aggregate DataFrames and computes Pearson *r* |
| `get_replicate_lf` | Filters a LazyFrame to one pseudo-replicate half |
| `pyc_feature_select` | Wraps `pycytominer.feature_select` with three operations |
| `main` | Hydra entry point orchestrating all steps |

---

## Example usage

### Pseudo-replicate correlations

```python
import polars as pl
from fisseq_data_pipeline.features import pseudo_replicate_correlation

lf = pl.scan_parquet("cells_normalized.parquet")

corr_df = pseudo_replicate_correlation(
    lf,
    label_col="meta_aa_changes",
    aggregator_name="mean",
    random_state=42,
)
# corr_df: DataFrame with columns feature, r, r_squared, p_value
```

### Building a block list and aggregating

```python
from fisseq_data_pipeline.aggregate import aggregate

block_list = set(
    corr_df.filter(pl.col("r") < 0.5)["feature"].to_list()
)

agg_df = aggregate(
    lf,
    label_col="meta_aa_changes",
    aggregator_name="mean",
    block_list=block_list,
)
```

### pycytominer feature selection

```python
from fisseq_data_pipeline.features import pyc_feature_select

selected_df = pyc_feature_select(agg_df)
```

---

## CLI

```bash
python -m fisseq_data_pipeline.features \
    output_dir=./out \
    input_file=out/cells_aggregated.parquet \
    minimum_correlation=0.5 \
    aggregator=mean
```

Output files:

- `feature_correlations.parquet` — per-feature Pearson *r*, *r²*, *p*-value, and `feature_ok` flag
- `{input_stem}.parquet` — feature-selected aggregate (one row per variant)

With `output_root`:

- `{output_root}.feature_correlations.parquet`
- `{output_root}.{input_stem}.parquet`

### Configuration fields

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to cell-level normalized parquet file. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `aggregator` | `"multi"` | Aggregation method used for pseudo-replicate splitting and final aggregation. |
| `minimum_correlation` | `0.5` | Features with pseudo-replicate *r* below this threshold are blocked. |
| `random_state` | `42` | Seed for the stratified pseudo-replicate split. |

---

## API reference

---

::: fisseq_data_pipeline.features.FeatureSelectConfig

---

::: fisseq_data_pipeline.features.get_replicate_lf

---

::: fisseq_data_pipeline.features.compute_feature_correlations

---

::: fisseq_data_pipeline.features.pseudo_replicate_correlation

---

::: fisseq_data_pipeline.features.pyc_feature_select

---

::: fisseq_data_pipeline.features.main

---
