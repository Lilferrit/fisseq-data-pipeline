# Utility functions

The `fisseq_data_pipeline.utils.utils` module provides helper functions for feature selection, dataset construction, and splitting. These utilities are used internally by the pipeline but can also be reused in standalone scripts.

## Overview

- **`get_feature_selector`**: Build a Polars selector expression for feature
  columns based on the pipeline configuration.
- **`get_feature_columns`**: Select feature columns from a `LazyFrame` using a
  config.
- **`get_data_dfs`**: Build aligned feature and metadata DataFrames from a
  Polars `LazyFrame`.
- **`train_test_split`**: Create stratified train/test splits for features and
  metadata.

## Environment variables

- **`FISSEQ_PIPELINE_RAND_STATE`**  
  Random seed used for reproducible stratified train/test splits.  
  Default: `42`.

### Example

```bash
# Use a fixed seed of 1234 for train/test splitting
FISSEQ_PIPELINE_RAND_STATE=1234 fisseq-data-pipeline validate ...
```

## Example Usage

```python
import polars as pl
from fisseq_data_pipeline.utils.config import Config
from fisseq_data_pipeline.utils.utils import (
    get_data_dfs, train_test_split
)

# Example dataset
df = pl.DataFrame({
    "gene1": [1.0, 2.0, 3.0, 4.0],
    "gene2": [5.0, 6.0, 7.0, 8.0],
    "batch": ["A", "A", "B", "B"],
    "label": ["X", "X", "Y", "Y"],
    "is_ctrl": [True, False, True, False],
}).lazy()

# Example config (using dict for simplicity)
cfg = Config({
    "feature_cols": ["gene1", "gene2"],
    "batch_col_name": "batch",
    "label_col_name": "label",
    "control_sample_query": "col('is_ctrl')",
})

# Build feature + metadata DataFrames
feature_df, meta_data_df = get_data_dfs(df, cfg)

# Stratified train/test split
train_f, train_m, test_f, test_m = train_test_split(feature_df, meta_data_df, test_size=0.5)
```

## API Reference

---

::: fisseq_data_pipeline.utils.utils.get_feature_selector

---

::: fisseq_data_pipeline.utils.utils.get_feature_columns

---

::: fisseq_data_pipeline.utils.utils.get_data_dfs

---

::: fisseq_data_pipeline.utils.utils.train_test_split

---
