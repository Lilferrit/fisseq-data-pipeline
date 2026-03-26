# Utility functions

The `fisseq_data_pipeline.utils` module provides helper functions for feature
selection and dataset construction. These utilities are used internally by the
pipeline but can also be reused in standalone scripts.

## Overview

- **`get_feature_selector`**: Build a Polars selector expression for feature
  columns based on the pipeline configuration.
- **`get_data_lf`**: Construct a combined feature + metadata `LazyFrame` from
  a raw input `LazyFrame`.
- **`get_feature_cols`**: Return column expressions (or names) for all
  non-metadata columns in a `LazyFrame`.
- **`get_feature_lf`**: Return a `LazyFrame` containing only feature columns.

## Environment variables

- **`FISSEQ_PIPELINE_RAND_STATE`**
  Random seed used for reproducible operations.
  Default: `42`.

## Example Usage

```python
import polars as pl
from fisseq_data_pipeline.config import Config
from fisseq_data_pipeline.utils import get_data_lf, get_feature_cols

# Example raw dataset
db_lf = pl.DataFrame({
    "gene1": [1.0, 2.0, 3.0, 4.0],
    "gene2": [5.0, 6.0, 7.0, 8.0],
    "batch": ["A", "A", "B", "B"],
    "label": ["X", "X", "Y", "Y"],
    "is_ctrl": [True, False, True, False],
}).lazy()

# Config (using dict for simplicity)
cfg = Config({
    "feature_cols": ["gene1", "gene2"],
    "batch_col_name": "batch",
    "label_col_name": "label",
    "control_sample_query": "is_ctrl = true",
})

# Build combined data LazyFrame
data_lf = get_data_lf(db_lf, cfg)

# Get feature column names
feature_names = get_feature_cols(data_lf, as_string=True)
```

## API Reference

---

::: fisseq_data_pipeline.utils.get_feature_selector

---

::: fisseq_data_pipeline.utils.get_data_lf

---

::: fisseq_data_pipeline.utils.get_feature_cols

---

::: fisseq_data_pipeline.utils.get_feature_lf

---
