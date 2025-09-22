# Data Cleaning Utilities

The `fisseq_data_pipeline.filter` module provides functions to **clean and
filter feature/metadata tables** prior to normalization and harmonization.
These utilities are invoked automatically in the pipeline, but can also be
used independently.

## Overview

- **`clean_data`**: Removes invalid rows/columns from feature and metadata
  tables while keeping them aligned.
- **`drop_infrequent_pairs`**: Drops rows from rare `(label, batch)` groups
  according to a configurable threshold.


## Environment variables

- **`FISSEQ_PIPELINE_MIN_CLASS_MEMBERS`**  
  Minimum number of samples required per `(label, batch)` group when running
  `drop_infrequent_pairs`.  
  Default: `2`.

Example:

```bash
# Require at least 5 samples per label–batch group
FISSEQ_PIPELINE_MIN_CLASS_MEMBERS=5 fisseq-data-pipeline validate ...
```

### Example Usage

```python
import polars as pl
from fisseq_data_pipeline.filter import clean_data, drop_infrequent_pairs

# Example feature matrix
feature_df = pl.DataFrame({
    "f1": [1.0, 2.0, float("nan"), 4.0],
    "f2": [5.0, 6.0, 7.0, 8.0],
})

# Example metadata with batch + label
meta_df = pl.DataFrame({
    "_label": ["A", "A", "B", "B"],
    "_batch": ["X", "Y", "X", "Y"],
})

# Clean non-finite and zero-variance columns/rows
feature_df, meta_df = clean_data(feature_df, meta_df)

# Drop infrequent (label, batch) pairs
feature_df, meta_df = drop_infrequent_pairs(feature_df, meta_df)
```

## API reference

---

::: fisseq_data_pipeline.filter.clean_data

---

::: fisseq_data_pipeline.filter.drop_infrequent_pairs

---
