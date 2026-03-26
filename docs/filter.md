# Data Cleaning Utilities

The `fisseq_data_pipeline.filter` module provides functions to **clean and
filter feature tables** prior to normalization. These utilities are invoked
automatically in the pipeline, but can also be used independently.

## Overview

- **`clean_data`**: Applies a configurable sequence of filtering stages to a
  `LazyFrame`. The default pipeline removes all-non-finite columns then rows
  with any non-finite value.
- **`drop_cols_all_nonfinite`**: Drops columns where every value is NaN, +inf,
  or -inf.
- **`drop_rows_any_nonfinite`**: Drops rows containing any non-finite feature
  value.

## Example Usage

```python
import polars as pl
from fisseq_data_pipeline.filter import clean_data

# Example combined data LazyFrame (features + metadata)
data_lf = pl.DataFrame({
    "_meta_batch": ["A", "A", "B"],
    "_meta_label": ["X", "X", "Y"],
    "f1": [1.0, float("nan"), 3.0],
    "f2": [5.0, 6.0, float("inf")],
    "f3": [float("nan"), float("nan"), float("nan")],
}).lazy()

# Run default pipeline: drop all-nonfinite columns, then rows with any nonfinite
cleaned_lf = clean_data(data_lf)

# Run only one stage
cleaned_lf = clean_data(data_lf, stages=["drop_cols_all_nonfinite"])

# Insert a custom filtering stage
def my_filter(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("f1") > 0)

cleaned_lf = clean_data(data_lf, stages=["drop_cols_all_nonfinite", my_filter])
```

## Notes

- Only columns **not** prefixed with `_meta` are treated as feature columns
  and considered during non-finite checks. Metadata columns are carried
  through unchanged.
- Unknown string stage names are skipped with a `WARNING` log message.
- Custom stages must accept and return a `pl.LazyFrame`.

## API reference

---

::: fisseq_data_pipeline.filter.clean_data

---

::: fisseq_data_pipeline.filter.drop_cols_all_nonfinite

---

::: fisseq_data_pipeline.filter.drop_rows_any_nonfinite

---
