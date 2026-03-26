# Normalize

The `fisseq_data_pipeline.normalize` module provides utilities for computing
and applying **z-score normalization** to feature matrices.

Normalization is typically run as part of the FISSEQ pipeline, but these
functions can also be used independently when you need to standardize feature
values.

---

## Overview

- **`Normalizer`**: A dataclass container storing per-feature means and
  standard deviations, plus a flag indicating whether statistics were computed
  batch-wise.
- **`fit_normalizer`**: Compute normalization statistics from a `data_lf`
  LazyFrame containing feature columns and `_meta_batch` / `_meta_is_control`
  metadata columns.
- **`normalize`**: Apply z-score normalization to a `data_lf` LazyFrame using
  a fitted `Normalizer`.

---

## Example usage

```python
import polars as pl
from fisseq_data_pipeline.normalize import fit_normalizer, normalize

# Combined data LazyFrame: features + metadata columns
data_lf = pl.DataFrame({
    "_meta_batch":      ["A", "A", "A", "B", "B", "B"],
    "_meta_is_control": [True, False, True, False, True, False],
    "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
}).lazy()

# Fit batch-wise normalizer on control samples only
normalizer = fit_normalizer(data_lf, fit_batch_wise=True, fit_only_on_control=True)

# Apply normalization
normalized_lf = normalize(data_lf, normalizer)

# Save for later reuse
normalizer.save("normalizer.pkl")
```

### Global normalization

```python
# Fit a single global normalizer across all samples
normalizer = fit_normalizer(data_lf, fit_batch_wise=False)
normalized_lf = normalize(data_lf, normalizer)
```

---

## Notes

- Columns with zero or near-zero variance are automatically dropped from the
  fitted `Normalizer` to avoid division-by-zero during normalization.
- `normalize` silently drops feature columns that are absent from the
  `Normalizer` (e.g. columns removed during fitting due to zero variance).
- Metadata columns (prefixed `_meta`) are preserved in the output of
  `normalize`.

---

## API reference

---

::: fisseq_data_pipeline.normalize.Normalizer

---

::: fisseq_data_pipeline.normalize.fit_normalizer

---

::: fisseq_data_pipeline.normalize.normalize

---
