# Normalize

The `fisseq_data_pipeline.normalize` module provides utilities for computing
and applying **z-score normalization** to feature matrices.

Normalization is typically run as part of the FISSEQ pipeline, but these
functions can also be used independently when you need to standardize feature
values.

---

## Overview

- **`Normalizer`**: A dataclass container storing per-column means and
  standard deviations.  
- **`fit_normalizer`**: Compute normalization statistics (means and stds) from
  a feature DataFrame, optionally restricted to control samples.  
- **`normalize`**: Apply z-score normalization to a feature DataFrame using a
  fitted `Normalizer`.

---

## Example usage

```python
import polars as pl
from fisseq_data_pipeline.normalize import fit_normalizer, normalize

# Example feature matrix
feature_df = pl.DataFrame({
    "x": [1.0, 2.0, 3.0],
    "y": [2.0, 4.0, 6.0],
})

# Fit the normalizer
normalizer = fit_normalizer(feature_df)

# Apply normalization
normalized_df = normalize(feature_df, normalizer)

print(normalized_df)
```

### Output

```bash
shape: (3, 2)
┌──────────┬──────────┐
│ x        │ y        │
│ ---      │ ---      │
│ f64      │ f64      │
╞══════════╪══════════╡
│ -1.0     │ -1.0     │
│  0.0     │  0.0     │
│  1.0     │  1.0     │
└──────────┴──────────┘
```

## API reference

---

::: fisseq_data_pipeline.normalize.Normalizer

---

::: fisseq_data_pipeline.normalize.fit_normalizer

---

::: fisseq_data_pipeline.normalize.normalize

---