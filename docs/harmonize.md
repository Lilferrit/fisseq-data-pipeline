# Harmonization utilities

The `fisseq_data_pipeline.harmonize` module provides utilities for
**batch-effect correction** using [ComBat](https://pubmed.ncbi.nlm.nih.gov/16632515/)
via the [`neuroHarmonize`](https://github.com/rpomponio/neuroHarmonize) package.

Harmonization is an essential step when combining data from multiple
experiments or sources, ensuring that technical variation (batch effects)
does not obscure biological signal.

---

## Overview

- **`fit_harmonizer`**: Learn a ComBat-based harmonization model from feature
  and metadata DataFrames.
- **`harmonize`**: Apply a fitted harmonization model to adjust new feature
  matrices for batch effects.

---

## Example usage

```python
import polars as pl
from fisseq_data_pipeline.harmonize import fit_harmonizer, harmonize

# Example feature matrix
feature_df = pl.DataFrame({
    "gene1": [1.0, 2.0, 3.0, 4.0],
    "gene2": [2.0, 3.0, 4.0, 5.0],
})

# Example metadata with batch column
meta_data_df = pl.DataFrame({
    "_batch": [0, 0, 1, 1],
    "_is_control": [True, True, True, False],
})

# Fit harmonizer (on control samples only)
harmonizer = fit_harmonizer(feature_df, meta_data_df, fit_only_on_control=True)

# Apply harmonization to full dataset
harmonized_df = harmonize(feature_df, meta_data_df, harmonizer)

print(harmonized_df)
```

## API Reference

---

::: fisseq_data_pipeline.harmonize.fit_harmonizer

---

::: fisseq_data_pipeline.harmonize.harmonize

---
