# Data Cleaning

The `fisseq_data_pipeline.filter` module has been removed. Data cleaning is no longer a built-in pipeline step.

## Recommended approach

Pre-filter your input parquet file before passing it to the normalize step. Common operations using Polars:

```python
import polars as pl
from polars import selectors as cs

lf = pl.scan_parquet("raw_cells.parquet")

# Drop columns that are entirely null or NaN
feature_cols = lf.select(cs.exclude("^meta_.*$")).columns
lf = lf.select(
    cs.by_name("^meta_.*$"),
    *[pl.col(c) for c in feature_cols if lf.select(pl.col(c).is_finite().any()).collect().item()]
)

# Drop rows with any non-finite feature value
lf = lf.filter(cs.exclude("^meta_.*$").is_finite().all_horizontal())

lf.collect().write_parquet("cells_cleaned.parquet")
```

Then run normalization on the cleaned file:

```bash
python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    input_file=cells_cleaned.parquet
```
