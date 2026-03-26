# Aggregate

The `fisseq_data_pipeline.aggregate` module provides aggregators that
summarize a normalized feature matrix into per-(batch, label) statistics.
It also exposes a two-subcommand CLI for computing aggregations and
normalizing aggregate DataFrames to synonymous variants.

---

## Overview

### Class hierarchy

```
BaseAggregator (abstract)
├── NativeAggregator (abstract) — Polars group-by expressions, no reference
│   ├── MeanAggregator
│   ├── MedianAggregator
│   ├── MADAggregator
│   └── StdAggregator
└── ReferenceBaseAggregator (abstract) — requires control reference distribution
    ├── EMDAggregator
    ├── KSAggregator
    ├── QQCorrelationAggregator
    └── AUROCAggregator
```

**Native aggregators** use Polars `group_by().agg()` and require no reference
distribution. They accept only `agg_df` as an argument to `aggregate()`.

**Reference-based aggregators** compare each (batch, label) group against a
control reference distribution. They are constructed with a `reference_df` and
use joblib for parallel dispatch.

### Aggregator registry

| Key | Class | Type |
|-----|-------|------|
| `"mean"` | `MeanAggregator` | Native |
| `"median"` | `MedianAggregator` | Native |
| `"MAD"` | `MADAggregator` | Native |
| `"std"` | `StdAggregator` | Native |
| `"EMD"` | `EMDAggregator` | Reference |
| `"KS"` | `KSAggregator` | Reference |
| `"QQ"` | `QQCorrelationAggregator` | Reference |
| `"AUROC"` | `AUROCAggregator` | Reference |

---

## Example usage

### Native aggregator

```python
import polars as pl
from fisseq_data_pipeline.aggregate import MeanAggregator

data_df = pl.DataFrame({
    "_meta_batch": ["A", "A", "B", "B"],
    "_meta_label": ["X", "X", "Y", "Y"],
    "f1": [1.0, 2.0, 3.0, 4.0],
    "f2": [5.0, 6.0, 7.0, 8.0],
})

agg = MeanAggregator()
result = agg.aggregate(data_df)
```

### Reference-based aggregator

```python
from fisseq_data_pipeline.aggregate import EMDAggregator

reference_df = data_df.filter(pl.col("_meta_label") == "WT")
agg = EMDAggregator(reference_df)
result = agg.aggregate(data_df)
```

---

## CLI

The module exposes two subcommands via Python Fire:

### `compute`

Compute per-(batch, label) aggregation statistics from a normalized Parquet
file. Outputs `aggregated.parquet` containing only the aggregate feature
columns.

```bash
python -m fisseq_data_pipeline.aggregate compute \
  --norm_df normalized.parquet \
  --out_dir out/ \
  --aggregator mean
```

Valid `--aggregator` values: `mean`, `median`, `MAD`, `std`, `EMD`, `KS`,
`QQ`, `AUROC`.

### `normalize`

Normalize an aggregate DataFrame to synonymous (synonymous-variant) rows,
fitting and applying a normalizer. Outputs `normalized.parquet` and
`normalizer.pkl`.

```bash
python -m fisseq_data_pipeline.aggregate normalize \
  --agg_df aggregated.parquet \
  --out_dir out/
```

---

## API reference

---

::: fisseq_data_pipeline.aggregate.BaseAggregator

---

::: fisseq_data_pipeline.aggregate.NativeAggregator

---

::: fisseq_data_pipeline.aggregate.MeanAggregator

---

::: fisseq_data_pipeline.aggregate.MedianAggregator

---

::: fisseq_data_pipeline.aggregate.MADAggregator

---

::: fisseq_data_pipeline.aggregate.StdAggregator

---

::: fisseq_data_pipeline.aggregate.ReferenceBaseAggregator

---

::: fisseq_data_pipeline.aggregate.EMDAggregator

---

::: fisseq_data_pipeline.aggregate.KSAggregator

---

::: fisseq_data_pipeline.aggregate.QQCorrelationAggregator

---

::: fisseq_data_pipeline.aggregate.AUROCAggregator

---

::: fisseq_data_pipeline.aggregate.compute_cli

---

::: fisseq_data_pipeline.aggregate.normalize_cli

---
