# Aggregate

The `fisseq_data_pipeline.aggregate` module summarizes cell-level normalized feature matrices into per-variant statistics. After aggregation, variants are classified as synonymous or non-synonymous and a second z-score normalization pass is applied using synonymous variants as the reference baseline.

---

## Overview

### Class hierarchy

All aggregators inherit from a single `BaseAggregator` abstract base class:

```
BaseAggregator (abstract)
├── MeanAggregator
├── MedianAggregator
├── MADAggregator
├── StdAggregator
├── EMDAggregator            — requires reference_df
├── KSAggregator             — requires reference_df
├── QQCorrelationAggregator  — requires reference_df
├── AUROCAggregator          — requires reference_df
└── MultiAggregator          — composes a list of aggregators
```

Aggregators that compare against a reference distribution (`EMD`, `KS`, `QQ`, `AUROC`) accept a `reference_df` (the control population) at construction time. If `reference_df` is `None`, `aggregate()` raises `ValueError`.

### Aggregator registry

| Key | Class | Statistic |
| --- | ----- | --------- |
| `"mean"` | `MeanAggregator` | Per-group mean |
| `"median"` | `MedianAggregator` | Per-group median |
| `"MAD"` | `MADAggregator` | Per-group median absolute deviation |
| `"std"` | `StdAggregator` | Per-group standard deviation |
| `"EMD"` | `EMDAggregator` | Earth Mover's Distance vs. control |
| `"KS"` | `KSAggregator` | KS statistic vs. control |
| `"QQ"` | `QQCorrelationAggregator` | Q-Q Pearson correlation vs. control |
| `"AUROC"` | `AUROCAggregator` | AUROC vs. control |
| `"multi"` | `MultiAggregator` | All of the above except EMD (default) |

---

## Example usage

### Single aggregator

```python
import polars as pl
from fisseq_data_pipeline.aggregate import MeanAggregator

df = pl.read_parquet("cells_normalized.parquet")
control_df = df.filter(pl.col("meta_is_control"))

agg = MeanAggregator(reference_df=control_df, label_col="meta_aa_changes")
result = agg.aggregate(df)
```

### Reference-based aggregator

```python
from fisseq_data_pipeline.aggregate import KSAggregator

agg = KSAggregator(reference_df=control_df, label_col="meta_aa_changes")
result = agg.aggregate(df)
```

### Module-level `aggregate()` function

```python
import polars as pl
from fisseq_data_pipeline.aggregate import aggregate

lf = pl.scan_parquet("cells_normalized.parquet")

# Run all non-EMD aggregators (default)
result = aggregate(lf, label_col="meta_aa_changes", aggregator_name="multi")

# Run a single aggregator
result = aggregate(lf, label_col="meta_aa_changes", aggregator_name="KS")
```

---

## CLI

```bash
python -m fisseq_data_pipeline.aggregate \
    output_dir=./out \
    input_file=data/cells_normalized.parquet \
    aggregator=multi \
    label_column=meta_aa_changes
```

Valid `aggregator` values: `mean`, `median`, `MAD`, `std`, `EMD`, `KS`, `QQ`, `AUROC`, `multi`.

Output path:

- `output_root` set → `{output_root}.{stem}.{ext}`
- `output_root` not set → `{output_dir}/{filename}` (same name as input)

When `save_normalizer=true`, the fitted synonymous-baseline normalizer is written as `normalizer.parquet` using the same path convention.

---

## API reference

---

::: fisseq_data_pipeline.aggregate.AggregateConfig

---

::: fisseq_data_pipeline.aggregate.variant_classification

---

::: fisseq_data_pipeline.aggregate.BaseAggregator

---

::: fisseq_data_pipeline.aggregate.MeanAggregator

---

::: fisseq_data_pipeline.aggregate.MedianAggregator

---

::: fisseq_data_pipeline.aggregate.MADAggregator

---

::: fisseq_data_pipeline.aggregate.StdAggregator

---

::: fisseq_data_pipeline.aggregate.EMDAggregator

---

::: fisseq_data_pipeline.aggregate.KSAggregator

---

::: fisseq_data_pipeline.aggregate.QQCorrelationAggregator

---

::: fisseq_data_pipeline.aggregate.AUROCAggregator

---

::: fisseq_data_pipeline.aggregate.MultiAggregator

---

::: fisseq_data_pipeline.aggregate.aggregate

---

::: fisseq_data_pipeline.aggregate.main

---
