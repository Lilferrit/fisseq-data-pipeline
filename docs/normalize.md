# Normalize

The `fisseq_data_pipeline.normalize` module fits and applies **z-score normalization** to cell-level feature matrices. By default, normalization statistics are computed only from wild-type (WT) control cells, ensuring that biological variation in non-control variants is captured rather than normalized away.

---

## Overview

- **`Normalizer`** — Dataclass storing per-feature means and standard deviations. Supports `from_lazyframe`, `apply`, `save`, and `load`.
- **`add_control_indicator_column`** — Annotates a LazyFrame with a boolean `meta_is_control` column using a configurable SQL predicate.
- **`NormalizeConfig`** — Hydra structured config extending `AppConfig`.
- **`main`** — Hydra entry point: reads a parquet file, annotates controls, fits and applies a `Normalizer`, writes output.

---

## Example usage

```python
import polars as pl
from fisseq_data_pipeline.normalize import Normalizer, NormalizeConfig, add_control_indicator_column

lf = pl.scan_parquet("cells.parquet")

cfg = NormalizeConfig(output_dir="/tmp", input_file="cells.parquet")
lf = add_control_indicator_column(lf, cfg)

normalizer = Normalizer.from_lazyframe(lf, fit_only_on_control=True)
normalized_lf = normalizer.apply(lf)

# Persist for later reuse
normalizer.save("normalizer.parquet")

# Reload
loaded = Normalizer.load("normalizer.parquet")
```

---

## Notes

- Feature columns are all columns whose names do **not** match `^meta_.*$`.
- Columns with zero or near-zero variance have their std stored as `None` and produce `null` values after `apply()` — use this to identify and drop uninformative features.
- `NaN` inputs are converted to `null` before normalization; any `NaN` outputs (from division by a `None` std) are also converted to `null`.
- Metadata columns (prefixed `meta_`) pass through `apply()` unchanged.

---

## CLI

```bash
python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    input_file=data/cells.parquet \
    control_sample_query="meta_aa_changes = 'WT'" \
    save_normalizer=true
```

Output path:

- `output_root` set → `{output_root}.{stem}.{ext}`
- `output_root` not set → `{output_dir}/{filename}` (same name as input)

When `save_normalizer=true`, the fitted normalizer is written as `normalizer.parquet` using the same path convention.

---

## API reference

---

::: fisseq_data_pipeline.normalize.Normalizer

---

::: fisseq_data_pipeline.normalize.NormalizeConfig

---

::: fisseq_data_pipeline.normalize.add_control_indicator_column

---

::: fisseq_data_pipeline.normalize.main

---
