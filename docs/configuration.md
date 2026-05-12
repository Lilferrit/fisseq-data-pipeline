# Configuration

The pipeline uses [Hydra](https://hydra.cc) structured configs. Each entry point defines its own config dataclass that extends `AppConfig`. All fields can be overridden on the command line using Hydra's key=value syntax.

---

## `AppConfig`

Shared base configuration inherited by all module configs.

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `output_dir` | `str` | **required** | Directory where output files and logs are written. |
| `output_root` | `str \| None` | `None` | Optional prefix for output file names. When set, files are named `{output_root}.{stem}.{ext}` instead of `{output_dir}/{filename}`. |
| `log_level` | `str` | `"info"` | Logging verbosity: `debug`, `info`, `warning`, `error`, `critical`. |

---

## `NormalizeConfig`

Extends `AppConfig` with normalization-specific fields.

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `input_file` | `str` | **required** | Path to the input parquet file. |
| `control_sample_query` | `str` | `"meta_aa_changes = 'WT'"` | SQL-like WHERE clause identifying control rows used to fit the normalizer. |
| `save_normalizer` | `bool` | `True` | Whether to write the fitted `Normalizer` to `normalizer.parquet`. |

---

## `AggregateConfig`

Extends `AppConfig` with aggregation-specific fields.

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `input_file` | `str` | **required** | Path to the cell-level normalized parquet file. |
| `label_column` | `str` | `"meta_aa_changes"` | Column identifying variant labels, used for group-by and synonymous classification. |
| `aggregator` | `str` | `"multi"` | Aggregation method. One of: `mean`, `median`, `MAD`, `std`, `EMD`, `KS`, `QQ`, `AUROC`, `multi`. |
| `save_normalizer` | `bool` | `True` | Whether to write the synonymous-baseline normalizer to `normalizer.parquet`. |

---

## Command-line overrides

Any field can be overridden with `key=value` on the command line:

```bash
python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    input_file=data/cells.parquet \
    control_sample_query="meta_treatment = 'DMSO'" \
    log_level=debug
```

```bash
python -m fisseq_data_pipeline.aggregate \
    output_dir=./out \
    input_file=out/cells.parquet \
    aggregator=KS \
    label_column=meta_treatment
```
