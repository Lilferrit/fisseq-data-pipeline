# Normalize

`fisseq-normalize` (Nextflow process `NORMALIZE`) fits per-feature z-score
statistics on control (WT) cells and applies them across the full dataset. Control
rows are identified via a SQL WHERE clause evaluated against the input frame,
making it easy to adapt to non-standard control labels without code changes.
Features with zero variance are stored as `null` in the output.

## Config fields

Extends `InputConfig` (adds `input_file`) plus the
[common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to QC-filtered cell parquet. |
| `control_sample_query` | `"meta_aa_changes = 'WT'"` | SQL-like WHERE clause identifying control rows used to fit the normalizer. |
| `save_normalizer` | `true` | Write the fitted normalizer to `normalizer.parquet`. |

## Output files

- `{output_dir}/{filename}` (same name as the input file), or
  `{output_root}.{stem}.{ext}` when `output_root` is set — normalized cell data
  with an added `meta_is_control` boolean column
- `normalizer.parquet` (or `{output_root}.normalizer.parquet`) — fitted
  normalizer, when `save_normalizer=true`

## Example

```bash
uv run fisseq-normalize \
    output_dir=./out \
    input_file=out/filtered_cells.parquet
```

See [API Reference: normalize](../api/normalize.md) for full function
documentation, including the reusable `Normalizer` class.
