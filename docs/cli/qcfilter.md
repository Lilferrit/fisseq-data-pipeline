# QC Filter

`fisseq-qc-filter` (Nextflow process `QC_FILTER`) reads one or more raw CSV/Parquet
cell files, renames columns to canonical `meta_*` names, and applies three
sequential filters:

1. **Edit distance** — drops cells with edit distance greater than
   `edit_distance_threshold`.
2. **Barcode cell count** — drops barcodes represented by fewer than
   `bc_threshold` cells.
3. **Variant barcode count** — drops variants supported by fewer than
   `variant_bc_threshold` distinct barcodes.

## Config fields

Extends the common `output_dir` / `output_root` / `log_level` fields (see
[Common config fields](#common-config-fields) below).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cell_files` | **required** | Path or list of paths to raw cell files (CSV or Parquet). |
| `bc_threshold` | `10` | Minimum cells required per barcode. |
| `variant_bc_threshold` | `4` | Minimum distinct barcodes required per variant. |
| `edit_distance_threshold` | `1` | Maximum allowed edit distance. |
| `barcode_col_name` | `"upBarcode"` | Input column name for cell barcodes. |
| `aa_changes_col_name` | `"aaChanges"` | Input column name for amino-acid change labels. |
| `edit_distance_col_name` | `"editDistance"` | Input column name for edit distances. |
| `label_column` | `"meta_aa_changes"` | Output column name for the variant label. |

## Output files

Written to `output_dir` (each prefixed `{output_root}.` when `output_root` is set):

- `filtered_cells.parquet` — cells passing all three filters
- `barcode_counts.parquet` — per-barcode cell counts and pass/fail flags
- `variants_per_barcode.parquet` — per-variant barcode counts and pass/fail flags

## Example

```bash
uv run fisseq-qc-filter \
    output_dir=./out \
    'cell_files=[data/plate1.parquet, data/plate2.parquet]' \
    bc_threshold=10 \
    variant_bc_threshold=4 \
    edit_distance_threshold=1
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |

See [API Reference: qcfilter](../api/qcfilter.md) for full function documentation.
