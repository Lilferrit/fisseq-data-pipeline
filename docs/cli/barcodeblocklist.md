# Barcode Block-list

`python -m fisseq_data_pipeline.barcodeblocklist` (Nextflow process `BARCODE_BLOCKLIST`, run once
per batch, same cadence as [Check Barcodes](checkbarcodes.md)) derives a
barcode block-list from the output of `python -m fisseq_data_pipeline.checkbarcodes`. Each
barcode's `p_adj` values are pooled across every comparison it took part in
(a barcode can appear as either `barcode` or `comparison_barcode` in a given
pair, so both columns are unioned before aggregating) and summarized as
their median. A barcode is blocked (`barcode_ok = false`) when that median
is strictly less than `pvalue_threshold`, i.e. its cells score anomalously
relative to the variant's other barcodes; otherwise it is kept
(`barcode_ok = true`). The output follows the same key/`_ok` convention as
[ANOVA Block-list](anovablocklist.md), and is consumed by the
`barcode_block_list_file` field on [OvWT](ovwt.md).

## Config fields

Extends `AppConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `check_barcodes_file` | **required** | Path to a single batch's CHECK_BARCODES results parquet (output of `python -m fisseq_data_pipeline.checkbarcodes`), with columns `barcode`, `comparison_barcode`, `p_adj`. |
| `pvalue_threshold` | `0.05` | A barcode is blocked when the median of its `p_adj` values is strictly less than this threshold. |

## Output files

- `{output_dir}/barcode_blocklist.parquet` — one row per distinct barcode,
  with columns `barcode`, `p_adj` (median across all comparisons), and
  `barcode_ok`.

## Example

```bash
uv run python -m fisseq_data_pipeline.barcodeblocklist \
    output_dir=./out \
    check_barcodes_file=out/results.parquet \
    pvalue_threshold=0.05
```

See [API Reference: barcodeblocklist](../api/barcodeblocklist.md) for full
function documentation.
