# ANOVA Block-list

`python -m fisseq_data_pipeline.anovablocklist` (Nextflow process `ANOVA_BLOCKLIST`, always runs)
derives a feature block-list from the output of `python -m fisseq_data_pipeline.anova` (run against
normalized cells). A feature is blocked (`feature_ok = false`) when its
ANOVA `p_value` is strictly less than `pvalue_threshold`, i.e. when a
statistically significant batch effect was detected; otherwise it is kept
(`feature_ok = true`). The output follows the same `feature`/`feature_ok`
convention as the correlation-based blocklist (`python -m fisseq_data_pipeline.blocklist`) and the
`block_list_file` fields on `python -m fisseq_data_pipeline.aggregate`, `python -m fisseq_data_pipeline.featureselect`,
`python -m fisseq_data_pipeline.ovwt`, and `python -m fisseq_data_pipeline.batchvsbatch`.

## Config fields

Extends `AppConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `anova_file` | **required** | Path to the ANOVA results parquet file (output of `python -m fisseq_data_pipeline.anova`), with columns `feature`, `f_value`, `p_value`. |
| `pvalue_threshold` | `0.05` | A feature is blocked when its `p_value` is strictly less than this threshold. |

## Output files

- `{output_dir}/anova_blocklist.parquet` — one row per input feature, with
  columns `feature`, `p_value`, `feature_ok`.

## Example

```bash
uv run python -m fisseq_data_pipeline.anovablocklist \
    output_dir=./out \
    anova_file=out/anova.parquet \
    pvalue_threshold=0.05
```

See [API Reference: anovablocklist](../api/anovablocklist.md) for full
function documentation.
