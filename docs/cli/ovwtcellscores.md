# OvWT Cell Scores

`python -m fisseq_data_pipeline.ovwtcellscores` (Nextflow process `OVWT_CELLSCORES_BATCHWISE`) scores
every cell against each trained one-vs-wildtype model produced by
[`python -m fisseq_data_pipeline.ovwt`](ovwt.md). `input_file` accepts either a full feature parquet or a
split index parquet (as written by `python -m fisseq_data_pipeline.ovwt` when `save_splits=true`) — the
file type is auto-detected by its schema.

## Config fields

Extends `LabeledInputConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Full feature parquet, or a split index parquet from `python -m fisseq_data_pipeline.ovwt`. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `models_path` | **required** | Path to the pickled `dict[str, xgb.Booster]` produced by `python -m fisseq_data_pipeline.ovwt`. |
| `wt_label` | `"WT"` | Label string identifying wildtype cells. |
| `batch_size` | `10000` | Number of rows to process per batch when iterating over the input. |

## Output files

- `{output_dir}/cell_scores.parquet` — one column per variant model, plus all
  `meta_*` columns from the input.

## Example

```bash
uv run python -m fisseq_data_pipeline.ovwtcellscores \
    output_dir=./out \
    input_file=out/features.parquet \
    models_path=out/models.pkl
```

See [API Reference: ovwtcellscores](../api/ovwtcellscores.md) for full function
documentation.
