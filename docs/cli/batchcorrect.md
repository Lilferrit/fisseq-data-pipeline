# Batch Correction

`batchcorrect.py` implements two-pass centroid batch correction as two Hydra entry
points, backing the Nextflow processes `BATCH_CORRECT_FIT` and
`BATCH_CORRECT_TRANSFORM`. Fitting computes per-(variant, batch) statistics and
per-variant centroids across all batches; transforming rescales one batch's cells
first to its variant's own centroid, then to the wildtype centroid.

Both configs extend `LabeledInputConfig` plus the
[common config fields](qcfilter.md#common-config-fields).

## `fisseq-batch-correct-fit` (`BATCH_CORRECT_FIT`)

Runs once globally, after all batches have been QC-filtered.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern matching one file per batch. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `wt_label` | `"WT"` | Value of `label_column` identifying wildtype rows. |
| `use_parent_name` | `false` | If `true`, label each batch by its file's parent directory name instead of the file stem — needed when every batch file shares the same name (e.g. `qc_filter/*/filtered_cells.parquet`). |

**Output**: `stats_vb.parquet` (per-(variant, batch) statistics), `centroids.parquet`
(per-variant centroids).

```bash
uv run fisseq-batch-correct-fit \
    output_dir=./out \
    'input_file=data/batches/*.parquet'
```

## `fisseq-batch-correct-transform` (`BATCH_CORRECT_TRANSFORM`)

Runs once per batch.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to one batch's cell-level parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `stats_file` | **required** | Path to the `stats_vb.parquet` written by the fit step. |
| `centroids_file` | **required** | Path to the `centroids.parquet` written by the fit step. |
| `batch` | **required** | Label identifying which batch `input_file` belongs to (passed explicitly, since batch files may share an identical name). |
| `wt_label` | `"WT"` | Value of `label_column` identifying wildtype rows. |

**Output**: `{output_dir}/{filename}` (same name as the input file), or
`{output_root}.{stem}.{ext}` when `output_root` is set.

```bash
uv run fisseq-batch-correct-transform \
    output_dir=./out \
    input_file=data/batch1.parquet \
    batch=batch1 \
    stats_file=./fit/stats_vb.parquet \
    centroids_file=./fit/centroids.parquet
```

See [API Reference: batchcorrect](../api/batchcorrect.md) for full function
documentation, including the `BatchCorrector` class.
