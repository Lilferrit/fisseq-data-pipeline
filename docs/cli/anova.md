# ANOVA

`fisseq-anova` (Nextflow process `ANOVA`, run once against normalized cells and
once against batch-corrected cells) assesses batch effects in cell-level data
using a per-feature one-way ANOVA. `input_file` is treated as a glob pattern;
each matching file becomes one batch (labeled by its filename stem). Restricted
to cells classified as Synonymous and not tagged `:downsampled`, each feature
column's per-batch-group non-null sum, sum of squares, and count are computed
in a single pass (nulls are excluded per feature, so a feature's null pattern
does not affect other features) and used to derive an F-statistic via the
standard one-way ANOVA sum-of-squares decomposition. A closed-form p-value is
computed from the F-distribution's survival function (`scipy.stats.f.sf`),
with no permutation test involved. A feature is skipped (with a logged
warning) if fewer than 2 non-null samples or fewer than 2 batches with
non-null data remain after excluding nulls.

## Config fields

Extends `LabeledInputConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern matching one or more batch parquet files. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |

## Output files

- `{prefix}anova.parquet` — one row per feature, with `feature`, `f_value`,
  and `p_value`. `prefix` is `{output_root}.` when `output_root` is set,
  otherwise empty.

## Example

```bash
uv run fisseq-anova \
    output_dir=./out \
    'input_file=data/batches/*.parquet'
```

See [API Reference: anova](../api/anova.md) for full function documentation.
