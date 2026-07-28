# Check Barcodes

`python -m fisseq_data_pipeline.checkbarcodes` (Nextflow process `CHECK_BARCODES`, run once per batch,
gated by `params.run_check_barcodes`) detects barcodes whose cells score
differently from the rest of the cells carrying the same variant's other
barcodes. `input_file` is a `cell_scores.parquet` produced by
[OvWT Cell Scores](ovwtcellscores.md) (one column per trained variant model,
plus `meta_*` columns). Each cell's own-model score (the value of the column
named by its own `label_column` value) is used as the response variable,
grouped by variant and barcode; for each variant with more than one distinct
barcode remaining after the `min_cells` filter, a pairwise Tukey HSD is run
across its barcodes.

The comparison is fully vectorized: per-(variant, barcode) sufficient
statistics (non-null count, sum, sum of squares) are computed in a single
`group_by(...).agg(...)` query (the same pattern [ANOVA](anova.md) uses), a
pooled within-variant MSE and degrees of freedom are derived from those
statistics, all barcode pairs within a variant are enumerated via one
vectorized self-join, and the studentized-range p-value is computed via
`scipy.stats.studentized_range.sf` — the same closed-form distribution
`statsmodels.stats.multicomp.pairwise_tukeyhsd` uses internally (verified to
match bit-for-bit), applied across every pairwise comparison in every variant
at once, rather than looping `pairwise_tukeyhsd` per variant.

## Config fields

Extends `LabeledInputConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to a `cell_scores.parquet` (wide-format single-cell scores). |
| `label_column` | `"meta_aa_changes"` | Column identifying each cell's variant label. |
| `min_cells` | `10` | Minimum cells required per barcode (within a variant) to include it in the comparison. |
| `alpha` | `0.05` | Family-wise significance level; a barcode pair is flagged (`reject = True`) when its adjusted p-value is below this. |

## Output files

- `results.parquet` — one row per compared barcode pair, with columns
  `variant`, `barcode`, `group_mean`, `comparison_barcode`,
  `comparison_group_mean`, `mean_diff`, `p_adj`, `reject`. Variants with only
  one qualifying barcode are skipped (nothing to compare).

## Example

```bash
uv run python -m fisseq_data_pipeline.checkbarcodes \
    output_dir=./out \
    input_file=out/cell_scores.parquet \
    min_cells=10 \
    alpha=0.05
```

See [API Reference: checkbarcodes](../api/checkbarcodes.md) for full function
documentation.
