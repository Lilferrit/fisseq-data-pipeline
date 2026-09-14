# Global OvWT

`python -m fisseq_data_pipeline.globalovwt` (Nextflow process `GLOBAL_OVWT`)
combines several experiments' [OvWT](ovwt.md) scores into one cross-experiment
distinguishability score per variant. It runs once per active entry in
`global_channels`, over that channel's member experiments only.

## Why two steps

Raw AUROC is **not comparable across experiments**: differing cell counts,
feature quality, and batch effects all shift where a genuinely-neutral variant's
classifier score sits. Taking a plain median of raw AUROC across experiments
would therefore pool numbers that do not mean the same thing.

So the stage does two things, in order:

1. **Score correction.** Per experiment, z-score `auroc_pooled` and
   `auroc_median_barcode` against *that experiment's own synonymous variants*.
   Synonymous variants are the natural neutral baseline, so this re-centers each
   experiment on its own null.
2. **Aggregation.** Take the cross-experiment **median** of the z-scored values.

Both halves reuse existing machinery unchanged:
`aggregate.variant_classification` flags synonymous, untagged labels as
controls, and `Normalizer.from_lazyframe(..., fit_only_on_control=True)` does
the z-scoring. `Normalizer.apply` needs no changes either — it operates on
`FEATURE_SELECTOR` (exclude `meta_*`), which already matches exactly
`auroc_pooled`/`auroc_median_barcode` and excludes `meta_n_barcodes` /
`meta_n_cells`.

!!! note "Each experiment needs at least two synonymous variants"
    The per-experiment normalizer fits on synonymous rows. A single synonymous
    variant makes the standard deviation (ddof=1) undefined, which nulls every
    score for that experiment.

## Relationship to the removed `OVWT_GLOBAL`

This **replaces**, rather than reimplements, the old `OVWT_GLOBAL` stage. That
one fit a single classifier on cells pooled across every batch. Aggregating
per-experiment scores instead keeps each experiment's own wildtype and
synonymous baselines intact rather than blending them into one.

## Config fields

Extends `AppConfig` — see the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `batch_stems` | **required** | This channel's experiment identifiers, one per contributing `results.parquet`, in the same order the files were staged. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |

Input files are read as `res_input_*.parquet`, the `stageAs` pattern the
Nextflow module uses so that identically-named `results.parquet` files do not
collide. Nextflow substitutes the `*` with an *empty string* when exactly one
file is staged, so a single-experiment channel yields `res_input_.parquet` —
`utils.nextflow_staging.reconstruct_staged_paths` encodes that rule.

## Output file

`global_scores.parquet` — one row per variant:

| Column | Meaning |
| ------ | ------- |
| `label_column` | Variant label. |
| `meta_median_auroc_pooled` | Cross-experiment median of the z-scored `auroc_pooled`. |
| `meta_median_auroc_median_barcode` | Cross-experiment median of the z-scored `auroc_median_barcode`. |
| `meta_num_experiments` | How many experiments contributed a non-null value for this variant. |

## Example

```bash
uv run python -m fisseq_data_pipeline.globalovwt \
    output_dir=./out \
    'batch_stems=[plate1,plate2]' \
    label_column=meta_aa_changes
```
