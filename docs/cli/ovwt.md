# One-vs-WT

`python -m fisseq_data_pipeline.ovwt` (Nextflow process `OVWT_BATCHWISE`) scores
how distinguishable each variant is from wildtype, one experiment at a time.

For each non-wildtype variant it runs **cross-validation** over that variant's
cells plus the (optionally downsampled) wildtype pool, training one XGBoost
binary classifier per fold. Because the folds partition the data, every cell
ends up with exactly one *out-of-fold* score — a prediction from a model that
never saw it. That is what makes the per-barcode metric below well defined.

## Two cross-validation modes

`cv_mode` selects how the folds are cut. Both give every cell exactly one
out-of-fold score, and both produce the same two output columns — what changes
is the question those columns answer.

### `"kfold"` (default)

`n_folds` folds, stratified jointly on `(meta_barcode, is_wt)`, so both barcode
composition and the wildtype/variant balance are preserved fold to fold. A
`(barcode, is_wt)` stratum with fewer than 10 members collapses into a shared
`rare|wt` / `rare|variant` bucket; the wildtype/variant half of the key is never
merged across, so barcode resolution degrades gracefully without ever
sacrificing class balance.

Every fold's model has seen every barcode, so the AUROCs measure separability
*within* the barcodes the classifier trained on.

### `"leave_one_barcode_out"`

One fold per variant barcode. Fold *i* holds out **every** cell of the *i*-th
barcode, and the variant's other barcodes do the training — so the model
scoring a barcode has never seen that barcode. Wildtype cells are still split
across the folds (into as many disjoint blocks as there are variant barcodes,
stratified on the same composite key), rather than held out wholesale.

`n_folds` is ignored: a variant gets exactly as many folds as it has barcodes.
A variant with only **one** barcode cannot be scored at all — holding it out
would leave no variant cells to train on — and is skipped with a warning.
`variant_barcode_count_threshold` (default `4`) already makes that rare.

Use this mode to ask whether a variant's signal *generalizes to an unseen
barcode*. A barcode-specific technical artifact inflates the `"kfold"` numbers
invisibly; here it is penalized, because no model is ever trained and scored on
the same barcode.

## Two scores per variant

| Column | Meaning |
| ------ | ------- |
| `auroc_pooled` | AUROC over all of the variant's out-of-fold scores at once. |
| `auroc_median_barcode` | Each of the variant's barcodes scored separately against the **full** wildtype set, then medianed. |

`auroc_median_barcode` exists to surface whether a variant's apparent
distinguishability is broad-based across its barcodes or driven by one or two
outlier barcodes — which a single pooled number hides. It is `null` only in the
defensive case of a variant with no barcodes of its own, and deliberately
`null` rather than `NaN` so the cross-experiment median in
[Global OvWT](globalovwt.md) excludes it cleanly.

## Wildtype downsampling

`downsample_wt: true` shrinks the wildtype pool to the size of the largest
remaining variant group, **barcode-proportionally**: a wildtype barcode holding
fraction `p` of the pool keeps roughly `p × target` of its cells. Preserving
wildtype barcode composition matters specifically because
`auroc_median_barcode` measures every variant barcode against that same
wildtype set — a uniform draw could skew it. Per-barcode rounding can leave the
final count off target by up to (number of wildtype barcodes) cells; this is
accepted rather than corrected.

## Progress logging

Each variant logs a `[i/N]` header (barcode and cell counts), one line per fold
(the held-out barcode under `"leave_one_barcode_out"`, the train/calibration/test
sizes, and that fold's own AUROC), one line per barcode, and a closing summary
with `auroc_pooled`, `auroc_median_barcode` and elapsed time. A fold whose test
slice happens to hold a single class logs `auroc=n/a` rather than failing the
variant.

## Resilience

A variant whose folds raise — most often a stratum too small for the inner
train/calibration split to survive — is skipped with a logged warning rather
than aborting the run. This is load-bearing, not decoration: small variants
legitimately hit it, and the alternative is losing every other variant's
results alongside. If no variant survives, the output files are still written,
empty but correctly typed.

## A note on normalization

OvWT consumes [NORMALIZE](normalize.md)'s output, which is z-scored against
**wildtype** cells. The sibling `fisseq-embeddings-pipeline`, from which this
implementation was ported, instead z-scores its features against **synonymous**
variants before training. That difference is deliberate here: cell-level
wildtype normalization is unchanged, and the synonymous re-centering happens
downstream on the AUROCs instead, in [Global OvWT](globalovwt.md).

## Config fields

Extends `LabeledInputConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to normalized cell-level parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `wt_label` | `"WT"` | Label identifying wildtype cells. Wildtype is the positive class, so models predict P(wildtype). |
| `cv_mode` | `"kfold"` | Fold scheme: `"kfold"` or `"leave_one_barcode_out"`. |
| `n_folds` | `5` | Cross-validation folds per variant, under `"kfold"`. Ignored by `"leave_one_barcode_out"`. |
| `calibrate` | `true` | Fit a per-fold sigmoid (Platt) calibrator on a slice held out of that fold's training data. |
| `min_cells` | `250` | Drop variants with fewer than this many cells before scoring; wildtype is always kept. `null` disables. |
| `downsample_wt` | `true` | Barcode-proportional wildtype downsampling to the largest remaining variant group. |
| `random_seed` | `0` | The shared pipeline seed — drives the fold shuffle, the inner split, wildtype downsampling, and XGBoost's own `seed`. |
| `xgboost.*` | see `XGBoostConfig` | Booster hyperparameters and training-loop settings. |

## Output files

| File | Contents |
| ---- | -------- |
| `results.parquet` | One row per surviving variant: `label_column`, `auroc_pooled`, `auroc_median_barcode`, `meta_n_barcodes`, `meta_n_cells`. |
| `cell_scores.parquet` | One row per cell per variant it was scored against: every `meta_*` column plus `score` (the out-of-fold score) and `meta_variant_scored_against`. Wildtype cells appear once per variant. Join back to the cell table on `meta_cell_index`. |
| `models.pkl` | `dict[variant, list[(Booster, calibrator_or_None)]]` — one tuple per fold, so `n_folds` entries under `"kfold"` and one per barcode under `"leave_one_barcode_out"`. |

## Example

```bash
uv run python -m fisseq_data_pipeline.ovwt \
    output_dir=./out \
    input_file=out/normalized.parquet \
    cv_mode=kfold \
    n_folds=5 \
    calibrate=true \
    min_cells=250 \
    downsample_wt=true \
    random_seed=0
```
