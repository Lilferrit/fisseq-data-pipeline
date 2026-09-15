"""OVWT_BATCHWISE: k-fold cross-validated one-vs-wildtype variant scoring.

Ported back from fisseq-embeddings-pipeline's ``ovwt.py``, which was itself
adapted from this module's own earlier implementation. The round trip replaced
the single 80/10/10 train/val/test split with k-fold cross-validation
stratified jointly on ``(meta_barcode, is_wt)``, so every cell gets an
out-of-fold score and every variant gets two distinguishability numbers
instead of one:

- ``auroc_pooled`` -- over all of the variant's cells at once.
- ``auroc_median_barcode`` -- each of the variant's barcodes scored separately
  against the full wildtype set, then medianed. This surfaces whether a
  variant's apparent distinguishability is broad-based across its barcodes or
  driven by one or two outliers, which a single pooled number hides.

**One deliberate divergence from the reference implementation.** There, the
features fed to the classifier are z-scored against the experiment's own
*synonymous* variants. Here they are ``NORMALIZE``'s output, z-scored against
*wildtype* cells -- this pipeline's cell-level normalization is unchanged. The
synonymous re-centering still happens, downstream and on the AUROCs rather than
on the features, in :mod:`fisseq_data_pipeline.globalovwt`. Do not "fix" this
by adding a second normalizer fit here.

The feature-filtered and barcode-filtered variants of this stage are gone along
with ANOVA_BLOCKLIST and BARCODE_BLOCKLIST, as is the separate
``ovwtcellscores`` pass -- ``cell_scores.parquet`` below is emitted directly.
"""

import dataclasses
import logging
import pathlib
import pickle
from collections import Counter
from typing import Optional

import hydra
import numpy as np
import polars as pl
import sklearn.calibration
import sklearn.metrics
import sklearn.model_selection
import xgboost as xgb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.constants import FEATURE_SELECTOR, META_BARCODE_COL, META_SELECTOR
from .utils.log import setup_logging
from .utils.xgbparams import (
    XGBoostConfig,
    get_dmatrix,
    split_indices_stratified,
    train_binary_xgboost,
)

# Minimum members a (barcode, is_wt) stratum needs before StratifiedKFold's
# outer split is guaranteed not to raise. The *inner* split
# (split_indices_stratified) tolerates strata this leaves behind -- it moves
# singleton strata into its train half rather than raising -- but a variant can
# still be too small for the outer split itself, which is why
# ovwt_batchwise()'s per-variant try/except stays.
_MIN_STRATUM_SIZE = 10

_cs = ConfigStore.instance()


@dataclasses.dataclass
class OvwtConfig(LabeledInputConfig):
    """
    Hydra structured configuration for OVWT_BATCHWISE.

    Extends :class:`~fisseq_data_pipeline.config.input.LabeledInputConfig`
    (``output_dir``, ``output_root``, ``log_level``, ``random_seed``,
    ``input_file``, ``label_column``). Every stochastic step below --
    ``StratifiedKFold``'s shuffle, the inner fit/calibration split, wildtype
    downsampling, and XGBoost's own ``seed`` -- consumes the single shared
    ``random_seed``; there is deliberately no stage-local ``random_state``.

    Attributes
    ----------
    wt_label : str
        Label value in ``label_column`` identifying wildtype cells. Wildtype is
        the positive class, so the trained models predict P(wildtype). Defaults
        to ``"WT"``.
    n_folds : int
        Number of cross-validation folds per variant. Defaults to ``5``.
    calibrate : bool
        If ``True``, fit a per-fold sigmoid (Platt) probability calibrator on a
        slice held out of that fold's training data before scoring its test
        slice. Defaults to ``True``.
    min_cells : int or None
        Minimum number of cells a variant must have to be scored; variants
        below this are dropped before the per-variant loop (wildtype is always
        kept regardless of count). ``None`` disables this filter. Defaults to
        ``250``.
    downsample_wt : bool
        If ``True``, downsample wildtype cells -- barcode-proportionally -- to
        the size of the largest remaining variant group before the per-variant
        loop. Defaults to ``True``.
    xgboost : XGBoostConfig
        Booster hyperparameters and training-loop settings.
    """

    wt_label: str = "WT"
    n_folds: int = 5
    calibrate: bool = True
    min_cells: Optional[int] = 250
    downsample_wt: bool = True
    xgboost: XGBoostConfig = dataclasses.field(default_factory=XGBoostConfig)


_cs.store(name="ovwt_main", node=OvwtConfig)


def predict_binary(
    df: pl.DataFrame, model: xgb.Booster, label_col: str, wt_label: str
) -> np.ndarray:
    """
    Raw predicted P(wildtype) scores for every row of ``df``.

    Thin wrapper around :func:`~fisseq_data_pipeline.utils.xgbparams.get_dmatrix`
    plus ``model.predict`` -- no metric computation, since this pipeline needs
    one raw score per cell rather than an aggregate against known labels.
    Reusing ``get_dmatrix`` keeps scoring identical to the feature handling and
    non-finite-to-NaN masking used to fit the model.

    Parameters
    ----------
    df : pl.DataFrame
        Rows to score. Every non-``label_col`` column is treated as a feature.
    model : xgb.Booster
        A trained booster.
    label_col : str
        Name of the label column, used only to exclude it from the features --
        the true labels themselves are not read.
    wt_label : str
        Wildtype label string, passed through to ``get_dmatrix`` (it only
        affects that function's own label encoding, which this discards).

    Returns
    -------
    np.ndarray
        1-D array of predicted P(wildtype) scores, one per row, in row order.
    """
    return model.predict(get_dmatrix(df, label_col, wt_label))


def filter_min_cells(
    data_df: pl.DataFrame,
    label_col: str,
    wt_label: str,
    min_cells: Optional[int],
) -> pl.DataFrame:
    """
    Remove non-wildtype variant groups with fewer than ``min_cells`` cells.

    Wildtype rows are always retained regardless of count. A no-op when
    ``min_cells`` is ``None``.

    Parameters
    ----------
    data_df : pl.DataFrame
        DataFrame containing all variant and wildtype rows.
    label_col : str
        Name of the label column.
    wt_label : str
        Label string identifying wildtype rows (always kept).
    min_cells : int or None
        Minimum number of cells a variant must have to be retained. ``None``
        disables the filter entirely.

    Returns
    -------
    pl.DataFrame
        DataFrame with small variant groups removed.
    """
    if min_cells is None:
        return data_df

    variant_counts = (
        data_df.filter(pl.col(label_col) != wt_label).group_by(label_col).len()
    )
    keep_labels = (
        variant_counts.filter(pl.col("len") >= min_cells)
        .get_column(label_col)
        .to_list()
    )
    return data_df.filter(
        (pl.col(label_col) == wt_label) | pl.col(label_col).is_in(keep_labels)
    )


def downsample_wildtype(
    data_df: pl.DataFrame,
    label_col: str,
    wt_label: str,
    seed: int,
) -> pl.DataFrame:
    """
    Downsample wildtype rows, barcode-proportionally, to the size of the
    largest remaining non-wildtype variant group.

    If wildtype barcode B holds fraction ``p_B`` of the wildtype pool, roughly
    ``p_B * target`` of its cells are kept, so wildtype barcode proportions are
    preserved rather than sampled uniformly across all wildtype cells. That
    matters here specifically because ``auroc_median_barcode`` scores each
    variant barcode against the whole wildtype set: a uniform draw could
    silently skew the wildtype composition every one of those comparisons is
    measured against. (This replaces an earlier uniform implementation that
    delegated to a generic ``downsample_group_to_target`` helper.)

    Rounding each barcode's target to the nearest integer can leave the final
    wildtype count off the requested target by up to (number of wildtype
    barcodes) cells -- accepted as negligible, not corrected with a
    largest-remainder adjustment.

    Parameters
    ----------
    data_df : pl.DataFrame
        DataFrame containing all variant and wildtype rows, including
        ``meta_barcode``.
    label_col : str
        Name of the label column.
    wt_label : str
        Label string identifying wildtype rows.
    seed : int
        Random seed for sampling.

    Returns
    -------
    pl.DataFrame
        DataFrame with wildtype rows downsampled, or unchanged if already at or
        below the target (including when there are no non-wildtype rows to size
        the target against).
    """
    wt_df = data_df.filter(pl.col(label_col) == wt_label)
    other_df = data_df.filter(pl.col(label_col) != wt_label)
    wt_n = len(wt_df)
    if wt_n == 0:
        return data_df

    target = other_df.group_by(label_col).len().get_column("len").max()
    if target is None or wt_n <= target:
        return data_df

    fraction = target / wt_n
    barcode_targets = (
        wt_df.group_by(META_BARCODE_COL)
        .len()
        .with_columns(
            (pl.col("len") * fraction).round(0).cast(pl.Int64).alias("__target__")
        )
    )
    shuffled = wt_df.sample(fraction=1.0, shuffle=True, seed=seed).join(
        barcode_targets.select([META_BARCODE_COL, "__target__"]),
        on=META_BARCODE_COL,
        how="left",
    )
    row_in_barcode = pl.int_range(pl.len()).over(META_BARCODE_COL)
    kept_wt = shuffled.filter(row_in_barcode < pl.col("__target__")).drop("__target__")
    return pl.concat([other_df, kept_wt])


def _stratification_key(barcodes: np.ndarray, is_wt: np.ndarray) -> np.ndarray:
    """
    Composite ``(barcode, is_wt)`` stratification key, with a rare-stratum
    fallback.

    Any ``(barcode, is_wt)`` stratum with fewer than ``_MIN_STRATUM_SIZE``
    members collapses into a shared ``"rare|wt"`` / ``"rare|variant"`` bucket --
    the wt/variant half of the key is preserved and never merged across, so
    barcode composition can degrade gracefully without ever sacrificing the
    wildtype/variant balance ``StratifiedKFold`` is there to preserve.

    Parameters
    ----------
    barcodes : np.ndarray
        1-D array of barcode strings, one per cell.
    is_wt : np.ndarray
        1-D boolean array, ``True`` for wildtype cells, aligned with
        ``barcodes``.

    Returns
    -------
    np.ndarray
        1-D array of composite stratification key strings.
    """
    half = np.where(is_wt, "wt", "variant")
    raw = np.array([f"{b}|{h}" for b, h in zip(barcodes, half)])
    counts = Counter(raw)
    return np.array(
        [
            s if counts[s] >= _MIN_STRATUM_SIZE else f"rare|{h}"
            for s, h in zip(raw, half)
        ]
    )


def ovwt_batchwise(
    cells_lf: pl.LazyFrame,
    cfg: OvwtConfig,
    feature_selector: pl.Expr = FEATURE_SELECTOR,
) -> "tuple[pl.DataFrame, pl.DataFrame, dict[str, list[tuple[xgb.Booster, Optional[object]]]]]":
    """
    K-fold cross-validated one-vs-wildtype scoring, per variant.

    Every cell in a variant's vs.-wildtype subset gets exactly one out-of-fold
    (OOF) score, which is what makes the per-barcode metric below well-defined.
    Folds are stratified jointly on ``(meta_barcode, is_wt)`` via a composite
    key (see :func:`_stratification_key`), so barcode composition and the
    wildtype/variant balance are both preserved fold to fold.

    A variant whose fold training or evaluation raises -- e.g. a variant with
    too few cells for the outer ``StratifiedKFold``, despite
    :func:`_stratification_key`'s mitigation -- is skipped with a logged
    warning rather than aborting the run. This is load-bearing, not defensive
    decoration: small variants legitimately hit it, and the alternative is
    losing every other variant's results alongside.

    Parameters
    ----------
    cells_lf : pl.LazyFrame
        QC-passed, wildtype-normalized cell-level features (NORMALIZE's
        output).
    cfg : OvwtConfig
        Supplies ``label_column``, ``wt_label``, ``n_folds``, ``calibrate``,
        ``min_cells``, ``downsample_wt``, ``xgboost``, and ``random_seed``.
    feature_selector : pl.Expr
        Polars selector identifying feature columns. Defaults to
        ``FEATURE_SELECTOR`` (every non-``meta_*`` column).

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, dict]
        ``(results, cell_scores, models)``:

        - ``results``: one row per surviving variant, columns
          ``cfg.label_column``, ``auroc_pooled``, ``auroc_median_barcode``,
          ``meta_n_barcodes``, ``meta_n_cells``.
        - ``cell_scores``: ``META_SELECTOR`` columns plus ``score`` (the OOF
          score) and ``meta_variant_scored_against`` -- one row per cell per
          variant it was scored against, so wildtype cells appear once per
          variant.
        - ``models``: one entry per surviving variant, a list of
          ``(model, calibrator_or_None)`` tuples, one per fold.

        If no variant survives pre-filtering, or every variant's loop raises,
        both DataFrames come back empty but correctly schema'd.
    """
    df = cells_lf.collect()
    label_col = cfg.label_column
    wt_label = cfg.wt_label

    df = filter_min_cells(df, label_col, wt_label, cfg.min_cells)
    if cfg.downsample_wt:
        df = downsample_wildtype(df, label_col, wt_label, cfg.random_seed)

    feature_cols = df.select(feature_selector).columns
    variants = (
        df.filter(pl.col(label_col) != wt_label)
        .get_column(label_col)
        .unique()
        .sort()
        .to_list()
    )
    logging.info("Scoring %d variant(s) against %r", len(variants), wt_label)

    # train_binary_xgboost does `dict(cfg.xgboost.params)` internally, which
    # raises on a plain dataclass -- OmegaConf.structured() produces a properly
    # nested DictConfig all the way down. Computed once and reused for every
    # fold/variant below, since cfg does not change across the loop.
    xgb_cfg = OmegaConf.structured(cfg)

    per_variant_results: list[dict] = []
    per_cell_scores: list[pl.DataFrame] = []
    models: "dict[str, list[tuple[xgb.Booster, Optional[object]]]]" = {}

    for variant in variants:
        try:
            subset = df.filter(pl.col(label_col).is_in([variant, wt_label]))
            is_wt = (subset.get_column(label_col) == wt_label).to_numpy()
            barcodes = subset.get_column(META_BARCODE_COL).to_numpy().astype(str)
            strata = _stratification_key(barcodes, is_wt)

            splitter = sklearn.model_selection.StratifiedKFold(
                n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_seed
            )
            oof_scores = np.full(len(subset), np.nan)
            fold_models: "list[tuple[xgb.Booster, Optional[object]]]" = []

            for fold_idx, (fit_idx, test_idx) in enumerate(
                splitter.split(subset, strata)
            ):
                fit_df, test_df = subset[fit_idx], subset[test_idx]
                # An 80/20 train/calibration split of the fold's fit rows.
                # There is no third (test) slot: the fold's own test_idx above
                # already serves that role.
                train_pos, calib_pos = split_indices_stratified(
                    strata[fit_idx], cfg.random_seed + fold_idx
                )
                train_df = fit_df[train_pos].select([label_col, *feature_cols])
                calib_df = fit_df[calib_pos].select([label_col, *feature_cols])

                # calib_df does double duty: XGBoost's early-stopping eval set
                # AND the calibrator's fitting set. Not an independent
                # calibration set -- matching the reference implementation.
                model = train_binary_xgboost(train_df, calib_df, xgb_cfg)

                calibrator = None
                if cfg.calibrate:
                    calib_raw = predict_binary(calib_df, model, label_col, wt_label)
                    calib_is_wt = (
                        calib_df.get_column(label_col) == wt_label
                    ).to_numpy()
                    # Private sklearn API. There is no public single-feature
                    # Platt scaler; CalibratedClassifierCV wraps an estimator,
                    # not a raw score vector. Pinned by test coverage rather
                    # than by a version bound -- if a sklearn upgrade breaks
                    # this import, tests/unit/test_ovwt.py fails loudly.
                    calibrator = sklearn.calibration._SigmoidCalibration().fit(
                        calib_raw, calib_is_wt
                    )

                test_raw = predict_binary(
                    test_df.select([label_col, *feature_cols]),
                    model,
                    label_col,
                    wt_label,
                )
                oof_scores[test_idx] = (
                    calibrator.predict(test_raw) if calibrator is not None else test_raw
                )
                fold_models.append((model, calibrator))

            models[variant] = fold_models

            auroc_pooled = float(sklearn.metrics.roc_auc_score(is_wt, oof_scores))

            variant_barcodes = (
                subset.filter(pl.col(label_col) != wt_label)
                .get_column(META_BARCODE_COL)
                .unique()
                .to_list()
            )
            barcode_aurocs = []
            for barcode in variant_barcodes:
                mask = (barcodes == str(barcode)) | is_wt
                barcode_aurocs.append(
                    sklearn.metrics.roc_auc_score(is_wt[mask], oof_scores[mask])
                )
            # None, not float("nan"), so the cross-experiment median in
            # globalovwt.py excludes this cleanly instead of being poisoned by
            # a NaN. Defensive only -- a variant only enters this loop with at
            # least one barcode of its own.
            auroc_median_barcode = (
                float(np.median(barcode_aurocs)) if barcode_aurocs else None
            )

            per_variant_results.append(
                {
                    label_col: variant,
                    "auroc_pooled": auroc_pooled,
                    "auroc_median_barcode": auroc_median_barcode,
                    "meta_n_barcodes": len(variant_barcodes),
                    "meta_n_cells": len(subset),
                }
            )
            per_cell_scores.append(
                subset.select(META_SELECTOR).with_columns(
                    pl.Series("score", oof_scores),
                    pl.lit(variant).alias("meta_variant_scored_against"),
                )
            )
        except Exception:
            logging.warning(
                "Skipping variant %r due to an error during training/evaluation:",
                variant,
                exc_info=True,
            )
            continue

    if not per_variant_results:
        results_df = pl.DataFrame(
            schema={
                label_col: pl.String,
                "auroc_pooled": pl.Float64,
                "auroc_median_barcode": pl.Float64,
                "meta_n_barcodes": pl.Int64,
                "meta_n_cells": pl.Int64,
            }
        )
        cell_scores_schema = {c: df.schema[c] for c in df.select(META_SELECTOR).columns}
        cell_scores_schema["score"] = pl.Float64
        cell_scores_schema["meta_variant_scored_against"] = pl.String
        cell_scores_df = pl.DataFrame(schema=cell_scores_schema)
    else:
        results_df = pl.DataFrame(per_variant_results)
        cell_scores_df = pl.concat(per_cell_scores)

    return results_df, cell_scores_df, models


@hydra.main(version_base=None, config_path=None, config_name="ovwt_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: k-fold one-vs-wildtype scoring for every variant in an
    experiment.

    Output files
    ------------
    - ``{output_dir}/{prefix}results.parquet``
    - ``{output_dir}/{prefix}cell_scores.parquet``
    - ``{output_dir}/{prefix}models.pkl``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.ovwt \\
            output_dir=./out \\
            input_file=out/normalized.parquet \\
            n_folds=5 \\
            calibrate=true \\
            min_cells=250 \\
            downsample_wt=true \\
            random_seed=0
    """
    ovwt_cfg: OvwtConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ovwt_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ovwt_cfg.output_dir = str(output_dir)
    setup_logging(ovwt_cfg, "ovwt")

    prefix = f"{ovwt_cfg.output_root}." if ovwt_cfg.output_root is not None else ""

    logging.info("Loading normalized cells from %s", ovwt_cfg.input_file)
    cells_lf = load_batches(ovwt_cfg.input_file)[0]

    logging.info(
        "Running %d-fold one-vs-wildtype scoring (calibrate=%s, min_cells=%s)",
        ovwt_cfg.n_folds,
        ovwt_cfg.calibrate,
        ovwt_cfg.min_cells,
    )
    results_df, cell_scores_df, models = ovwt_batchwise(cells_lf, ovwt_cfg)

    results_path = output_dir / f"{prefix}results.parquet"
    logging.info("Writing %s", results_path)
    results_df.write_parquet(results_path)

    cell_scores_path = output_dir / f"{prefix}cell_scores.parquet"
    logging.info("Writing %s", cell_scores_path)
    cell_scores_df.write_parquet(cell_scores_path)

    models_path = output_dir / f"{prefix}models.pkl"
    logging.info("Writing %s", models_path)
    with open(models_path, "wb") as f:
        pickle.dump(models, f)

    logging.info("Done")


if __name__ == "__main__":
    main()
