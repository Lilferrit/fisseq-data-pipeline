"""Per-variant multiclass batch classifier for batch-effect detection.

Hydra entry point ``fisseq-batch-vs-batch``, backing the Nextflow process
``BATCHVSBATCH`` (run once pre- and once post-normalization). Trains one
multiclass XGBoost model per variant to predict batch label, then extracts a
one-vs-rest AUROC and Mann-Whitney p-value for every (variant, batch) pair from the
held-out test split.
"""

import dataclasses
import logging
import pathlib
import traceback
from typing import Optional

import hydra
import numpy as np
import polars as pl
import scipy.stats
import sklearn.metrics
import sklearn.utils
import xgboost as xgb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.log import setup_logging
from .utils.xgbparams import (
    XGBoostConfig,
    get_dmatrix_multiclass,
    get_feature_cols,
    split_indices_stratified,
)


@dataclasses.dataclass
class BvbConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the batch-vs-batch entry point.

    Extends :class:`.config.LabeledInputConfig` with parameters controlling
    per-variant multiclass batch classification.

    Attributes
    ----------
    batch_column : str
        Name of the column identifying batch labels. Defaults to
        ``"meta_batch"``.
    random_state : int
        Random seed for train/test/val splitting. Defaults to ``42``.
    feature_cols : list or None
        Explicit list of feature column names. If ``None``, columns are
        auto-detected by :func:`.xgbparams.get_feature_cols`. Defaults to
        ``None``.
    min_cells : int
        Minimum number of cells a variant must have (across all batches) to
        be profiled. Defaults to ``50``.
    min_batches : int
        Minimum number of unique batches a variant must appear in to be
        profiled. Defaults to ``2``.
    use_parent_name : bool
        If ``True``, derive the batch label from each input file's parent
        directory name rather than its stem. Set this when all input files share
        the same filename but live in different subdirectories (e.g. the pre-QC
        glob ``qc_filter/*/filtered_cells.parquet``). Defaults to ``False``.
    xgboost : XGBoostConfig
        XGBoost training configuration. Defaults to :class:`.xgbparams.XGBoostConfig`.
    """

    batch_column: str = "meta_batch"
    random_state: int = 42
    feature_cols: Optional[list] = None
    min_cells: int = 50
    min_batches: int = 2
    use_parent_name: bool = False
    xgboost: XGBoostConfig = dataclasses.field(default_factory=XGBoostConfig)


_cs = ConfigStore.instance()
_cs.store(name="bvb_main", node=BvbConfig)


def train_test_val_split(
    data_df: pl.DataFrame,
    cfg: DictConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a feature DataFrame into train, test, and validation sets.

    Stratification uses a composite key of ``label_column + "_" + batch_column``
    so that each variant's rows in every split span all batches.

    Parameters
    ----------
    data_df : pl.DataFrame
        Full feature DataFrame.
    cfg : DictConfig
        Config supplying ``label_column``, ``batch_column``, ``feature_cols``,
        and ``random_state``.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        ``(train, test, val)`` DataFrames containing feature columns,
        ``label_column``, and ``batch_column``.
    """
    label_col = cfg.label_column
    batch_col = cfg.batch_column

    feature_cols = (
        list(cfg.feature_cols)
        if cfg.feature_cols is not None
        else get_feature_cols(data_df)
    )
    select_cols = feature_cols + [label_col, batch_col]

    data_df = data_df.select(select_cols)
    data_df = data_df.filter(
        pl.col(label_col).is_not_null() & pl.col(batch_col).is_not_null()
    )
    data_df = data_df.with_row_index("__idx__")

    composite = (
        data_df.get_column(label_col).cast(pl.Utf8)
        + "_"
        + data_df.get_column(batch_col).cast(pl.Utf8)
    ).to_numpy()

    train_idx, test_idx, val_idx = split_indices_stratified(composite, cfg.random_state)

    def select_rows(idx: np.ndarray) -> pl.DataFrame:
        return data_df.filter(pl.col("__idx__").is_in(idx)).drop("__idx__")

    return select_rows(train_idx), select_rows(test_idx), select_rows(val_idx)


def train_batch_classifier(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: list[str],
    batch_col: str,
    classes: list[str],
    cfg: DictConfig,
) -> xgb.Booster:
    """
    Train a multiclass XGBoost classifier to predict batch label.

    Mirrors :func:`.ovwt.train_xgboost` in structure: same hyperparameter
    source, same early-stopping pattern, same sample-weight option.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training split containing ``feature_cols`` and ``batch_col``.
    val_df : pl.DataFrame
        Validation split used for early stopping.
    feature_cols : list[str]
        Names of the feature columns.
    batch_col : str
        Name of the batch label column.
    classes : list[str]
        Ordered list of unique batch names (``classes[i]`` maps to integer
        label ``i``).
    cfg : DictConfig
        Config supplying ``random_state`` and the ``xgboost`` sub-config.

    Returns
    -------
    xgb.Booster
        Trained XGBoost booster at the best iteration.
    """
    dtrain, _ = get_dmatrix_multiclass(train_df, feature_cols, batch_col)
    dval, _ = get_dmatrix_multiclass(val_df, feature_cols, batch_col)

    if cfg.xgboost.weigh_samples:
        y_train = dtrain.get_label().astype(int)
        sample_weight = sklearn.utils.compute_sample_weight("balanced", y_train)
        dtrain.set_weight(sample_weight)

    params = dict(cfg.xgboost.params)
    params["objective"] = "multi:softprob"
    params["num_class"] = len(classes)
    params["eval_metric"] = "mlogloss"
    params["seed"] = cfg.random_state

    return xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.xgboost.num_boost_round,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=cfg.xgboost.early_stopping_rounds,
        verbose_eval=True,
    )


def extract_ovr_stats(
    model: xgb.Booster,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    batch_col: str,
    classes: list[str],
) -> list[dict]:
    """
    Compute one-vs-rest ROC AUC and Mann-Whitney p-value for each batch.

    Uses the predicted probabilities from a single multiclass model trained on
    all batches for one variant. For batch ``k``, the OvR score is
    ``P(batch=k)`` and the Mann-Whitney U test compares in-batch scores against
    out-of-batch scores (one-sided: in-batch scores are stochastically greater).

    Parameters
    ----------
    model : xgb.Booster
        Trained multiclass booster (``multi:softprob`` objective).
    test_df : pl.DataFrame
        Held-out test split containing ``feature_cols`` and ``batch_col``.
    feature_cols : list[str]
        Names of the feature columns.
    batch_col : str
        Name of the batch label column.
    classes : list[str]
        Ordered class names matching the training encoding.

    Returns
    -------
    list[dict]
        One dict per batch with keys ``batch``, ``auroc``, ``mw_pvalue``,
        ``n_batch_cells``, ``n_cells``.
    """
    dtest, _ = get_dmatrix_multiclass(test_df, feature_cols, batch_col)
    raw = model.predict(dtest)
    n_rows = len(test_df)
    n_classes = len(classes)
    probs = raw.reshape(n_rows, n_classes)

    batch_labels = dtest.get_label().astype(int)
    results = []
    for k, batch_name in enumerate(classes):
        scores = probs[:, k]
        y_true = (batch_labels == k).astype(int)
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        if n_pos < 1 or n_neg < 1:
            logging.warning(
                "Batch '%s' has %d positive and %d negative cells in test split; skipping",
                batch_name,
                n_pos,
                n_neg,
            )
            continue
        auroc = sklearn.metrics.roc_auc_score(y_true, scores)
        _, pvalue = scipy.stats.mannwhitneyu(
            scores[y_true == 1], scores[y_true == 0], alternative="greater"
        )
        results.append(
            {
                "batch": batch_name,
                "auroc": auroc,
                "mw_pvalue": pvalue,
                "n_batch_cells": n_pos,
                "n_cells": len(y_true),
            }
        )
    return results


def profile_variant(
    variant: str,
    train_all: pl.DataFrame,
    test_all: pl.DataFrame,
    val_all: pl.DataFrame,
    feature_cols: list[str],
    cfg: DictConfig,
) -> list[dict]:
    """
    Train one multiclass model for ``variant`` and return per-batch OvR stats.

    Subsets each global split to rows where ``label_column == variant``, checks
    eligibility, trains a batch classifier, and extracts one-vs-rest statistics.

    Parameters
    ----------
    variant : str
        Variant label to profile.
    train_all : pl.DataFrame
        Full training split (all variants).
    test_all : pl.DataFrame
        Full test split (all variants).
    val_all : pl.DataFrame
        Full validation split (all variants).
    feature_cols : list[str]
        Names of the feature columns.
    cfg : DictConfig
        Config supplying ``label_column``, ``batch_column``, ``min_cells``,
        ``min_batches``, and XGBoost settings.

    Returns
    -------
    list[dict]
        Per-batch result dicts with a ``variant`` key prepended. Empty list
        if the variant does not meet eligibility criteria.
    """
    label_col = cfg.label_column
    batch_col = cfg.batch_column

    keep = pl.col(label_col) == variant
    train = train_all.filter(keep)
    test = test_all.filter(keep)
    val = val_all.filter(keep)

    n_cells = len(train) + len(test) + len(val)
    unique_batches = train.get_column(batch_col).unique().to_list()
    n_batches = len(unique_batches)

    if n_cells < cfg.min_cells:
        logging.warning(
            "Variant '%s' has %d cells (< min_cells=%d); skipping",
            variant,
            n_cells,
            cfg.min_cells,
        )
        return []
    if n_batches < cfg.min_batches:
        logging.warning(
            "Variant '%s' has %d unique batch(es) in train split (< min_batches=%d); skipping",
            variant,
            n_batches,
            cfg.min_batches,
        )
        return []

    classes = sorted(unique_batches)
    logging.info(
        "Variant '%s': %d cells, %d batches — %s",
        variant,
        n_cells,
        n_batches,
        classes,
    )

    model = train_batch_classifier(train, val, feature_cols, batch_col, classes, cfg)
    stats = extract_ovr_stats(model, test, feature_cols, batch_col, classes)
    return [{"variant": variant, **s} for s in stats]


@hydra.main(version_base=None, config_path=None, config_name="bvb_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: per-variant multiclass batch-effect detection.

    Steps
    -----
    1. Read the feature file at ``cfg.input_file``.
    2. Split into train/test/val via :func:`train_test_val_split`, stratified
       on a composite (variant, batch) label.
    3. For each non-null variant, train a multiclass XGBoost model to predict
       batch label via :func:`profile_variant`. Variants that raise an exception
       are skipped with a warning.
    4. Write per-(variant, batch) statistics to ``results.parquet``.

    Output files
    ------------
    - ``{output_dir}/results.parquet`` — columns: ``variant``, ``batch``,
      ``auroc``, ``mw_pvalue``, ``n_batch_cells``, ``n_cells``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.batchvsbatch \\
            output_dir=./out \\
            input_file=data/features.parquet \\
            batch_column=meta_batch
    """
    bvb_cfg: BvbConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(bvb_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bvb_cfg.output_dir = output_dir
    setup_logging(bvb_cfg, "batchvsbatch")

    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logging.info("Loading input from %s", cfg.input_file)
    feature_df = load_batches(cfg.input_file, use_parent_name=bvb_cfg.use_parent_name)[
        0
    ].collect()

    feature_cols = (
        list(cfg.feature_cols)
        if cfg.feature_cols is not None
        else get_feature_cols(feature_df)
    )
    logging.info("Using %d feature column(s)", len(feature_cols))

    train_all, test_all, val_all = train_test_val_split(feature_df, cfg)
    logging.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_all),
        len(val_all),
        len(test_all),
    )

    variants = [
        v
        for v in feature_df.get_column(cfg.label_column).drop_nulls().unique().to_list()
    ]
    logging.info("Found %d variant(s) to profile", len(variants))

    all_results = []
    for variant in variants:
        logging.info("Profiling variant '%s'", variant)
        try:
            rows = profile_variant(
                variant, train_all, test_all, val_all, feature_cols, cfg
            )
        except Exception:
            logging.warning(
                "Failed to profile variant '%s', skipping:\n%s",
                variant,
                traceback.format_exc(),
            )
            continue
        all_results.extend(rows)

    results_df = (
        pl.DataFrame(all_results)
        if all_results
        else pl.DataFrame(
            {
                "variant": [],
                "batch": [],
                "auroc": [],
                "mw_pvalue": [],
                "n_batch_cells": [],
                "n_cells": [],
            }
        )
    )

    results_path = output_dir / "results.parquet"
    results_df.write_parquet(results_path)
    logging.info("Results written to %s", results_path)
    logging.info("Done")


if __name__ == "__main__":
    main()
