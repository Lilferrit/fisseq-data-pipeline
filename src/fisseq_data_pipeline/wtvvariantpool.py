"""Wildtype-barcode-vs-variant-pool XGBoost classification.

Hydra entry point (``python -m fisseq_data_pipeline.wtvvariantpool``), backing the Nextflow
process ``WTVVARIANTPOOL_BATCHWISE``. Pools non-wildtype cells whose classified
variant class (via :func:`.utils.variant.classify_variant`) is in a configurable
set (default ``["Synonymous"]``), then trains one binary XGBoost classifier per
wildtype barcode to distinguish cells belonging to that barcode from cells
belonging to the pool, writing per-barcode AUROC/accuracy results, the
trained models, and per-barcode gain-based feature importance.
"""

import dataclasses
import logging
import pathlib
import pickle
import traceback
from typing import Optional, Union

import hydra
import polars as pl
import sklearn.metrics
import sklearn.utils
import xgboost as xgb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import LabeledInputConfig
from .utils.batches import load_batches
from .utils.constants import META_BARCODE_COL
from .utils.filtering import (
    _exclude_blocked_features,
    downsample_group_to_target,
    drop_small_groups,
)
from .utils.log import setup_logging
from .utils.variant import classify_variant
from .utils.xgbparams import (
    XGBoostConfig,
    get_dmatrix,
    get_feature_cols,
    resolve_feature_importance,
    split_indices_stratified,
)

# Sentinel value written into barcode_column for pooled variant rows, standing
# in for "the other barcode" so get_dmatrix/split_indices_stratified can be
# reused unmodified (see AGENTS.md's note on wtvwt.py's identical trick).
# Chosen to be implausible as a real barcode string; train_test_val_split
# defensively raises if a real barcode ever collides with it.
_POOL_GROUP = "__WTVVARIANTPOOL_POOL__"


@dataclasses.dataclass
class WtvvariantpoolConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the wildtype-vs-variant-pool entry point.

    Extends :class:`.config.LabeledInputConfig` with parameters controlling
    per-barcode-vs-pool XGBoost training.

    Attributes
    ----------
    wt_label : str
        Label string identifying wildtype cells. Defaults to ``"WT"``.
    barcode_column : str
        Name of the column in ``input_file`` identifying each cell's barcode.
        Defaults to :data:`.utils.constants.META_BARCODE_COL` (``"meta_barcode"``).
    random_state : int
        Random seed for train/test/val splitting, and for
        ``downsample_variant_pool``. Defaults to ``42``.
    feature_cols : list or None
        Explicit list of feature column names. If ``None``, columns are
        auto-detected by :func:`.xgbparams.get_feature_cols`. Defaults to
        ``None``.
    min_cells_per_barcode : int
        Minimum number of wildtype cells required for a barcode to be
        included. Barcodes with fewer cells are dropped before splitting.
        Defaults to ``100``.
    variant_classes : list[str]
        Classes (from :func:`.utils.variant.classify_variant`) eligible for
        the pooled non-wildtype set. Defaults to ``["Synonymous"]``.
    downsample_variant_pool : bool, int, or None
        If ``None`` or ``False``, the pool is not downsampled. If ``True``,
        the pool is downsampled to match the size of the largest surviving
        wildtype barcode group. If an integer, the pool is downsampled to
        that exact count (no-op if already at or below the target).
        Defaults to ``None``.
    feature_block_list_file : str or None
        Optional path to a parquet file with at least ``feature`` (str) and
        ``feature_ok`` (bool) columns. Features where ``feature_ok`` is
        ``False`` are excluded (dropped as columns) before splitting/training.
        Defaults to ``None`` (no features blocked).
    xgboost : XGBoostConfig
        XGBoost training configuration. Defaults to :class:`.xgbparams.XGBoostConfig`.
    """

    wt_label: str = "WT"
    barcode_column: str = META_BARCODE_COL
    random_state: int = 42
    feature_cols: Optional[list] = None
    min_cells_per_barcode: int = 100
    variant_classes: list = dataclasses.field(default_factory=lambda: ["Synonymous"])
    downsample_variant_pool: Optional[Union[bool, int]] = None
    feature_block_list_file: Optional[str] = None
    xgboost: XGBoostConfig = dataclasses.field(default_factory=XGBoostConfig)


_cs = ConfigStore.instance()
_cs.store(name="wtvvariantpool_main", node=WtvvariantpoolConfig)


def build_variant_pool(
    data_df: pl.DataFrame,
    label_column: str,
    wt_label: str,
    variant_classes: list[str],
) -> pl.DataFrame:
    """
    Build the pool of non-wildtype cells eligible for barcode-vs-pool training.

    Rows with ``label_column == wt_label`` are excluded first, by
    construction -- not by relying on :func:`.utils.variant.classify_variant`'s
    own ``"WT"`` category, since ``wt_label`` is a configurable string and
    need not literally be ``"WT"``. The remaining rows are classified via
    :func:`.utils.variant.classify_variant` and kept iff the result is in
    ``variant_classes``.

    Parameters
    ----------
    data_df : pl.DataFrame
        Full feature DataFrame containing ``label_column``.
    label_column : str
        Name of the column identifying variant labels.
    wt_label : str
        Label string identifying wildtype cells, excluded from the pool.
    variant_classes : list[str]
        Classes (from :func:`.utils.variant.classify_variant`) eligible for
        the pool.

    Returns
    -------
    pl.DataFrame
        Rows of ``data_df`` belonging to the pool. May be empty if no rows
        match.
    """
    non_wt = data_df.filter(pl.col(label_column) != wt_label)
    classified = non_wt.with_columns(
        pl.col(label_column)
        .map_elements(classify_variant, return_dtype=pl.String)
        .alias("__variant_class__")
    )
    return classified.filter(pl.col("__variant_class__").is_in(variant_classes)).drop(
        "__variant_class__"
    )


def train_test_val_split(
    data_df: pl.DataFrame,
    cfg: DictConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split wildtype cells and pooled variant cells into train, test, and validation sets.

    Restricts wildtype rows (``label_column == wt_label``) to those whose
    barcode has at least ``min_cells_per_barcode`` cells, builds the pooled
    non-wildtype set via :func:`build_variant_pool`, tags pool rows with a
    reserved sentinel value in ``barcode_column``, optionally downsamples the
    pool via ``downsample_variant_pool``, then produces an 80/10/10 split
    stratified jointly by ``barcode_column`` (individual wildtype barcodes
    plus the pool sentinel) so every surviving barcode's rows -- and the
    pool's -- span all three splits.

    Parameters
    ----------
    data_df : pl.DataFrame
        Full feature DataFrame containing feature columns, ``cfg.label_column``,
        and ``cfg.barcode_column``.
    cfg : DictConfig
        Hydra config supplying ``label_column``, ``wt_label``, ``feature_cols``,
        ``feature_block_list_file``, ``barcode_column``, ``min_cells_per_barcode``,
        ``variant_classes``, ``downsample_variant_pool``, and ``random_state``.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        ``(train, test, val)`` DataFrames, each containing feature columns and
        ``barcode_column`` (with pool rows carrying the reserved sentinel
        value).

    Raises
    ------
    ValueError
        If a real wildtype barcode collides with the reserved pool sentinel
        value, or if the variant pool is empty after filtering to
        ``variant_classes``.
    """
    barcode_col = cfg.barcode_column
    if cfg.feature_cols is not None:
        feature_cols = list(cfg.feature_cols)
    else:
        feature_cols = get_feature_cols(data_df)
    feature_cols = _exclude_blocked_features(feature_cols, cfg.feature_block_list_file)
    select_cols = feature_cols + [barcode_col]

    wt_df = data_df.filter(pl.col(cfg.label_column) == cfg.wt_label)
    wt_df = wt_df.select(select_cols)
    wt_df = wt_df.filter(pl.col(barcode_col).is_not_null())
    wt_df = drop_small_groups(wt_df, barcode_col, cfg.min_cells_per_barcode)

    if (wt_df.get_column(barcode_col) == _POOL_GROUP).any():
        raise ValueError(
            f"wtvvariantpool: a wildtype barcode collides with the reserved "
            f"pool sentinel value {_POOL_GROUP!r}; rename that barcode or the "
            f"sentinel."
        )

    pool_df = build_variant_pool(
        data_df, cfg.label_column, cfg.wt_label, list(cfg.variant_classes)
    )
    if len(pool_df) == 0:
        raise ValueError(
            f"wtvvariantpool: variant pool is empty after filtering to "
            f"variant_classes={list(cfg.variant_classes)!r}; no cells matched"
        )
    pool_df = pool_df.select(feature_cols).with_columns(
        pl.lit(_POOL_GROUP).alias(barcode_col)
    )

    merged = pl.concat([wt_df, pool_df], how="vertical")

    if cfg.downsample_variant_pool is not None and cfg.downsample_variant_pool is not False:
        n = (
            cfg.downsample_variant_pool
            if not isinstance(cfg.downsample_variant_pool, bool)
            else None
        )
        merged = downsample_group_to_target(
            merged, barcode_col, _POOL_GROUP, cfg.random_state, n=n
        )

    merged = merged.with_row_index("__idx__")
    groups = merged.get_column(barcode_col).to_numpy()

    train_idx, test_idx, val_idx = split_indices_stratified(groups, cfg.random_state)

    def select_rows(idx) -> pl.DataFrame:
        return merged.filter(pl.col("__idx__").is_in(idx)).select(select_cols)

    return select_rows(train_idx), select_rows(test_idx), select_rows(val_idx)


def train_xgboost(
    train: pl.DataFrame,
    val: pl.DataFrame,
    barcode_column: str,
    positive_barcode: str,
    cfg: DictConfig,
) -> xgb.Booster:
    """
    Train an XGBoost binary classifier to distinguish a wildtype barcode from the pool.

    Uses ``binary:logistic`` objective with AUC as the eval metric. Sample
    weights are computed with :func:`sklearn.utils.compute_sample_weight`
    when ``cfg.xgboost.weigh_samples`` is ``True``. Early stopping is applied
    against the validation set.

    Parameters
    ----------
    train : pl.DataFrame
        Training split containing feature columns and ``barcode_column``,
        restricted to the current barcode vs. the pool.
    val : pl.DataFrame
        Validation split used for early stopping and eval logging.
    barcode_column : str
        Name of the barcode label column.
    positive_barcode : str
        Barcode value treated as the positive class.
    cfg : DictConfig
        Hydra config supplying ``random_state`` and the ``xgboost`` sub-config.

    Returns
    -------
    xgb.Booster
        Trained XGBoost booster at the best iteration.
    """
    y_train = train.get_column(barcode_column).to_numpy() == positive_barcode
    sample_weight = (
        sklearn.utils.compute_sample_weight("balanced", y_train)
        if cfg.xgboost.weigh_samples
        else None
    )

    dtrain = get_dmatrix(train, barcode_column, positive_barcode, weight=sample_weight)
    deval = get_dmatrix(val, barcode_column, positive_barcode)

    params = dict(cfg.xgboost.params)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "auc"
    params["seed"] = cfg.random_state

    return xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.xgboost.num_boost_round,
        evals=[(dtrain, "train"), (deval, "eval")],
        early_stopping_rounds=cfg.xgboost.early_stopping_rounds,
        verbose_eval=True,
    )


def evaluate(
    df: pl.DataFrame, model: xgb.Booster, barcode_column: str, positive_barcode: str
) -> tuple[float, float]:
    """
    Compute AUROC and accuracy for a trained model on a DataFrame split.

    Parameters
    ----------
    df : pl.DataFrame
        Split to evaluate. Must contain ``barcode_column`` and the same feature
        columns used during training.
    model : xgb.Booster
        Trained XGBoost booster.
    barcode_column : str
        Name of the barcode label column.
    positive_barcode : str
        Barcode value treated as the positive class, passed to
        :func:`.xgbparams.get_dmatrix`.

    Returns
    -------
    tuple[float, float]
        ``(auroc, accuracy)`` where accuracy uses a 0.5 probability threshold.
    """
    dmatrix = get_dmatrix(df, barcode_column, positive_barcode)
    y_true = dmatrix.get_label()
    y_prob = model.predict(dmatrix)
    auroc = sklearn.metrics.roc_auc_score(y_true, y_prob)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_prob >= 0.5)

    return auroc, accuracy


def evaluate_barcode(
    model: xgb.Booster,
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    barcode_column: str,
    barcode: str,
) -> dict:
    """
    Evaluate a trained model on train, validation, and test splits.

    Parameters
    ----------
    model : xgb.Booster
        Trained XGBoost booster.
    train : pl.DataFrame
        Training split, restricted to ``barcode``/the pool.
    val : pl.DataFrame
        Validation split, restricted to ``barcode``/the pool.
    test : pl.DataFrame
        Held-out test split, restricted to ``barcode``/the pool.
    barcode_column : str
        Name of the barcode label column.
    barcode : str
        Wildtype barcode being profiled (treated as the positive class).

    Returns
    -------
    dict
        Dictionary with keys ``barcode``, ``train_auroc``, ``train_accuracy``,
        ``val_auroc``, ``val_accuracy``, ``test_auroc``, ``test_accuracy``,
        ``n_cells_barcode``, ``n_cells_pool``.
    """
    evaluate_wrapper = lambda df: evaluate(df, model, barcode_column, barcode)

    train_auroc, train_accuracy = evaluate_wrapper(train)
    val_auroc, val_accuracy = evaluate_wrapper(val)
    test_auroc, test_accuracy = evaluate_wrapper(test)

    all_rows = pl.concat([train, val, test])
    n_cells_barcode = int((all_rows.get_column(barcode_column) == barcode).sum())
    n_cells_pool = int((all_rows.get_column(barcode_column) == _POOL_GROUP).sum())

    return {
        "barcode": barcode,
        "train_auroc": train_auroc,
        "train_accuracy": train_accuracy,
        "val_auroc": val_auroc,
        "val_accuracy": val_accuracy,
        "test_auroc": test_auroc,
        "test_accuracy": test_accuracy,
        "n_cells_barcode": n_cells_barcode,
        "n_cells_pool": n_cells_pool,
    }


def profile_barcode(
    barcode: str,
    train_all: pl.DataFrame,
    test_all: pl.DataFrame,
    val_all: pl.DataFrame,
    cfg: DictConfig,
) -> tuple[dict, xgb.Booster]:
    """
    Train and evaluate an XGBoost model for one wildtype barcode vs. the pool.

    Subsets ``train_all``, ``test_all``, and ``val_all`` to rows belonging to
    ``barcode`` or the pool, trains a model via :func:`train_xgboost`, and
    evaluates it via :func:`evaluate_barcode`.

    Parameters
    ----------
    barcode : str
        Wildtype barcode to profile.
    train_all : pl.DataFrame
        Full training split (all barcodes plus the pool).
    test_all : pl.DataFrame
        Full test split (all barcodes plus the pool).
    val_all : pl.DataFrame
        Full validation split (all barcodes plus the pool).
    cfg : DictConfig
        Hydra config supplying ``barcode_column`` and XGBoost settings.

    Returns
    -------
    tuple[dict, xgb.Booster]
        ``(result_dict, model)`` where ``result_dict`` contains the evaluation
        metrics from :func:`evaluate_barcode`.
    """
    barcode_col = cfg.barcode_column
    keep = pl.col(barcode_col).is_in([barcode, _POOL_GROUP])
    train, test, val = (
        train_all.filter(keep),
        test_all.filter(keep),
        val_all.filter(keep),
    )
    logging.info(
        "Subset sizes for barcode '%s' vs. pool — train: %d, val: %d, test: %d",
        barcode,
        len(train),
        len(val),
        len(test),
    )
    model = train_xgboost(train, val, barcode_col, barcode, cfg)
    result = evaluate_barcode(model, train, val, test, barcode_col, barcode)
    logging.info(
        "Results for barcode '%s' vs. pool: train_auroc=%.4f, val_auroc=%.4f, test_auroc=%.4f",
        barcode,
        result["train_auroc"],
        result["val_auroc"],
        result["test_auroc"],
    )
    return result, model


_EMPTY_RESULTS_SCHEMA = {
    "barcode": pl.Utf8,
    "train_auroc": pl.Float64,
    "train_accuracy": pl.Float64,
    "val_auroc": pl.Float64,
    "val_accuracy": pl.Float64,
    "test_auroc": pl.Float64,
    "test_accuracy": pl.Float64,
    "n_cells_barcode": pl.Int64,
    "n_cells_pool": pl.Int64,
}


@hydra.main(version_base=None, config_path=None, config_name="wtvvariantpool_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: wildtype-barcode-vs-variant-pool XGBoost profiling.

    Steps
    -----
    1. Read the feature file at ``cfg.input_file``.
    2. Build the pooled non-wildtype set (cells whose classified variant
       class is in ``cfg.variant_classes``) and split wildtype-vs-pool cells
       into train/test/val via :func:`train_test_val_split`.
    3. For each surviving wildtype barcode, train and evaluate an XGBoost
       binary classifier vs. the pool via :func:`profile_barcode`. Barcodes
       that raise an exception are skipped with a warning.
    4. Write per-barcode evaluation metrics to ``results.parquet`` and all
       trained models (keyed by barcode) to ``models.pkl``.
    5. Write per-barcode gain-based feature importance to
       ``feature_importance.parquet``.

    Output files
    ------------
    - ``{output_dir}/results.parquet``
    - ``{output_dir}/models.pkl``
    - ``{output_dir}/feature_importance.parquet`` (one row per barcode, one
      column per feature that appeared in at least one barcode's splits, plus
      ``barcode``)

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.wtvvariantpool \\
            output_dir=./out \\
            input_file=data/features.parquet \\
            wt_label=WT
    """
    wtvvariantpool_cfg: WtvvariantpoolConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(wtvvariantpool_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wtvvariantpool_cfg.output_dir = output_dir
    setup_logging(wtvvariantpool_cfg, "wtvvariantpool")

    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logging.info("Loading input from %s", cfg.input_file)
    feature_df = load_batches(cfg.input_file)[0].collect()
    train_all, test_all, val_all = train_test_val_split(feature_df, cfg)

    barcodes = sorted(
        b
        for b in train_all.get_column(cfg.barcode_column).unique().to_list()
        if b != _POOL_GROUP
    )
    logging.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_all),
        len(val_all),
        len(test_all),
    )

    results = []
    models = {}

    if len(barcodes) < 1:
        logging.warning(
            "No wildtype barcodes meet min_cells_per_barcode=%d; nothing to profile",
            cfg.min_cells_per_barcode,
        )
    else:
        logging.info("Found %d wildtype barcode(s) to profile", len(barcodes))

        for barcode in barcodes:
            logging.info("Training model for barcode '%s' vs. variant pool", barcode)
            try:
                result, model = profile_barcode(
                    barcode, train_all, test_all, val_all, cfg
                )
            except Exception:
                logging.warning(
                    "Failed to profile barcode '%s', skipping:\n%s",
                    barcode,
                    traceback.format_exc(),
                )
                continue
            results.append(result)
            models[barcode] = model

    results_df = (
        pl.DataFrame(results) if results else pl.DataFrame(schema=_EMPTY_RESULTS_SCHEMA)
    )

    results_path = output_dir / "results.parquet"
    results_df.write_parquet(results_path)
    logging.info("Results written to %s", results_path)

    models_path = output_dir / "models.pkl"
    logging.info("Writing models to %s", models_path)
    with open(models_path, "wb") as f:
        pickle.dump(models, f)

    logging.info("Computing feature importance")
    feature_cols = [c for c in train_all.columns if c != cfg.barcode_column]
    importance_dicts = []
    for barcode, model in models.items():
        importance = resolve_feature_importance(model, feature_cols)
        importance["barcode"] = barcode
        importance_dicts.append(importance)
    importance_df = (
        pl.from_dicts(importance_dicts)
        if importance_dicts
        else pl.DataFrame({"barcode": []})
    )
    importance_path = output_dir / "feature_importance.parquet"
    importance_df.write_parquet(importance_path)
    logging.info("Feature importance written to %s", importance_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
