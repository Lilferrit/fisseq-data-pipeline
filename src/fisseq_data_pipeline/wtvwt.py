"""Wildtype-vs-wildtype pairwise barcode XGBoost classification.

Hydra entry point (``python -m fisseq_data_pipeline.wtvwt``), backing the Nextflow process
``WTVWT_BATCHWISE``. Restricts to wildtype-labeled cells only, then trains one
binary XGBoost classifier per pair of wildtype barcodes to distinguish cells
belonging to either barcode, writing per-pair AUROC/accuracy results plus the
trained models.
"""

import dataclasses
import itertools
import logging
import pathlib
import pickle
import traceback
from typing import Optional

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
from .utils.log import setup_logging
from .utils.xgbparams import (
    XGBoostConfig,
    get_dmatrix,
    get_feature_cols,
    split_indices_stratified,
)


@dataclasses.dataclass
class WtvwtConfig(LabeledInputConfig):
    """
    Hydra structured configuration for the wildtype-vs-wildtype entry point.

    Extends :class:`.config.LabeledInputConfig` with parameters controlling
    per-barcode-pair XGBoost training restricted to wildtype cells.

    Attributes
    ----------
    wt_label : str
        Label string identifying wildtype cells; only rows with this label are
        used. Defaults to ``"WT"``.
    barcode_column : str
        Name of the column in ``input_file`` identifying each cell's barcode.
        Defaults to :data:`.utils.constants.META_BARCODE_COL` (``"meta_barcode"``).
    random_state : int
        Random seed for train/test/val splitting. Defaults to ``42``.
    feature_cols : list or None
        Explicit list of feature column names. If ``None``, columns are
        auto-detected by :func:`.xgbparams.get_feature_cols`. Defaults to
        ``None``.
    min_cells_per_barcode : int
        Minimum number of wildtype cells required for a barcode to be
        included in pairing. Barcodes with fewer cells are dropped before
        splitting. Defaults to ``100``.
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
    feature_block_list_file: Optional[str] = None
    xgboost: XGBoostConfig = dataclasses.field(default_factory=XGBoostConfig)


_cs = ConfigStore.instance()
_cs.store(name="wtvwt_main", node=WtvwtConfig)


def _exclude_blocked_features(
    feature_cols: list[str], feature_block_list_file: Optional[str]
) -> list[str]:
    """
    Drop blocked feature names from a feature column list.

    Parameters
    ----------
    feature_cols : list[str]
        Candidate feature column names.
    feature_block_list_file : str or None
        Path to a parquet file with ``feature`` (str) and ``feature_ok``
        (bool) columns, or ``None`` to skip filtering entirely.

    Returns
    -------
    list[str]
        ``feature_cols`` with any feature whose ``feature_ok`` is ``False``
        removed. Unchanged if ``feature_block_list_file`` is ``None``.
    """
    if feature_block_list_file is None:
        return feature_cols
    bl_df = pl.read_parquet(feature_block_list_file)
    blocked = set(bl_df.filter(~pl.col("feature_ok"))["feature"].to_list())
    return [f for f in feature_cols if f not in blocked]


def filter_min_cells_per_barcode(
    data_df: pl.DataFrame,
    barcode_column: str,
    min_cells: int,
) -> pl.DataFrame:
    """
    Remove barcode groups with fewer than ``min_cells`` cells.

    Parameters
    ----------
    data_df : pl.DataFrame
        DataFrame containing wildtype rows, including ``barcode_column``.
    barcode_column : str
        Name of the column identifying each cell's barcode.
    min_cells : int
        Minimum number of cells a barcode must have to be retained.

    Returns
    -------
    pl.DataFrame
        DataFrame with small barcode groups removed.
    """
    barcode_counts = data_df.group_by(barcode_column).len()
    keep_barcodes = (
        barcode_counts.filter(pl.col("len") >= min_cells)
        .get_column(barcode_column)
        .to_list()
    )
    return data_df.filter(pl.col(barcode_column).is_in(keep_barcodes))


def train_test_val_split(
    data_df: pl.DataFrame,
    cfg: DictConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split wildtype cells into train, test, and validation sets.

    Restricts ``data_df`` to wildtype rows (``label_column == wt_label``),
    drops barcodes with fewer than ``min_cells_per_barcode`` cells, and
    produces an 80/10/10 split stratified by ``barcode_column`` so every
    surviving barcode's rows span all three splits.

    Parameters
    ----------
    data_df : pl.DataFrame
        Full feature DataFrame containing feature columns, ``cfg.label_column``,
        and ``cfg.barcode_column``.
    cfg : DictConfig
        Hydra config supplying ``label_column``, ``wt_label``, ``feature_cols``,
        ``feature_block_list_file``, ``barcode_column``, ``min_cells_per_barcode``,
        and ``random_state``.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        ``(train, test, val)`` DataFrames, each containing feature columns and
        ``barcode_column``.
    """
    barcode_col = cfg.barcode_column
    if cfg.feature_cols is not None:
        feature_cols = list(cfg.feature_cols)
    else:
        feature_cols = get_feature_cols(data_df)
    feature_cols = _exclude_blocked_features(feature_cols, cfg.feature_block_list_file)

    data_df = data_df.filter(pl.col(cfg.label_column) == cfg.wt_label)
    select_cols = feature_cols + [barcode_col]
    data_df = data_df.select(select_cols)
    data_df = data_df.filter(pl.col(barcode_col).is_not_null())
    data_df = filter_min_cells_per_barcode(
        data_df, barcode_col, cfg.min_cells_per_barcode
    )

    data_df = data_df.with_row_index("__idx__")
    barcodes = data_df.get_column(barcode_col).to_numpy()

    train_idx, test_idx, val_idx = split_indices_stratified(barcodes, cfg.random_state)

    def select_rows(idx) -> pl.DataFrame:
        return data_df.filter(pl.col("__idx__").is_in(idx)).select(select_cols)

    return select_rows(train_idx), select_rows(test_idx), select_rows(val_idx)


def train_xgboost(
    train: pl.DataFrame,
    val: pl.DataFrame,
    barcode_column: str,
    positive_barcode: str,
    cfg: DictConfig,
) -> xgb.Booster:
    """
    Train an XGBoost binary classifier to distinguish two wildtype barcodes.

    Uses ``binary:logistic`` objective with AUC as the eval metric. Sample
    weights are computed with :func:`sklearn.utils.compute_sample_weight`
    when ``cfg.xgboost.weigh_samples`` is ``True``. Early stopping is applied
    against the validation set.

    Parameters
    ----------
    train : pl.DataFrame
        Training split containing feature columns and ``barcode_column``,
        restricted to the current barcode pair.
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


def evaluate_pair(
    model: xgb.Booster,
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    barcode_column: str,
    barcode_a: str,
    barcode_b: str,
) -> dict:
    """
    Evaluate a trained model on train, validation, and test splits.

    Parameters
    ----------
    model : xgb.Booster
        Trained XGBoost booster.
    train : pl.DataFrame
        Training split, restricted to ``barcode_a``/``barcode_b``.
    val : pl.DataFrame
        Validation split, restricted to ``barcode_a``/``barcode_b``.
    test : pl.DataFrame
        Held-out test split, restricted to ``barcode_a``/``barcode_b``.
    barcode_column : str
        Name of the barcode label column.
    barcode_a : str
        First barcode of the pair (treated as the positive class).
    barcode_b : str
        Second barcode of the pair.

    Returns
    -------
    dict
        Dictionary with keys ``barcode_a``, ``barcode_b``, ``train_auroc``,
        ``train_accuracy``, ``val_auroc``, ``val_accuracy``, ``test_auroc``,
        ``test_accuracy``, ``n_cells_a``, ``n_cells_b``.
    """
    evaluate_wrapper = lambda df: evaluate(df, model, barcode_column, barcode_a)

    train_auroc, train_accuracy = evaluate_wrapper(train)
    val_auroc, val_accuracy = evaluate_wrapper(val)
    test_auroc, test_accuracy = evaluate_wrapper(test)

    all_rows = pl.concat([train, val, test])
    n_cells_a = int((all_rows.get_column(barcode_column) == barcode_a).sum())
    n_cells_b = int((all_rows.get_column(barcode_column) == barcode_b).sum())

    return {
        "barcode_a": barcode_a,
        "barcode_b": barcode_b,
        "train_auroc": train_auroc,
        "train_accuracy": train_accuracy,
        "val_auroc": val_auroc,
        "val_accuracy": val_accuracy,
        "test_auroc": test_auroc,
        "test_accuracy": test_accuracy,
        "n_cells_a": n_cells_a,
        "n_cells_b": n_cells_b,
    }


def profile_pair(
    a: str,
    b: str,
    train_all: pl.DataFrame,
    test_all: pl.DataFrame,
    val_all: pl.DataFrame,
    cfg: DictConfig,
) -> tuple[dict, xgb.Booster]:
    """
    Train and evaluate an XGBoost model for one pair of wildtype barcodes.

    Subsets ``train_all``, ``test_all``, and ``val_all`` to rows belonging to
    barcode ``a`` or ``b``, trains a model via :func:`train_xgboost`, and
    evaluates it via :func:`evaluate_pair`.

    Parameters
    ----------
    a : str
        First barcode of the pair.
    b : str
        Second barcode of the pair.
    train_all : pl.DataFrame
        Full training split (all barcodes).
    test_all : pl.DataFrame
        Full test split (all barcodes).
    val_all : pl.DataFrame
        Full validation split (all barcodes).
    cfg : DictConfig
        Hydra config supplying ``barcode_column`` and XGBoost settings.

    Returns
    -------
    tuple[dict, xgb.Booster]
        ``(result_dict, model)`` where ``result_dict`` contains the evaluation
        metrics from :func:`evaluate_pair`.
    """
    barcode_col = cfg.barcode_column
    keep = pl.col(barcode_col).is_in([a, b])
    train, test, val = (
        train_all.filter(keep),
        test_all.filter(keep),
        val_all.filter(keep),
    )
    logging.info(
        "Subset sizes for ('%s', '%s') — train: %d, val: %d, test: %d",
        a,
        b,
        len(train),
        len(val),
        len(test),
    )
    model = train_xgboost(train, val, barcode_col, a, cfg)
    result = evaluate_pair(model, train, val, test, barcode_col, a, b)
    logging.info(
        "Results for ('%s', '%s'): train_auroc=%.4f, val_auroc=%.4f, test_auroc=%.4f",
        a,
        b,
        result["train_auroc"],
        result["val_auroc"],
        result["test_auroc"],
    )
    return result, model


_EMPTY_RESULTS_SCHEMA = {
    "barcode_a": pl.Utf8,
    "barcode_b": pl.Utf8,
    "train_auroc": pl.Float64,
    "train_accuracy": pl.Float64,
    "val_auroc": pl.Float64,
    "val_accuracy": pl.Float64,
    "test_auroc": pl.Float64,
    "test_accuracy": pl.Float64,
    "n_cells_a": pl.Int64,
    "n_cells_b": pl.Int64,
}


@hydra.main(version_base=None, config_path=None, config_name="wtvwt_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: wildtype-vs-wildtype pairwise barcode XGBoost profiling.

    Steps
    -----
    1. Read the feature file at ``cfg.input_file``.
    2. Restrict to wildtype cells and split into train/test/val via
       :func:`train_test_val_split`.
    3. For each pair of wildtype barcodes, train and evaluate an XGBoost
       binary classifier via :func:`profile_pair`. Pairs that raise an
       exception are skipped with a warning.
    4. Write per-pair evaluation metrics to ``results.parquet`` and all
       trained models (keyed by ``(barcode_a, barcode_b)``) to ``models.pkl``.

    Output files
    ------------
    - ``{output_dir}/results.parquet``
    - ``{output_dir}/models.pkl``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.wtvwt \\
            output_dir=./out \\
            input_file=data/features.parquet \\
            wt_label=WT
    """
    wtvwt_cfg: WtvwtConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(wtvwt_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wtvwt_cfg.output_dir = output_dir
    setup_logging(wtvwt_cfg, "wtvwt")

    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logging.info("Loading input from %s", cfg.input_file)
    feature_df = load_batches(cfg.input_file)[0].collect()
    train_all, test_all, val_all = train_test_val_split(feature_df, cfg)

    barcodes = sorted(train_all.get_column(cfg.barcode_column).unique().to_list())
    logging.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_all),
        len(val_all),
        len(test_all),
    )

    results = []
    models = {}

    if len(barcodes) < 2:
        logging.warning(
            "Fewer than 2 wildtype barcodes meet min_cells_per_barcode=%d "
            "(found %d); nothing to profile",
            cfg.min_cells_per_barcode,
            len(barcodes),
        )
    else:
        pairs = list(itertools.combinations(barcodes, 2))
        logging.info("Found %d barcode pair(s) to profile", len(pairs))

        for a, b in pairs:
            logging.info("Training model for barcode '%s' vs. '%s'", a, b)
            try:
                result, model = profile_pair(a, b, train_all, test_all, val_all, cfg)
            except Exception:
                logging.warning(
                    "Failed to profile pair ('%s', '%s'), skipping:\n%s",
                    a,
                    b,
                    traceback.format_exc(),
                )
                continue
            results.append(result)
            models[(a, b)] = model

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

    logging.info("Done")


if __name__ == "__main__":
    main()
