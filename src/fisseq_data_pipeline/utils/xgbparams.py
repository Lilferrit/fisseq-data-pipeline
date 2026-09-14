"""Shared XGBoost configuration, DMatrix construction, training, and split helpers.

Defines :class:`XGBoostParams` / :class:`XGBoostConfig` (shared Hydra sub-config),
:func:`get_dmatrix`, :func:`split_indices_stratified`, and
:func:`train_binary_xgboost`, used by :mod:`.ovwt`.

:func:`train_binary_xgboost` lived in ``ovwt.py`` as ``train_xgboost`` until the
OvWT rewrite; it is generic over any binary variant-vs-wildtype split, so it
belongs here. It reads ``cfg.random_seed`` (the one shared
:class:`~fisseq_data_pipeline.config.app.AppConfig` seed), not a stage-local
``random_state``.

:func:`get_dmatrix_multiclass` and :func:`resolve_feature_importance` were
dropped along with their only callers -- ``batchvsbatch.py`` and the old OvWT's
``feature_importance.parquet`` output respectively.
"""

import dataclasses
from typing import Optional

import numpy as np
import polars as pl
import sklearn.model_selection
import sklearn.utils
import xgboost as xgb
from omegaconf import DictConfig


@dataclasses.dataclass
class XGBoostParams:
    """
    XGBoost booster hyperparameters passed directly to :func:`xgb.train`.

    Attributes
    ----------
    nthread : int
        Number of parallel threads. ``-1`` uses all available. Defaults to ``-1``.
    max_depth : int
        Maximum tree depth. Defaults to ``3``.
    colsample_bytree : float
        Fraction of features sampled per tree. Defaults to ``0.7``.
    colsample_bylevel : float
        Fraction of features sampled per level. Defaults to ``0.7``.
    colsample_bynode : float
        Fraction of features sampled per split node. Defaults to ``0.7``.
    subsample : float
        Fraction of training rows sampled per tree. Defaults to ``0.5``.
    """

    nthread: int = -1
    max_depth: int = 3
    colsample_bytree: float = 0.7
    colsample_bylevel: float = 0.7
    colsample_bynode: float = 0.7
    subsample: float = 0.5


@dataclasses.dataclass
class XGBoostConfig:
    """
    Training-loop configuration for XGBoost.

    Attributes
    ----------
    num_boost_round : int
        Maximum number of boosting rounds. Defaults to ``100``.
    early_stopping_rounds : int
        Stop training if the eval metric does not improve for this many rounds.
        Defaults to ``5``.
    weigh_samples : bool
        If ``True``, use :func:`sklearn.utils.compute_sample_weight` with the
        ``"balanced"`` strategy to up-weight the minority class. Defaults to
        ``True``.
    params : XGBoostParams
        Booster hyperparameters. Defaults to :class:`XGBoostParams`.
    """

    num_boost_round: int = 100
    early_stopping_rounds: int = 5
    weigh_samples: bool = True
    params: XGBoostParams = dataclasses.field(default_factory=XGBoostParams)


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """
    Return the feature column names from a DataFrame.

    Feature columns are identified as those whose name starts with an uppercase
    letter and contains an underscore, matching the CellProfiler naming
    convention.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.

    Returns
    -------
    list[str]
        List of feature column names.
    """
    return [
        col for col in df.columns if len(col) > 0 and col[0].isupper() and "_" in col
    ]


def get_dmatrix(
    df: pl.DataFrame,
    label_col: str,
    wt_label: str,
    weight: Optional[np.ndarray] = None,
) -> xgb.DMatrix:
    """
    Build an XGBoost DMatrix from a Polars DataFrame for binary classification.

    Feature columns are all columns except ``label_col``. Non-finite values
    are replaced with ``NaN`` so XGBoost treats them as missing. Labels are
    boolean (``True`` = wildtype).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing feature columns and ``label_col``.
    label_col : str
        Name of the label column.
    wt_label : str
        Wildtype label string; rows with this label get label ``True``.
    weight : np.ndarray or None
        Optional per-sample weights array. Defaults to ``None``.

    Returns
    -------
    xgb.DMatrix
        DMatrix with boolean labels and optional weights.
    """
    feature_cols = [col for col in df.columns if col != label_col]
    x = df.select(feature_cols).cast(pl.Float64).to_numpy().copy()
    x[~np.isfinite(x)] = np.nan
    y = df.get_column(label_col).to_numpy() == wt_label
    return xgb.DMatrix(x, label=y, weight=weight)


def split_indices_stratified(
    labels: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce an 80/10/10 stratified train/test/val index split.

    Parameters
    ----------
    labels : np.ndarray
        1-D array of group labels used for stratification. May be any
        hashable dtype (strings, integers, etc.).
    random_state : int
        Random seed passed to :func:`sklearn.model_selection.train_test_split`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(train_idx, test_idx, val_idx)`` — 0-based positions into ``labels``.
    """
    all_idx = np.arange(len(labels))
    train_idx, val_test_idx = sklearn.model_selection.train_test_split(
        all_idx,
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )
    test_idx, val_idx = sklearn.model_selection.train_test_split(
        val_test_idx,
        test_size=0.5,
        stratify=labels[val_test_idx],
        random_state=random_state,
    )
    return train_idx, test_idx, val_idx


def train_binary_xgboost(
    train: pl.DataFrame,
    val: pl.DataFrame,
    cfg: DictConfig,
) -> xgb.Booster:
    """
    Train an XGBoost binary classifier on a variant-vs-wildtype split.

    Uses the ``binary:logistic`` objective with AUC as the eval metric, so the
    trained booster predicts P(wildtype). Sample weights are computed with
    :func:`sklearn.utils.compute_sample_weight` when
    ``cfg.xgboost.weigh_samples`` is ``True``. Early stopping is applied against
    ``val``.

    Parameters
    ----------
    train : pl.DataFrame
        Training split containing feature columns and ``cfg.label_column``.
    val : pl.DataFrame
        Validation split used for early stopping and eval logging. In
        :func:`fisseq_data_pipeline.ovwt.ovwt_batchwise` this is the same slice
        that fits the probability calibrator -- deliberately, matching the
        reference implementation; it is not an independent calibration set.
    cfg : DictConfig
        Hydra config supplying ``label_column``, ``wt_label``, ``random_seed``,
        and the ``xgboost`` sub-config. Must be a ``DictConfig`` (e.g. via
        ``OmegaConf.structured(cfg)``), not a bare dataclass: ``dict(...)`` on
        ``cfg.xgboost.params`` below raises otherwise.

    Returns
    -------
    xgb.Booster
        Trained XGBoost booster at the best iteration.
    """
    label_col = cfg.label_column
    wt_label = cfg.wt_label

    y_train = train.get_column(label_col).to_numpy() == wt_label
    sample_weight = (
        sklearn.utils.compute_sample_weight("balanced", y_train)
        if cfg.xgboost.weigh_samples
        else None
    )

    dtrain = get_dmatrix(train, label_col, wt_label, weight=sample_weight)
    deval = get_dmatrix(val, label_col, wt_label)

    params = dict(cfg.xgboost.params)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "auc"
    params["seed"] = cfg.random_seed

    return xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.xgboost.num_boost_round,
        evals=[(dtrain, "train"), (deval, "eval")],
        early_stopping_rounds=cfg.xgboost.early_stopping_rounds,
        verbose_eval=True,
    )
