import dataclasses
from typing import Optional

import numpy as np
import polars as pl
import sklearn.model_selection
import xgboost as xgb


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


def get_dmatrix_multiclass(
    df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[xgb.DMatrix, list[str]]:
    """
    Build a multiclass XGBoost DMatrix from a Polars DataFrame.

    String labels are encoded as consecutive integers in sorted order.
    Non-finite feature values are replaced with ``NaN``.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing ``feature_cols`` and ``label_col``.
    feature_cols : list[str]
        Names of the feature columns to include.
    label_col : str
        Name of the label column (string labels).

    Returns
    -------
    tuple[xgb.DMatrix, list[str]]
        ``(dmatrix, classes)`` where ``classes[i]`` is the label string for
        integer class ``i``.
    """
    x = df.select(feature_cols).cast(pl.Float64).to_numpy().copy()
    x[~np.isfinite(x)] = np.nan
    raw_labels = df.get_column(label_col).to_numpy()
    classes = sorted(set(raw_labels))
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_int[v] for v in raw_labels], dtype=np.int32)
    return xgb.DMatrix(x, label=y), classes


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
