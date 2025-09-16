from os import PathLike
from typing import Optional

import numpy as np
import sklearn.preprocessing

from .utils import Config

Normalizer = sklearn.preprocessing.StandardScaler


def fit_normalizer(
    feature_matrix: np.ndarray,
    config: Optional[PathLike | Config],
    is_control: Optional[np.ndarray] = None,
) -> Normalizer:
    """
    Fit a normalization model (scikit-learn StandardScaler) on feature data.

    If a control mask is provided, the normalizer is fit only on the subset
    of control samples. Otherwise, all samples are used.

    Parameters
    ----------
    feature_matrix : np.ndarray
        2D array of shape (n_samples, n_features) containing the feature data.
    config : PathLike or Config, optional
        Configuration object or path. Currently unused for fitting but
        included for API consistency with other functions.
    is_control : np.ndarray, optional
        Boolean mask of shape (n_samples,) indicating which samples are
        controls. If provided, only control samples are used to fit the
        normalizer.

    Returns
    -------
    Normalizer
        A fitted ``sklearn.preprocessing.StandardScaler`` instance that stores
        the mean and variance used for centering and scaling.
    """
    config = Config(config)
    if is_control is not None:
        feature_matrix = feature_matrix[is_control]

    normalizer = sklearn.preprocessing.StandardScaler()
    normalizer.fit(feature_matrix)
    return normalizer


def normalize(feature_matrix: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    """
    Apply a fitted normalization model (StandardScaler) to feature data.

    Parameters
    ----------
    feature_matrix : np.ndarray
        2D array of shape (n_samples, n_features) containing the feature data
        to be normalized.
    normalizer : Normalizer
        A fitted ``sklearn.preprocessing.StandardScaler`` instance returned by
        ``fit_normalizer``.

    Returns
    -------
    np.ndarray
        Normalized feature matrix of shape (n_samples, n_features), where each
        feature has been centered and scaled according to the fitted normalizer.
    """
    return normalizer.transform(feature_matrix)
