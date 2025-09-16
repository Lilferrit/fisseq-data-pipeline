from os import PathLike
from typing import Any, Dict, Optional

import neuroHarmonize
import numpy as np
import pandas as pd

from .utils import Config

Harmonizer = Dict[str, Any]


def fit_harmonizer(
    feature_matrix: np.ndarray,
    batch_idx: np.ndarray,
    config: Optional[PathLike | Config],
    is_control: Optional[np.ndarray] = None,
) -> Harmonizer:
    """
    Fit a harmonization model to remove batch effects.

    The function learns a batch-effect correction model using the provided
    feature matrix and batch assignments. If a control mask is given, only the
    subset of control samples is used to fit the harmonizer.

    Parameters
    ----------
    feature_matrix : np.ndarray
        2D array of shape (n_samples, n_features) containing the feature data.
    batch_idx : np.ndarray
        1D array of length n_samples with integer or categorical batch labels.
    config : PathLike or Config, optional
        Configuration object or path to a config file. Must contain the
        attribute ``batch_col_name`` specifying the batch column.
    is_control : np.ndarray, optional
        Boolean mask of shape (n_samples,) indicating which samples are
        controls. If provided, only control samples are used to fit the model.

    Returns
    -------
    Harmonizer
        A harmonization model learned by
        ``neuroHarmonize.harmonizationLearn`` that can later be applied
        to correct batch effects in new data.
    """
    config = Config(config)
    if is_control is not None:
        feature_matrix = feature_matrix[is_control]
        batch_idx = batch_idx[is_control]

    covar_df = pd.DataFrame(batch_idx, columns=["SITE"])
    model, _ = neuroHarmonize.harmonizationLearn(feature_matrix, covar_df)
    return model


def harmonize(
    feature_matrix: np.ndarray,
    batch_idx: np.ndarray,
    config: Optional[PathLike | Config],
    harmonizer: Harmonizer,
) -> np.ndarray:
    """
    Apply a fitted harmonization model to feature data.

    This function uses a pre-trained harmonizer to correct for batch effects
    in a new feature matrix, given the associated batch assignments.

    Parameters
    ----------
    feature_matrix : np.ndarray
        2D array of shape (n_samples, n_features) containing the feature data
        to be harmonized.
    batch_idx : np.ndarray
        1D array of length n_samples with integer or categorical batch labels.
    config : PathLike or Config, optional
        Configuration object or path to a config file. Must contain the
        attribute ``batch_col_name`` specifying the batch column.
    harmonizer : Harmonizer
        A harmonization model previously learned by
        ``fit_harmonizer`` (wrapper around
        ``neuroHarmonize.harmonizationLearn``).

    Returns
    -------
    np.ndarray
        Harmonized feature matrix of shape (n_samples, n_features),
        corrected for batch effects.
    """
    config = Config(config)
    covar_df = pd.DataFrame(batch_idx, columns=["SITE"])
    harmonized_matrix = neuroHarmonize.harmonizationApply(
        feature_matrix, covar_df, harmonizer
    )

    return harmonized_matrix
