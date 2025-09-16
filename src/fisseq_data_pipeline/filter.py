from typing import Tuple

import numpy as np


def get_clean_masks(
    feature_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate boolean masks identifying problematic columns and rows.

    This function checks for three types of issues:
      1. Columns that consist entirely of NaN values.
      2. Columns with zero variance.
      3. Rows that contain NaNs in columns that are not entirely NaN.

    Parameters
    ----------
    feature_matrix : np.ndarray
        2D array of shape (n_samples, n_features) containing feature data.

    Returns
    -------
    all_nan : np.ndarray
        Boolean mask of shape (n_features,), where True indicates a column
        with all NaN values.
    zero_variance : np.ndarray
        Boolean mask of shape (n_features,), where True indicates a column
        with zero variance (within machine epsilon).
    contains_nan : np.ndarray
        Boolean mask of shape (n_samples,), where True indicates a row that
        contains one or more NaN values in non-all-NaN columns.
    """
    # Get columns that are all NaN
    all_nan = np.all(np.isnan(feature_matrix), axis=0)

    # Get mask tha contains any rows that have NaNs not in a all NaN row
    contains_nan = np.any(np.isnan(feature_matrix[:, ~all_nan]), axis=1)

    # Get mask containing columns that have 0 variance
    variance = np.var(feature_matrix[~contains_nan], axis=0)
    variance[all_nan] = 0.0
    zero_variance = variance <= np.finfo(feature_matrix.dtype).eps

    return all_nan, zero_variance, contains_nan
