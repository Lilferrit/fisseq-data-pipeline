from os import PathLike
from typing import Any, Dict, Optional

import neuroHarmonize
import polars as pl

from .utils import Config, get_control_samples, get_feature_matrix, set_feature_matrix

Harmonizer = Dict[str, Any]


def fit_harmonizer(
    train_data_df: pl.LazyFrame,
    config: Optional[PathLike | Config],
) -> Harmonizer:
    """
    Fit a harmonization model using control samples from the training data.

    Parameters
    ----------
    train_data_df : pl.LazyFrame
        The training data as a Polars LazyFrame.
    config : PathLike or Config, optional
        Configuration object or path. Must contain the attribute
        ``batch_col_name`` specifying the batch column.

    Returns
    -------
    Harmonizer : dict
        A harmonization model dictionary learned by
        ``neuroHarmonize.harmonizationLearn``.
    """
    config = Config(config)
    control_df = get_control_samples(train_data_df, config)
    _, control_feature_matrix = get_feature_matrix(control_df, config)
    control_covar_df = control_df.select(pl.col(config.batch_col_name).alias("SITE"))
    model, _ = neuroHarmonize.harmonizationLearn(
        control_feature_matrix, control_covar_df.collect().to_pandas()
    )

    return model


def harmonize(
    data_df: pl.LazyFrame, config: Optional[PathLike | Config], harmonizer: Harmonizer
) -> pl.LazyFrame:
    """
    Apply a previously fitted harmonization model to new data.

    Parameters
    ----------
    data_df : pl.LazyFrame
        The input data as a Polars LazyFrame.
    config : PathLike or Config, optional
        Configuration object or path. Must contain the attribute
        ``batch_col_name`` specifying the batch column.
    harmonizer : dict
        A harmonization model produced by ``fit_harmonizer``.

    Returns
    -------
    pl.LazyFrame
        A LazyFrame containing the original data with feature columns replaced
        by the harmonized feature values.
    """
    config = Config(config)
    feature_cols, data_feature_matrix = get_feature_matrix(data_df, config)
    data_covar_df = data_df.select(pl.col(config.batch_col_name).alias("SITE"))
    harmonized_data_matrix = neuroHarmonize.harmonizationApply(
        data_feature_matrix, data_covar_df.collect().to_pandas(), harmonizer
    )
    data_df = set_feature_matrix(data_df, feature_cols, harmonized_data_matrix)

    return data_df
