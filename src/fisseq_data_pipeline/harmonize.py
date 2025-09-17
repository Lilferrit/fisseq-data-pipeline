from typing import Any, Dict

import neuroHarmonize
import pandas as pd
import polars as pl

Harmonizer = Dict[str, Any]


def fit_harmonizer(
    feature_df: pl.DataFrame,
    meta_data_df: pl.DataFrame,
    fit_only_on_control: bool = False,
) -> Harmonizer:
    """
    Fit a ComBat-based harmonization model using `neuroHarmonize`.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature matrix with shape (n_samples, n_features), numeric only.
    meta_data_df : pl.DataFrame
        Metadata aligned with rows in `feature_df`. Must contain a `_batch`
        column indicating batch membership. If `fit_only_on_control=True`,
        must also contain a boolean `_is_control` column.
    fit_only_on_control : bool, default=False
        If True, compute the harmonization model only from control samples
        (rows where `_is_control` is True).

    Returns
    -------
    Harmonizer
        A fitted harmonization model dictionary returned by
        ``neuroHarmonize.harmonizationLearn``.
    """
    if fit_only_on_control:
        feature_df = feature_df.filter(meta_data_df.get_column("_is_control"))
        meta_data_df = meta_data_df.filter(meta_data_df.get_column("_is_control"))

    covar_df = meta_data_df.select(pl.col("_batch").alias("SITE")).to_pandas()
    model, _ = neuroHarmonize.harmonizationLearn(feature_df.to_numpy(), covar_df)

    return model


def harmonize(
    feature_df: pl.DataFrame,
    meta_data_df: pl.DataFrame,
    harmonizer: Harmonizer,
) -> pl.DataFrame:
    """
    Apply a fitted harmonization model to adjust features for batch effects.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature matrix to harmonize; shape (n_samples, n_features).
    meta_data_df : pl.DataFrame
        Metadata aligned with rows in `feature_df`. Must contain a `_batch`
        column indicating batch membership.
    harmonizer : Harmonizer
        A fitted model dictionary produced by ``fit_harmonizer``.

    Returns
    -------
    pl.DataFrame
        Harmonized feature matrix with the same shape and column names
        as the input `feature_df`.
    """
    covar_df = meta_data_df.select(pl.col("_batch").alias("SITE")).to_pandas()
    harmonized_matrix = neuroHarmonize.harmonizationApply(
        feature_df.to_numpy(), covar_df, harmonizer
    )
    feature_df = pl.DataFrame(harmonized_matrix, schema=feature_df.columns)

    return feature_df
