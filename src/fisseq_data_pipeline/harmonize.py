from typing import Dict, Tuple, Optional, Any
from os import PathLike

import neuroHarmonize
import polars as pl

from .utils import Config, get_control_samples, get_feature_matrix, set_feature_matrix

Harmonizer = Dict[str, Any]


def fit_harmonizer(
    train_data_df: pl.DataFrame,
    config: Optional[PathLike | Config],
) -> Harmonizer:
    config = Config(config)
    control_df = get_control_samples(train_data_df, config)
    _, control_feature_matrix = get_feature_matrix(control_df, config)
    control_covar_df = control_df.select(pl.col(config.batch_col_name).alias("SITE"))
    model, _ = neuroHarmonize.harmonizationLearn(
        control_feature_matrix, control_covar_df.to_pandas()
    )

    return model


def harmonize(
    data_df: pl.DataFrame, config: Optional[PathLike | Config], harmonizer: Harmonizer
) -> pl.DataFrame:
    config = Config(config)
    feature_cols, data_feature_matrix = get_feature_matrix(data_df, config)
    data_covar_df = data_df.select(pl.col(config.batch_col_name).alias("SITE"))
    harmonized_data_matrix = neuroHarmonize.harmonizationApply(
        data_feature_matrix, data_covar_df.to_pandas, harmonizer
    )
    data_df = set_feature_matrix(data_df, feature_cols, harmonized_data_matrix)

    return data_df
