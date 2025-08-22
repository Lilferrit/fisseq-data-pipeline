import pathlib
from typing import Collection
from typing_extensions import TypeAlias, Dict, Optional

import yaml
import polars as pl
import sklearn.base
import sklearn.preprocessing

SklearnNormalizer: TypeAlias = (
    sklearn.base.TransformerMixin
    & sklearn.base.OneToOneFeatureMixin
    & sklearn.base.BaseEstimator
)

NORMALIZERS = {
    "standard-scaler": sklearn.preprocessing.StandardScaler,
}


def normalize(
    data_df: pl.DataFrame,
    normalizer: SklearnNormalizer | str,
    control_sample_query: str,
    feature_cols: Collection[str],
) -> pl.DataFrame:
    if isinstance(normalizer, str):
        try:
            normalizer_cls = NORMALIZERS[normalizer]
        except KeyError:
            raise ValueError(f"Unsupported normalizer string: {normalizer}")
        
        normalizer = normalizer_cls()

    control_df = data_df.sql(control_sample_query)
    control_features = control_df.select(feature_cols).to_numpy()
    normalizer.fit(control_features)

    data_features = data_df.select(feature_cols).to_numpy()
    data_features = normalizer.transform(data_features)
    data_df = data_df.with_columns(
        [pl.Series(name, data_features[:, i]) for i, name in enumerate(feature_cols)]
    )

    return data_df

def normalize_from_file(
    data_file: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    config: Dict | str | pathlib.Path,
    output_root: Optional[str],
) -> None:
    if not isinstance(config, Dict):
        with open(config) as f:
            config = yaml.load(f)

    output_dir = pathlib.Path(output_dir)
    data_df = pl.read_parquet(data_file)

    norm_df = normalize(
        data_df=data_df,
        normalizer=config["normalizer"],
        control_sample_query=config["control_sample_query"],
        feature_cols=config["feature_col_pattern"]
    )