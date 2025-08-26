import pathlib
import pickle
import shutil
from typing import Optional
from os import PathLike

import sklearn.model_selection
import polars as pl
import fire

from .utils import Config
from .utils.config import DEFAULT_CFG_PATH
from .normalize import fit_normalizer, normalize
from .harmonize import fit_harmonizer, harmonize


def validate(
    input_data_path: PathLike,
    config: Optional[Config | PathLike],
    output_dir: Optional[PathLike] = None,
    n_folds: int = 5,
) -> None:
    # Mixing pandas and polars is a necessary evil here :(
    data_df = pl.read_parquet(input_data_path).to_pandas()
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    config = Config(config)

    skf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=42
    )
    strata = (
        data_df[[config.batch_col_name, config.label_col_name]]
        .astype(str)
        .agg("_".join, axis=1)
    )

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_df, strata), 1):
        train_df = pl.from_pandas(data_df.iloc[train_idx])
        test_df = pl.from_pandas(data_df.iloc[test_idx])

        normalizer = fit_normalizer(train_df, config)
        normalized_data = normalize(test_df, config, normalizer)
        harmonizer = fit_harmonizer(normalized_data, config)
        harmonized_data = harmonize(normalized_data, config, harmonizer)

        normalized_data.write_parquet(output_dir / f"normalized.fold_{fold:05}.parquet")
        harmonized_data.write_parquet(output_dir / f"harmonized.fold_{fold:05}.parquet")
        with open(output_dir / f"normalizer.fold_{fold:05}.pkl", "wb") as f:
            pickle.dump(normalizer, f)
        with open(output_dir / f"harmonizer.fold_{fold:05}.pkl", "wb") as f:
            pickle.dump(harmonizer, f)


def run(
    input_data_path: PathLike,
    config: Optional[Config | PathLike],
    output_dir: Optional[PathLike] = None,
) -> None:
    data_df = pl.read_parquet(input_data_path)
    output_dir = pathlib.Path.cwd() if output_dir is None else pathlib.Path(output_dir)
    config = Config(config)

    normalizer = fit_normalizer(data_df, config)
    normalized_data = normalize(data_df, config, normalizer)
    harmonizer = fit_harmonizer(normalized_data, config)
    harmonized_data = harmonize(normalized_data, config, harmonizer)

    normalized_data.write_parquet(output_dir / f"normalized.parquet")
    harmonized_data.write_parquet(output_dir / f"harmonized.parquet")
    with open(output_dir / f"normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    with open(output_dir / f"harmonizer.pkl", "wb") as f:
        pickle.dump(harmonizer, f)


def configure(output_path: Optional[PathLike] = None) -> None:
    if output_path is None:
        output_path = pathlib.Path.cwd() / "config.yaml"

    shutil.copy(DEFAULT_CFG_PATH, output_path)


def main() -> None:
    fire.Fire({"validate": validate, "run": run, "configure": configure})


if __name__ == "__main__":
    main()
