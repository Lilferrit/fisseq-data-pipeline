# test_validate_smoke.py
import pathlib

import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.pipeline import Config, configure, run

np.random.seed(42)


def make_toy_dataset(path: pathlib.Path):
    # 8 samples: balanced across batch/label, some controls
    df = pl.DataFrame(
        {
            "f1": np.random.rand(80),
            "f2": np.random.rand(80),
            "batch": ["B1", "B1", "B1", "B1", "B2", "B2", "B2", "B2"] * 10,
            "label": ["A", "A", "B", "B", "A", "A", "B", "B"] * 10,
            "is_ctrl": [True, False, True, False, True, False, True, False] * 10,
        }
    )
    df.write_parquet(path)


@pytest.mark.parametrize("eager_db_loading", [True, False])
def test_run(tmp_path: pathlib.Path, eager_db_loading: bool):
    # Create toy dataset
    data_path = tmp_path / "toy.parquet"
    make_toy_dataset(data_path)

    # Build config
    cfg = Config(
        {
            "feature_cols": ["f1", "f2"],
            "batch_col_name": "batch",
            "label_col_name": "label",
            "control_sample_query": "is_ctrl",
        }
    )

    run(data_path, cfg, output_dir=tmp_path, eager_db_loading=eager_db_loading)
    for fname in [
        "data-cleaned.parquet",
        "normalized.parquet",
        "normalizer.pkl",
    ]:
        assert (tmp_path / fname).is_file()


def test_configure(tmp_path: pathlib.Path):
    config_file = tmp_path / "config.yaml"
    configure(str(config_file))
    assert config_file.is_file()
