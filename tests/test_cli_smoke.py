# test_validate_smoke.py
import pathlib

import polars as pl
import pytest

from fisseq_data_pipeline.pipeline import Config, configure, validate


def make_toy_dataset(path: pathlib.Path):
    # 8 samples: balanced across batch/label, some controls
    df = pl.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 10,
            "f2": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0] * 10,
            "batch": ["B1", "B1", "B1", "B1", "B2", "B2", "B2", "B2"] * 10,
            "label": ["A", "A", "B", "B", "A", "A", "B", "B"] * 10,
            "is_ctrl": [True, False, True, False, True, False, True, False] * 10,
        }
    )
    df.write_parquet(path)


@pytest.mark.parametrize("write_train_results", [True, False])
def test_validate_smoke(tmp_path: pathlib.Path, write_train_results: bool):
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

    # Run validate
    output_dir = tmp_path / f"out_{write_train_results}"
    output_dir.mkdir()
    validate(
        input_data_path=data_path,
        config=cfg,
        output_dir=output_dir,
        test_size=0.25,
        write_train_results=write_train_results,
    )

    # Always expect test outputs
    expected_test_files = [
        "meta_data.test.parquet",
        "features.test.parquet",
        "normalized.test.parquet",
        "harmonized.test.parquet",
        "normalizer.pkl",
        "harmonizer.pkl",
    ]
    for fname in expected_test_files:
        assert (output_dir / fname).exists(), f"Missing expected output: {fname}"

    # Train outputs only if requested
    expected_train_files = [
        "meta_data.train.parquet",
        "features.train.parquet",
        "normalized.train.parquet",
        "harmonized.train.parquet",
    ]
    for fname in expected_train_files:
        path = output_dir / fname
        if write_train_results:
            assert path.exists(), f"Missing expected train output: {fname}"
        else:
            assert not path.exists(), f"Unexpected train output: {fname}"


def test_configure(tmp_path: pathlib.Path):
    config_file = tmp_path / "config.yaml"
    configure(str(config_file))
    assert config_file.is_file()
