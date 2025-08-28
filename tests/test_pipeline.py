import pathlib

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from fisseq_data_pipeline.pipeline import configure, run, validate


@pytest.fixture
def toy_parquet(tmp_path: pathlib.Path) -> pathlib.Path:
    rng = np.random.default_rng(0)
    n = 24
    df = pd.DataFrame(
        {
            "A": rng.normal(size=n),  # feature
            "B": rng.normal(size=n),  # feature
            "site": np.where(np.arange(n) % 2 == 0, "s1", "s2"),
            "label": [0, 1, 3] * (n // 3),
            "variant_class": ["A", "B", "WT"] * (n // 3),
        }
    )
    path = tmp_path / "input.parquet"
    pl.from_pandas(df).write_parquet(path)
    return path


@pytest.fixture
def toy_config_yaml(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a minimal config.yaml using yaml.safe_dump from a dict."""
    config_dict = {
        "cell_id_col_name": "cell_id",
        "cross_val_fold_id_col_name": "cv_fold_id",
        "feature_cols": "^[A-Z].*",  # regex for features
        "batch_col_name": "site",
        "label_col_name": "label",
        "control_sample_query": "variant_class = 'WT'",
    }
    p = tmp_path / "config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)
    return p


def test_run_smoke(
    tmp_path: pathlib.Path, toy_parquet: pathlib.Path, toy_config_yaml: pathlib.Path
):
    outdir = tmp_path / "run_out"
    outdir.mkdir(parents=True, exist_ok=True)

    run(input_data_path=toy_parquet, config=toy_config_yaml, output_dir=outdir)
    assert (outdir / "normalized.parquet").is_file()
    assert (outdir / "harmonized.parquet").is_file()
    assert (outdir / "normalizer.pkl").is_file()
    assert (outdir / "harmonizer.pkl").is_file()


@pytest.mark.parametrize("test_size", [0.3, 0.4])
def test_validate_smoke(
    tmp_path: pathlib.Path,
    toy_parquet: pathlib.Path,
    toy_config_yaml: pathlib.Path,
    test_size: float,
):
    outdir = tmp_path / "cv_out"
    outdir.mkdir(parents=True, exist_ok=True)

    validate(
        input_data_path=toy_parquet,
        config=toy_config_yaml,
        output_dir=outdir,
        test_size=test_size,
    )

    assert (outdir / f"unmodified.test.parquet").is_file()
    assert (outdir / f"normalized.test.parquet").is_file()
    assert (outdir / f"harmonized.test.parquet").is_file()
    assert (outdir / f"normalizer.test.pkl").is_file()
    assert (outdir / f"harmonizer.test.pkl").is_file()


def test_configure_writes_file(tmp_path: pathlib.Path):
    out_cfg = tmp_path / "written_config.yaml"
    try:
        configure(output_path=out_cfg)
    except FileNotFoundError:
        pytest.skip("DEFAULT_CFG_PATH not available in test environment.")
    else:
        assert out_cfg.is_file()
        # Optionally sanity-check it’s valid YAML
        text = out_cfg.read_text(encoding="utf-8")
        assert "feature_cols" in text
