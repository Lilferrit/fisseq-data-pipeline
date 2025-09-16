# tests/test_validate_smoke.py
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fisseq_data_pipeline.pipeline as mod_under_test


def _make_input_df():
    """
    8 samples, 2 features, and metadata:
      - batches: A/A/A/A, B/B/B/B
      - labels:  x/x, y/y per batch
      - controls: all True (simplifies control selection)
    Ensures at least 2 samples per (_batch, _label) stratum so
    stratified split won't error.
    """
    return pl.DataFrame(
        {
            "f1": [0.0, 0.5, 1.0, 1.5, 5.0, 5.5, 6.0, 6.5] * 10,
            "f2": [1.0, 1.5, 2.0, 2.5, 6.0, 6.5, 7.0, 7.5] * 10,
            "_batch": ["A", "A", "A", "A", "B", "B", "B", "B"] * 10,
            "_label": ["x", "x", "y", "y", "x", "x", "y", "y"] * 10,
            "_is_control": [True, False] * 40,
        }
    )


def _write_config(cfg_path: Path):
    """
    Minimal YAML that your Config class should accept.
    Fields used by the pipeline:
      - feature_cols
      - batch_col_name
      - label_col_name
      - control_sample_query  (boolean column name works as a WHERE expr)
    """
    cfg_path.write_text(
        "\n".join(
            [
                "feature_cols:",
                "  - f1",
                "  - f2",
                "batch_col_name: _batch",
                "label_col_name: _label",
                "control_sample_query: _is_control",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_validate_smoke(tmp_path: Path):
    # --- Arrange: write real input parquet and config ---
    input_df = _make_input_df()
    input_path = tmp_path / "input.parquet"
    input_df.write_parquet(input_path)

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)

    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Act: run the real validate function ---
    mod_under_test.validate(
        input_data_path=input_path,
        config=cfg_path,  # real PathLike config
        output_dir=outdir,
        test_size=0.25,  # with 8 rows: test=2, train=6
        write_train_results=False,  # keep the smoke test lightweight
    )

    # --- Assert: files were produced and are readable ---
    expected_files = [
        "unmodified.test.parquet",
        "normalized.test.parquet",
        "harmonized.test.parquet",
        "normalizer.test.pkl",
        "harmonizer.test.pkl",
    ]
    for name in expected_files:
        p = outdir / name
        assert p.exists(), f"Missing expected output: {p}"

    # Spot-check parquet contents
    unmod = pl.read_parquet(outdir / "unmodified.test.parquet")
    normd = pl.read_parquet(outdir / "normalized.test.parquet")
    harmd = pl.read_parquet(outdir / "harmonized.test.parquet")

    # Expect 2 rows in test split (25% of 8) and feature/metadata columns present
    for frame in (unmod, normd, harmd):
        assert frame.shape[0] == 20
        for col in ["_batch", "_label", "_is_control", "f1", "f2"]:
            assert col in frame.columns

    # Models should be pickle-loadable
    with open(outdir / "normalizer.test.pkl", "rb") as f:
        _ = pickle.load(f)
    with open(outdir / "harmonizer.test.pkl", "rb") as f:
        _ = pickle.load(f)
