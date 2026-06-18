from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from omegaconf import OmegaConf

import fisseq_data_pipeline.ovwtcellscores as m
from fisseq_data_pipeline.ovwt import get_dmatrix
from fisseq_data_pipeline.ovwtcellscores import OvwtCellScoresConfig, load_input

_LABEL_COL = "meta_aa_changes"
_WT = "WT"
_VARIANTS = ["V1", "V2"]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_feature_df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 30  # 10 rows per label
    labels = [_WT] * 10 + ["V1"] * 10 + ["V2"] * 10
    return pl.DataFrame(
        {
            "Intensity_Mean": rng.random(n).tolist(),
            "Texture_Var": rng.random(n).tolist(),
            _LABEL_COL: labels,
            "meta_barcode": [f"bc_{i:03d}" for i in range(n)],
            "meta_batch": ["batch_a"] * n,
        }
    )


@pytest.fixture
def toy_models(toy_feature_df: pl.DataFrame) -> dict[str, xgb.Booster]:
    models = {}
    for variant in _VARIANTS:
        subset = toy_feature_df.filter(pl.col(_LABEL_COL).is_in([_WT, variant])).select(
            "Intensity_Mean", "Texture_Var", _LABEL_COL
        )
        dmatrix = get_dmatrix(subset, _LABEL_COL, _WT)
        booster = xgb.train(
            {"objective": "binary:logistic", "nthread": 1, "verbosity": 0},
            dmatrix,
            num_boost_round=1,
        )
        models[variant] = booster
    return models


@pytest.fixture
def models_pkl(tmp_path: Path, toy_models: dict) -> Path:
    path = tmp_path / "models.pkl"
    with open(path, "wb") as f:
        pickle.dump(toy_models, f)
    return path


def _make_cfg(tmp_path: Path, models_pkl: Path, **overrides) -> OmegaConf:
    kwargs = dict(
        output_dir=str(tmp_path),
        input_file=str(tmp_path / "input.parquet"),
        models_path=str(models_pkl),
        wt_label=_WT,
        batch_size=10_000,
        label_column=_LABEL_COL,
    )
    kwargs.update(overrides)
    return OmegaConf.structured(OvwtCellScoresConfig(**kwargs))


# ---------------------------------------------------------------------------
# get_cell_scores — output columns
# ---------------------------------------------------------------------------


def test_output_has_variant_score_columns(tmp_path, toy_feature_df, models_pkl):
    cfg = _make_cfg(tmp_path, models_pkl)
    result = m.get_cell_scores(toy_feature_df.lazy(), cfg)
    assert "V1" in result.columns
    assert "V2" in result.columns


def test_output_has_meta_columns(tmp_path, toy_feature_df, models_pkl):
    cfg = _make_cfg(tmp_path, models_pkl)
    result = m.get_cell_scores(toy_feature_df.lazy(), cfg)
    assert "meta_barcode" in result.columns
    assert "meta_batch" in result.columns
    assert _LABEL_COL in result.columns


def test_output_has_no_feature_columns(tmp_path, toy_feature_df, models_pkl):
    cfg = _make_cfg(tmp_path, models_pkl)
    result = m.get_cell_scores(toy_feature_df.lazy(), cfg)
    assert "Intensity_Mean" not in result.columns
    assert "Texture_Var" not in result.columns


# ---------------------------------------------------------------------------
# get_cell_scores — row count and score range
# ---------------------------------------------------------------------------


def test_output_row_count_matches_input(tmp_path, toy_feature_df, models_pkl):
    cfg = _make_cfg(tmp_path, models_pkl)
    result = m.get_cell_scores(toy_feature_df.lazy(), cfg)
    assert len(result) == len(toy_feature_df)


def test_scores_are_in_unit_interval(tmp_path, toy_feature_df, models_pkl):
    cfg = _make_cfg(tmp_path, models_pkl)
    result = m.get_cell_scores(toy_feature_df.lazy(), cfg)
    for variant in _VARIANTS:
        scores = result[variant].to_numpy()
        assert np.all(scores >= 0.0), f"{variant} scores below 0"
        assert np.all(scores <= 1.0), f"{variant} scores above 1"


# ---------------------------------------------------------------------------
# get_cell_scores — batching equivalence
# ---------------------------------------------------------------------------


def test_small_batch_matches_large_batch(tmp_path, toy_feature_df, models_pkl):
    cfg_large = _make_cfg(tmp_path, models_pkl, batch_size=10_000)
    cfg_small = _make_cfg(tmp_path, models_pkl, batch_size=3)

    result_large = m.get_cell_scores(toy_feature_df.lazy(), cfg_large).sort(
        "meta_barcode"
    )
    result_small = m.get_cell_scores(toy_feature_df.lazy(), cfg_small).sort(
        "meta_barcode"
    )

    for variant in _VARIANTS:
        np.testing.assert_allclose(
            result_large[variant].to_numpy(),
            result_small[variant].to_numpy(),
            rtol=1e-5,
            err_msg=f"Scores differ between batch sizes for variant '{variant}'",
        )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_creates_output_file(tmp_path, toy_feature_df, models_pkl):
    input_path = tmp_path / "input.parquet"
    toy_feature_df.write_parquet(input_path)
    cfg = _make_cfg(tmp_path, models_pkl, input_file=str(input_path))
    with patch("fisseq_data_pipeline.ovwtcellscores.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "cell_scores.parquet").exists()


def test_main_output_has_correct_columns(tmp_path, toy_feature_df, models_pkl):
    input_path = tmp_path / "input.parquet"
    toy_feature_df.write_parquet(input_path)
    cfg = _make_cfg(tmp_path, models_pkl, input_file=str(input_path))
    with patch("fisseq_data_pipeline.ovwtcellscores.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "cell_scores.parquet")
    expected_cols = {"V1", "V2", "meta_barcode", "meta_batch", _LABEL_COL}
    assert set(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# load_input — split index detection
# ---------------------------------------------------------------------------


def test_load_input_full_feature_file(tmp_path, toy_feature_df):
    path = tmp_path / "features.parquet"
    toy_feature_df.write_parquet(path)
    result = load_input(str(path)).collect()
    assert len(result) == len(toy_feature_df)
    assert _LABEL_COL in result.columns


def test_load_input_split_index_loads_correct_rows(tmp_path, toy_feature_df):
    origin = tmp_path / "features.parquet"
    toy_feature_df.write_parquet(origin)
    row_idxs = [0, 5, 10, 15, 20]
    index_df = pl.DataFrame(
        {
            "row_idx": row_idxs,
            "origin_file": [str(origin)] * len(row_idxs),
        }
    )
    index_path = tmp_path / "test_index.parquet"
    index_df.write_parquet(index_path)
    result = load_input(str(index_path)).collect()
    assert len(result) == len(row_idxs)
    assert _LABEL_COL in result.columns


def test_load_input_split_index_row_count_matches_index(tmp_path, toy_feature_df):
    origin = tmp_path / "features.parquet"
    toy_feature_df.write_parquet(origin)
    row_idxs = list(range(0, 30, 3))  # every third row → 10 rows
    index_df = pl.DataFrame(
        {
            "row_idx": row_idxs,
            "origin_file": [str(origin)] * len(row_idxs),
        }
    )
    index_path = tmp_path / "index.parquet"
    index_df.write_parquet(index_path)
    result = load_input(str(index_path)).collect()
    assert len(result) == len(row_idxs)


def test_main_with_split_index(tmp_path, toy_feature_df, models_pkl):
    origin = tmp_path / "features.parquet"
    toy_feature_df.write_parquet(origin)
    row_idxs = list(range(0, 30, 2))  # 15 rows
    index_df = pl.DataFrame(
        {
            "row_idx": row_idxs,
            "origin_file": [str(origin)] * len(row_idxs),
        }
    )
    index_path = tmp_path / "test_index.parquet"
    index_df.write_parquet(index_path)
    cfg = _make_cfg(tmp_path, models_pkl, input_file=str(index_path))
    with patch("fisseq_data_pipeline.ovwtcellscores.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "cell_scores.parquet")
    assert len(result) == len(row_idxs)
    assert "V1" in result.columns
    assert "V2" in result.columns
