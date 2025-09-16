import pathlib
import pickle
from collections import Counter

import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.pipeline import _clean, get_clean_masks


@pytest.fixture
def feature_and_meta():
    """
    Make a small dataset with:
      - col_all_nan: all-NaN column
      - col_zero_var: zero-variance column
      - col_ok: valid numeric with one NaN (to create a bad row)
    Row 2 has a NaN in a non-all-NaN column -> dropped by _clean.
    """
    feature_df = pl.DataFrame(
        {
            "col_all_nan": [np.nan, np.nan, np.nan, np.nan] * 10,
            "col_zero_var": [1.0, 1.0, 1.0, 1.0] * 10,
            "col_ok": [0.0, 1.0, np.nan, 3.0] * 10,
        }
    )
    # 4 rows; row 2 (0-based) will be dropped by _clean
    meta = pl.DataFrame(
        {
            "_batch": ["A", "A", "B", "B"] * 10,
            "_label": ["x", "y", "x", "y"] * 10,
            "_is_control": [True, False, True, False] * 10,
            "_other": [10, 20, 30, 40] * 10,
        }
    )
    return feature_df, meta


def test_clean_removes_all_nan_zero_var_and_bad_rows(feature_and_meta, monkeypatch):
    feature_df, meta_df = feature_and_meta
    kept_cols = ["col_ok"]

    # Sanity: ensure get_clean_masks works on the raw numpy
    fm = feature_df.to_numpy()
    all_nan, zero_var, row_nan = get_clean_masks(fm)
    assert all_nan.tolist() == [True, False, False]
    assert zero_var.tolist() == [True, True, False]
    assert row_nan.tolist() == [False, False, True, False] * 10

    # Exercise _clean
    cleaned_features, cleaned_meta = _clean(feature_df, meta_df)

    assert cleaned_features.columns == kept_cols
    assert cleaned_features.height == 30
    assert cleaned_meta.height == 30

    # Row alignment preserved
    np.testing.assert_array_equal(
        cleaned_meta["_other"].to_numpy().ravel(), np.array([10, 20, 40] * 10)
    )


def test_get_train_test_stratifies(feature_and_meta):
    from fisseq_data_pipeline.pipeline import _get_train_test

    # Drop the NaN row (index 2) to avoid imbalance
    feature_df, meta_df = feature_and_meta
    feature_df = feature_df.filter(pl.Series([True, True, False, True] * 10))
    meta_df = meta_df.filter(pl.Series([True, True, False, True] * 10))

    train_f, train_m, test_f, test_m = _get_train_test(
        feature_df, meta_df, test_size=1 / 3
    )

    assert train_f.height + test_f.height == feature_df.height
    assert train_m.height + test_m.height == meta_df.height
    assert train_f.width == feature_df.width
    assert test_f.width == feature_df.width


def test_write_output_writes_three_parquets(tmp_path, monkeypatch):
    from fisseq_data_pipeline.pipeline import _write_output  # adjust module name

    meta_df = pl.DataFrame(
        {
            "_batch": ["A", "B", "B"],
            "_label": ["x", "x", "y"],
            "_is_control": [True, False, True],
            "_id": [0, 1, 2],
        }
    )
    feature_cols = ["f1", "f2"]
    unmodified = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    normalized = unmodified / 10.0
    harmonized = unmodified + 100.0

    # Fake set_feature_matrix for test isolation
    def _fake_set_feature_matrix(meta_lf, cols, mat):
        df = meta_lf.collect()
        feat_df = pl.DataFrame(mat, columns=cols)
        return pl.concat((df, feat_df), how="horizontal").lazy()

    monkeypatch.setattr("fisseq_data_pipeline.pipeline", _fake_set_feature_matrix)

    _write_output(
        unmodified_matrix=unmodified,
        normalized_matrix=normalized,
        harmonized_matrix=harmonized,
        meta_data=meta_df.lazy(),
        feature_cols=feature_cols,
        output_dir=tmp_path,
        split_name="test",
    )

    paths = {
        "unmodified": tmp_path / "unmodified.test.parquet",
        "normalized": tmp_path / "normalized.test.parquet",
        "harmonized": tmp_path / "harmonized.test.parquet",
    }
    for p in paths.values():
        assert p.exists(), f"Expected file not found: {p}"

    unmod = pl.read_parquet(paths["unmodified"])
    assert all(
        c in unmod.columns
        for c in ["_batch", "_label", "_is_control", "_id", "f1", "f2"]
    )
    assert unmod.shape == (3, 6)
    assert unmod["f2"].to_list() == [10.0, 20.0, 30.0]
