# test_features_utils.py
from typing import Any, Dict

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest

import fisseq_data_pipeline.utils.utils as mod_under_test


class DummyConfig:
    """Minimal config stub with only the fields the functions use."""

    def __init__(
        self,
        feature_cols,
        batch_col_name: str = "batch",
        label_col_name: str = "label",
        control_sample_query: str = "qc",
    ):
        self.feature_cols = feature_cols
        self.batch_col_name = batch_col_name
        self.label_col_name = label_col_name
        # Interpreted by pl.sql_expr; here we use a simple boolean column name.
        self.control_sample_query = control_sample_query


# ----------------------- get_feature_selector --------------------------------


def test_get_feature_selector_with_regex():
    df = pl.DataFrame({"f1": [1], "f2": [2], "x": [3]}).lazy()
    cfg = DummyConfig(feature_cols=r"^f\d+$")  # regex for f1, f2

    sel = mod_under_test.get_feature_selector(df, cfg)
    # Apply selector to see the resulting columns
    out_cols = df.select(sel).collect().columns
    assert out_cols == ["f1", "f2"]


def test_get_feature_selector_with_list_preserves_order_and_warns_missing(caplog):
    df = pl.DataFrame({"b": [0], "a": [1], "c": [2]}).lazy()
    # Include a missing column 'z' and out-of-order selection
    cfg = DummyConfig(feature_cols=["a", "z", "b"])

    with caplog.at_level("WARNING"):
        sel = mod_under_test.get_feature_selector(df, cfg)
        # After selection, only existing columns in requested order
        out_cols = df.select(sel).collect().columns
        assert out_cols == ["a", "b"]

        # Warning mentions missing column
        assert any("ignored" in rec.message for rec in caplog.records)


# ----------------------- get_feature_columns ---------------------------------


def test_get_feature_columns_returns_only_selected_columns():
    df = pl.DataFrame({"f1": [1, 2], "f2": [3, 4], "meta": [9, 9]}).lazy()
    cfg = DummyConfig(feature_cols=["f2", "f1"])  # order should be respected

    out_lf = mod_under_test.get_feature_columns(df, cfg)
    out = out_lf.collect()
    assert out.columns == ["f2", "f1"]
    assert out.shape == (2, 2)


def test_get_feature_columns_with_regex():
    df = pl.DataFrame({"a1": [1], "a2": [2], "b": [3]}).lazy()
    cfg = DummyConfig(feature_cols=r"^a\d$")
    out = mod_under_test.get_feature_columns(df, cfg).collect()
    assert out.columns == ["a1", "a2"]


# ----------------------- get_data_dfs ---------------------------------------


def test_get_data_dfs_shapes_and_columns():
    # Build a small dataset with features, batch, label, and qc (for controls)
    data = pl.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0],
            "f2": [10.0, 11.0, 12.0, 13.0],
            "batch": ["A", "A", "B", "B"],
            "label": ["x", "y", "x", "y"],
            "qc": [True, False, True, True],  # control mask query = "qc"
        }
    ).lazy()

    cfg = DummyConfig(
        feature_cols=["f2", "f1"],  # order matters
        batch_col_name="batch",
        label_col_name="label",
        control_sample_query="qc",
    )

    feat_df, meta_df = mod_under_test.get_data_dfs(data, cfg, dtype=pl.Float32)

    # Feature DataFrame keeps only the specified features in order, cast to f32
    assert feat_df.columns == ["f2", "f1"]
    assert all(feat_df.schema[c] == pl.Float32 for c in feat_df.columns)
    assert feat_df.shape == (4, 2)

    # Meta DataFrame has required columns
    assert meta_df.columns == ["_batch", "_label", "_is_control", "_sample_idx"]
    assert meta_df.shape == (4, 4)
    # Check that _is_control equals the qc column values
    assert meta_df["_is_control"].to_list() == [True, False, True, True]
    # sample_idx is a simple range starting at 0
    assert meta_df["_sample_idx"].to_list() == [0, 1, 2, 3]


# ----------------------- set_feature_matrix ----------------------------------


def test_set_feature_matrix_replaces_feature_columns(monkeypatch):
    """
    set_feature_matrix uses pl.LazyFrame(new_features, schema=feature_cols)
    which may not be a public constructor. Patch it to a safe equivalent
    for the purpose of testing behavior.
    """
    # Metadata LazyFrame (3 rows)
    meta_lf = pl.DataFrame(
        {
            "_batch": ["A", "B", "B"],
            "_label": ["x", "x", "y"],
            "_is_control": [True, False, True],
            "_sample_idx": [0, 1, 2],
            "keep": [42, 43, 44],  # some extra non-feature column
        }
    ).lazy()

    feature_cols = ["f1", "f2"]
    new_feats = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float)

    out_lf = mod_under_test.set_feature_matrix(
        meta_data_df=meta_lf,
        feature_cols=feature_cols,
        new_features=new_feats,
    )
    out = out_lf.collect()

    # All metadata columns are preserved + new feature columns appended
    for c in ["_batch", "_label", "_is_control", "_sample_idx", "keep", "f1", "f2"]:
        assert c in out.columns

    # Feature values match the provided matrix
    assert out["f1"].to_list() == [1.0, 2.0, 3.0]
    assert out["f2"].to_list() == [10.0, 20.0, 30.0]

    # Row count preserved
    assert out.shape[0] == 3
