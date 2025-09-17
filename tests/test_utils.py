# test_pipeline_utils.py
import numpy as np
import polars as pl
import polars.selectors as cs
import pytest

from fisseq_data_pipeline.utils.utils import (
    get_data_dfs,
    get_feature_columns,
    get_feature_selector,
    train_test_split,
)


class DummyConfig:
    def __init__(
        self,
        feature_cols,
        batch_col_name="_batch_src",
        label_col_name="_label_src",
        control_sample_query="_is_ctrl_src",
    ):
        self.feature_cols = feature_cols
        self.batch_col_name = batch_col_name
        self.label_col_name = label_col_name
        # Polars SQL expression string; should evaluate to a boolean per-row.
        self.control_sample_query = control_sample_query


def test_get_feature_selector_regex():
    lf = pl.LazyFrame(
        {
            "f_a": [1, 2],
            "f_b": [3, 4],
            "meta": [0, 1],
        }
    )
    cfg = DummyConfig(feature_cols=r"^f_")
    sel = get_feature_selector(lf, cfg)

    # Apply the selector to ensure it picks only f_a and f_b
    out = lf.select(sel).collect()
    assert out.columns == ["f_a", "f_b"]


def test_get_feature_selector_list_preserves_order_and_ignores_missing():
    lf = pl.LazyFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )
    cfg = DummyConfig(feature_cols=["a", "missing", "b"])
    sel = get_feature_selector(lf, cfg)

    out = lf.select(sel).collect()
    # Missing column gets ignored; order of present ones is preserved
    assert out.columns == ["a", "b"]


def test_get_feature_columns_from_lazyframe():
    lf = pl.LazyFrame(
        {
            "x1": [1, 2, 3],
            "x2": [4, 5, 6],
            "other": [0, 1, 0],
        }
    )
    cfg = DummyConfig(feature_cols=["x2", "x1"])
    out_lf = get_feature_columns(lf, cfg)
    out = out_lf.collect()
    # Only feature columns, in requested order
    assert out.columns == ["x2", "x1"]
    assert out.shape == (3, 2)


def test_get_data_dfs_basic():
    # Build a small LazyFrame with all required columns
    lf = pl.LazyFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "site": ["S1", "S1", "S2", "S2"],
            "y": ["A", "B", "A", "B"],
            "_is_ctrl_src": [True, False, True, False],  # used by control_sample_query
        }
    )
    cfg = DummyConfig(
        feature_cols=["a", "b"],
        batch_col_name="site",
        label_col_name="y",
        control_sample_query="_is_ctrl_src",
    )

    feature_df, meta_df = get_data_dfs(lf, cfg, dtype=pl.Float32)

    # Feature frame has only a,b with requested dtype
    assert feature_df.columns == ["a", "b"]
    assert feature_df.dtypes == [pl.Float32, pl.Float32]
    assert feature_df.shape == (4, 2)

    # Meta frame has expected columns
    assert meta_df.columns == ["_batch", "_label", "_is_control", "_sample_idx"]
    assert meta_df["_batch"].to_list() == ["S1", "S1", "S2", "S2"]
    assert meta_df["_label"].to_list() == ["A", "B", "A", "B"]
    assert meta_df["_is_control"].to_list() == [True, False, True, False]
    assert meta_df["_sample_idx"].to_list() == [0, 1, 2, 3]


def test_train_test_split_stratified_on_label_and_batch():
    # Create 8 samples with 4 groups: A:S1, A:S2, B:S1, B:S2 (2 samples each)
    feature_df = pl.DataFrame(
        {
            "f1": np.arange(8, dtype=float),
            "f2": np.arange(8, dtype=float) * 10.0,
        }
    )
    labels = ["A", "A", "A", "A", "B", "B", "B", "B"]
    batches = ["S1", "S1", "S2", "S2", "S1", "S1", "S2", "S2"]
    meta_df = pl.DataFrame(
        {
            "_label": labels,
            "_batch": batches,
        }
    )

    # With test_size=0.5 and 2 per group, we expect 1 per group in test
    trX, trM, teX, teM = train_test_split(feature_df, meta_df, test_size=0.5)

    assert trX.shape[1] == feature_df.shape[1]
    assert teX.shape[1] == feature_df.shape[1]
    assert trX.shape[0] + teX.shape[0] == feature_df.shape[0]
    assert trM.shape[0] + teM.shape[0] == meta_df.shape[0]

    # Check stratification: each (label,batch) appears once in test and once in train
    def counts(df):
        return (
            pl.DataFrame({"_label": df["_label"], "_batch": df["_batch"]})
            .group_by(["_label", "_batch"])
            .len()
            .sort(by=["_label", "_batch"])
            .to_dict(as_series=False)
        )

    train_counts = counts(trM)
    test_counts = counts(teM)

    # Expected keys
    expected_keys = {
        ("A", "S1"),
        ("A", "S2"),
        ("B", "S1"),
        ("B", "S2"),
    }
    got_keys_train = set(zip(train_counts["_label"], train_counts["_batch"]))
    got_keys_test = set(zip(test_counts["_label"], test_counts["_batch"]))
    assert got_keys_train == expected_keys
    assert got_keys_test == expected_keys

    # Each group should have exactly 1 sample in train and 1 in test
    assert all(n == 1 for n in train_counts["len"])
    assert all(n == 1 for n in test_counts["len"])
