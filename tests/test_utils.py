import logging

import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.utils import (
    Config,
    get_control_samples,
    get_feature_matrix,
    get_rows_by_idx,
    set_feature_matrix,
    get_feature_selector,
    get_feature_columns,
)


@pytest.fixture()
def small_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "batch": ["A", "A", "B", "B", "A"],
            "label": [1, 0, 1, 0, 1],
            "f_a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "f_b": [10, 20, 30, 40, 50],
            "other": ["x", "y", "z", "w", "v"],
        }
    )


@pytest.fixture()
def small_lf(small_df: pl.DataFrame) -> pl.LazyFrame:
    return small_df.lazy()


# ---- Tests ----


def test_get_control_samples_sql_filters_rows(small_lf: pl.LazyFrame):
    cfg = Config(
        {"feature_cols": ["f_a"], "control_sample_query": "batch = 'A' AND label = 1"}
    )
    out = get_control_samples(small_lf, cfg).collect(streaming=True)
    # Expect rows: indices 0 and 4 from the original fixture
    assert out.height == 2
    assert set(out.get_column("batch").to_list()) == {"A"}
    assert set(out.get_column("label").to_list()) == {1}


def test_get_feature_matrix_regex_selects_and_orders_columns(small_lf: pl.LazyFrame):
    cfg = Config({"feature_cols": "^f_", "control_sample_query": "label = 1"})
    cols, mat = get_feature_matrix(small_lf, cfg)
    # Should grab ["f_a", "f_b"] in schema order
    assert cols == ["f_a", "f_b"]
    # Matrix must have N rows equal to the source (no filter applied here) and 2 cols
    n = small_lf.select(pl.len()).collect(streaming=True).item()
    assert mat.shape == (n, 2)
    # Spot-check values for first row
    assert np.isclose(mat[0, 0], 0.1)
    assert np.isclose(mat[0, 1], 10.0)


def test_get_feature_matrix_list_selects_exact_columns(small_lf: pl.LazyFrame):
    cfg = Config({"feature_cols": ["f_b", "f_a"], "control_sample_query": "label = 1"})
    cols, mat = get_feature_matrix(small_lf, cfg)
    # Order should match the provided list
    assert cols == ["f_b", "f_a"]
    n = small_lf.select(pl.len()).collect(streaming=True).item()
    assert mat.shape == (n, 2)
    assert np.isclose(mat[0, 0], 10.0)
    assert np.isclose(mat[0, 1], 0.1)


def test_get_rows_by_idx_returns_correct_rows(
    small_df: pl.DataFrame, small_lf: pl.LazyFrame
):
    # Ask for rows 1 and 3 (zero-based), then compare to eager slice for truth
    wanted = [1, 3]
    out = get_rows_by_idx(small_lf, wanted).collect(streaming=True)

    expected = (
        small_df.select(pl.all())
        .slice(1, 1)
        .vstack(small_df.select(pl.all()).slice(3, 1))
    )
    # Order should follow the filter's order (set membership; may not guarantee order unless you sort)
    # Compare as sets of row tuples to be robust to order
    assert set(map(tuple, out.rows())) == set(map(tuple, expected.rows()))


def test_set_feature_matrix_replaces_columns(small_lf: pl.LazyFrame):
    # Create a replacement matrix for two feature columns
    feature_cols = ["f_a", "f_b"]
    n = small_lf.select(pl.len()).collect(streaming=True).item()
    new_features = np.vstack(
        [
            np.linspace(1.0, 2.0, n),
            np.linspace(100.0, 200.0, n),
        ]
    ).T  # shape (n, 2)

    out_lf = set_feature_matrix(small_lf, feature_cols, new_features)
    out = out_lf.collect(streaming=True)

    # Expect the two columns to match our replacements
    assert np.allclose(out["f_a"].to_numpy(), new_features[:, 0])
    assert np.allclose(out["f_b"].to_numpy(), new_features[:, 1])


def test_get_feature_selector_regex_matches_columns(small_lf: pl.LazyFrame):
    cfg = Config({"feature_cols": r"^f_", "control_sample_query": "label = 1"})
    selector = get_feature_selector(small_lf, cfg)
    out = small_lf.select(selector).collect(streaming=True)
    assert out.columns == ["f_a", "f_b"]


def test_get_feature_selector_list_warns_and_filters_missing(
    small_lf: pl.LazyFrame, caplog
):
    cfg = Config(
        {"feature_cols": ["f_a", "missing_col"], "control_sample_query": "label = 1"}
    )

    caplog.set_level(logging.WARNING)
    selector = get_feature_selector(small_lf, cfg)
    assert any(
        "ignored" in rec.message or "missing" in rec.message for rec in caplog.records
    )
    out = small_lf.select(selector).collect(streaming=True)
    assert out.columns == ["f_a"]


def test_get_feature_selector_list_preserves_requested_order(small_lf: pl.LazyFrame):
    cfg = Config({"feature_cols": ["f_b", "f_a"], "control_sample_query": "label = 1"})
    selector = get_feature_selector(small_lf, cfg)
    out = small_lf.select(selector).collect(streaming=True)
    assert out.columns == ["f_b", "f_a"]


def test_get_feature_columns_with_regex_returns_only_feature_cols(
    small_lf: pl.LazyFrame,
):
    cfg = Config({"feature_cols": r"^f_", "control_sample_query": "batch = 'A'"})
    out = get_feature_columns(small_lf, cfg).collect(streaming=True)
    assert out.columns == ["f_a", "f_b"]


def test_get_feature_columns_with_list_orders_and_logs_missing(
    small_lf: pl.LazyFrame, caplog
):
    cfg = Config(
        {
            "feature_cols": ["f_b", "missing_col", "f_a"],
            "control_sample_query": "batch = 'A'",
        }
    )
    caplog.set_level(logging.WARNING)

    out = get_feature_columns(small_lf, cfg).collect(streaming=True)

    # Check that the warning was logged
    assert any(
        "ignored" in rec.message or "missing" in rec.message for rec in caplog.records
    )

    # Should keep only the existing feature columns, in requested order
    assert out.columns == ["f_b", "f_a"]
