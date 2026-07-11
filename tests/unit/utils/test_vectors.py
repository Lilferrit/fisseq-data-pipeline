from __future__ import annotations

import polars as pl
import pytest

from fisseq_data_pipeline.utils.constants import CONTROL_COLUMN_NAME, IMPACT_SCORE_COL
from fisseq_data_pipeline.utils.vectors import (
    COSINE_DIST_COL,
    DOT_COL,
    NORM_COL,
    compute_cosine_distance,
    compute_impact_score,
    compute_norm,
    compute_query_dot,
)

# ---------------------------------------------------------------------------
# compute_norm
# ---------------------------------------------------------------------------


def test_compute_norm_pythagorean() -> None:
    lf = pl.LazyFrame({"f1": [3.0], "f2": [4.0]})
    result = compute_norm(lf).collect()
    assert result[NORM_COL][0] == pytest.approx(5.0)


def test_compute_norm_unit_vector() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [0.0]})
    result = compute_norm(lf).collect()
    assert result[NORM_COL][0] == pytest.approx(1.0)


def test_compute_norm_multiple_rows() -> None:
    lf = pl.LazyFrame({"f1": [3.0, 0.0], "f2": [4.0, 5.0]})
    result = compute_norm(lf).collect()
    assert result[NORM_COL].to_list() == pytest.approx([5.0, 5.0])


def test_compute_norm_preserves_meta_columns() -> None:
    lf = pl.LazyFrame({"meta_label": ["a"], "f1": [3.0], "f2": [4.0]})
    result = compute_norm(lf).collect()
    assert "meta_label" in result.columns
    assert result["meta_label"][0] == "a"


def test_compute_norm_excludes_meta_from_norm() -> None:
    # meta columns must not contribute to the norm value
    lf = pl.LazyFrame({"meta_label": [99.0], "f1": [3.0], "f2": [4.0]})
    result = compute_norm(lf).collect()
    assert result[NORM_COL][0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# compute_query_dot
# ---------------------------------------------------------------------------


def test_compute_query_dot_basic() -> None:
    value_lf = pl.LazyFrame({"f1": [1.0], "f2": [2.0]})
    query_lf = pl.LazyFrame({"f1": [3.0], "f2": [4.0]})
    result = compute_query_dot(value_lf, query_lf).collect()
    assert result[DOT_COL][0] == pytest.approx(11.0)  # 1·3 + 2·4


def test_compute_query_dot_orthogonal() -> None:
    value_lf = pl.LazyFrame({"f1": [1.0], "f2": [0.0]})
    query_lf = pl.LazyFrame({"f1": [0.0], "f2": [1.0]})
    result = compute_query_dot(value_lf, query_lf).collect()
    assert result[DOT_COL][0] == pytest.approx(0.0)


def test_compute_query_dot_drops_query_columns() -> None:
    value_lf = pl.LazyFrame({"f1": [1.0], "f2": [2.0]})
    query_lf = pl.LazyFrame({"f1": [3.0], "f2": [4.0]})
    result = compute_query_dot(value_lf, query_lf).collect()
    assert "f1_query" not in result.columns
    assert "f2_query" not in result.columns


def test_compute_query_dot_preserves_meta_columns() -> None:
    value_lf = pl.LazyFrame({"meta_label": ["x"], "f1": [1.0], "f2": [2.0]})
    query_lf = pl.LazyFrame({"f1": [3.0], "f2": [4.0]})
    result = compute_query_dot(value_lf, query_lf).collect()
    assert "meta_label" in result.columns
    assert result["meta_label"][0] == "x"


def test_compute_query_dot_raises_on_empty_query() -> None:
    value_lf = pl.LazyFrame({"f1": [1.0]})
    query_lf = pl.LazyFrame({"f1": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="query must have exactly 1 row"):
        compute_query_dot(value_lf, query_lf)


def test_compute_query_dot_raises_on_multi_row_query() -> None:
    value_lf = pl.LazyFrame({"f1": [1.0]})
    query_lf = pl.LazyFrame({"f1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="query must have exactly 1 row"):
        compute_query_dot(value_lf, query_lf)


# ---------------------------------------------------------------------------
# compute_impact_score
# ---------------------------------------------------------------------------


def test_compute_impact_score_control_is_zero() -> None:
    lf = pl.LazyFrame({CONTROL_COLUMN_NAME: [True], "f1": [1.0], "f2": [0.0]})
    result = compute_impact_score(lf).collect()
    assert result[IMPACT_SCORE_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_impact_score_orthogonal_is_half() -> None:
    # Control along f1; test row along f2 — cosine angle = 90° → score = 0.5
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, False],
            "f1": [1.0, 0.0],
            "f2": [0.0, 1.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(0.5, abs=1e-9)


def test_compute_impact_score_opposite_is_one() -> None:
    # Opposite direction to the control median → max impact score
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, False],
            "f1": [1.0, -1.0],
            "f2": [0.0, 0.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(1.0, abs=1e-9)


def test_compute_impact_score_drops_temp_columns() -> None:
    lf = pl.LazyFrame({CONTROL_COLUMN_NAME: [True], "f1": [1.0], "f2": [0.0]})
    result = compute_impact_score(lf).collect()
    assert NORM_COL not in result.columns
    assert DOT_COL not in result.columns


def test_compute_impact_score_output_columns() -> None:
    lf = pl.LazyFrame({CONTROL_COLUMN_NAME: [True], "f1": [1.0], "f2": [0.0]})
    result = compute_impact_score(lf).collect()
    assert IMPACT_SCORE_COL in result.columns
    assert CONTROL_COLUMN_NAME in result.columns
    assert "f1" in result.columns


def test_compute_impact_score_null_columns_excluded_from_calc() -> None:
    # f2 has a null so it is excluded; score is computed using f1 only.
    # Control median (f1=1); non-control (f1=-1) → opposite direction → score = 1.
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, False],
            "f1": [1.0, -1.0],
            "f2": [None, 1.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(1.0, abs=1e-9)


def test_compute_impact_score_null_columns_kept_in_output() -> None:
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True],
            "f1": [1.0],
            "f2": [None],
        }
    )
    result = compute_impact_score(lf).collect()
    assert "f2" in result.columns


def test_compute_impact_score_uses_control_median() -> None:
    # Two control rows whose median is (1, 0); verify the non-control score
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, True, False],
            "f1": [0.0, 2.0, 0.0],  # median f1 = 1.0
            "f2": [0.0, 0.0, 1.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    # control median = (1, 0); test = (0, 1) → orthogonal → 0.5
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_cosine_distance
# ---------------------------------------------------------------------------


def test_compute_cosine_distance_identical_vectors_is_zero() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [2.0], "f1_b": [1.0], "f2_b": [2.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_cosine_distance_orthogonal_vectors_is_one() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [0.0], "f1_b": [0.0], "f2_b": [1.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(1.0, abs=1e-9)


def test_compute_cosine_distance_opposite_vectors_is_two() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [0.0], "f1_b": [-1.0], "f2_b": [0.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(2.0, abs=1e-9)


def test_compute_cosine_distance_zero_norm_no_nan() -> None:
    lf = pl.LazyFrame({"f1": [0.0], "f2": [0.0], "f1_b": [1.0], "f2_b": [0.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert not result[COSINE_DIST_COL].is_nan().any()


def test_compute_cosine_distance_keeps_suffixed_columns() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f1_b": [1.0]})
    result = compute_cosine_distance(lf, ["f1"], suffix="_b").collect()
    assert "f1_b" in result.columns


def test_compute_cosine_distance_null_feature_excluded_one_side() -> None:
    # f2 is null on the unsuffixed side, so only f1 is used -> identical -> 0.0
    lf = pl.LazyFrame({"f1": [1.0], "f2": [None], "f1_b": [1.0], "f2_b": [5.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_cosine_distance_null_feature_excluded_other_side() -> None:
    # f2 is null on the suffixed side instead -> same exclusion, same result
    lf = pl.LazyFrame({"f1": [1.0], "f2": [5.0], "f1_b": [1.0], "f2_b": [None]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_cosine_distance_nan_feature_excluded() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [float("nan")], "f1_b": [1.0], "f2_b": [5.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_cosine_distance_inf_feature_excluded() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [float("inf")], "f1_b": [1.0], "f2_b": [5.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_cosine_distance_all_features_excluded_no_nan() -> None:
    lf = pl.LazyFrame({"f1": [None]}, schema={"f1": pl.Float64})
    lf = lf.with_columns(f1_b=pl.lit(None, dtype=pl.Float64))
    result = compute_cosine_distance(lf, ["f1"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(1.0, abs=1e-9)
    assert not result[COSINE_DIST_COL].is_nan().any()
