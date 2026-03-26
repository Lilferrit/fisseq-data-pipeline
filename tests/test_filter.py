import math

import polars as pl
import pytest

from fisseq_data_pipeline.filter import (
    clean_data,
    drop_cols_all_nonfinite,
    drop_rows_any_nonfinite,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def f32(**kwargs) -> pl.DataFrame:
    """Build a Float32 DataFrame from keyword-column arrays."""
    return pl.DataFrame(
        {k: pl.Series(k, v, dtype=pl.Float32) for k, v in kwargs.items()}
    )


# ---------------------------------------------------------------------------
# drop_cols_all_nonfinite
# ---------------------------------------------------------------------------


def test_drop_cols_all_nonfinite_removes_all_null_column():
    lf = f32(good=[1.0, 2.0, 3.0], all_null=[None, None, None]).lazy()
    out = drop_cols_all_nonfinite(lf).collect()
    assert "all_null" not in out.columns
    assert "good" in out.columns


def test_drop_cols_all_nonfinite_removes_all_nan_column():
    lf = f32(good=[1.0, 2.0], all_nan=[math.nan, math.nan]).lazy()
    out = drop_cols_all_nonfinite(lf).collect()
    assert "all_nan" not in out.columns
    assert "good" in out.columns


def test_drop_cols_all_nonfinite_removes_all_inf_column():
    inf = math.inf
    lf = f32(good=[1.0, 2.0], all_inf=[inf, -inf]).lazy()
    out = drop_cols_all_nonfinite(lf).collect()
    assert "all_inf" not in out.columns


def test_drop_cols_all_nonfinite_keeps_partially_null_column():
    lf = f32(partial=[None, 1.0, None]).lazy()
    out = drop_cols_all_nonfinite(lf).collect()
    assert "partial" in out.columns
    assert out.height == 3  # rows unchanged by this stage


def test_drop_cols_all_nonfinite_keeps_all_finite_column():
    lf = f32(a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0]).lazy()
    out = drop_cols_all_nonfinite(lf).collect()
    assert out.columns == ["a", "b"]


def test_drop_cols_all_nonfinite_returns_lazy_frame():
    lf = f32(a=[1.0, 2.0]).lazy()
    result = drop_cols_all_nonfinite(lf)
    assert isinstance(result, pl.LazyFrame)


def test_drop_cols_all_nonfinite_preserves_meta_columns():
    df = f32(feat=[1.0, 2.0], all_null=[None, None])
    df = df.with_columns(pl.Series("_meta_label", ["A", "B"]))
    out = drop_cols_all_nonfinite(df.lazy()).collect()
    assert "_meta_label" in out.columns
    assert "all_null" not in out.columns


def test_drop_cols_all_nonfinite_drops_multiple_bad_columns():
    lf = f32(
        good=[1.0, 2.0],
        bad1=[None, None],
        bad2=[math.nan, math.nan],
    ).lazy()
    out = drop_cols_all_nonfinite(lf).collect()
    assert out.columns == ["good"]


def test_drop_cols_all_nonfinite_no_bad_columns():
    lf = f32(a=[1.0], b=[2.0]).lazy()
    out = drop_cols_all_nonfinite(lf).collect()
    assert set(out.columns) == {"a", "b"}
    assert out.height == 1


# ---------------------------------------------------------------------------
# drop_rows_any_nonfinite
# ---------------------------------------------------------------------------


def test_drop_rows_any_nonfinite_removes_null_row():
    lf = f32(a=[1.0, None, 3.0], b=[4.0, 5.0, 6.0]).lazy()
    out = drop_rows_any_nonfinite(lf).collect()
    assert out.height == 2
    assert out["a"].to_list() == pytest.approx([1.0, 3.0])


def test_drop_rows_any_nonfinite_removes_nan_row():
    lf = f32(a=[1.0, math.nan, 3.0]).lazy()
    out = drop_rows_any_nonfinite(lf).collect()
    assert out.height == 2


def test_drop_rows_any_nonfinite_removes_inf_row():
    lf = f32(a=[1.0, math.inf, 3.0], b=[4.0, 5.0, 6.0]).lazy()
    out = drop_rows_any_nonfinite(lf).collect()
    assert out.height == 2


def test_drop_rows_any_nonfinite_removes_row_with_any_bad_column():
    # row 1: a is fine, b is null → should be dropped
    lf = f32(a=[1.0, 2.0, 3.0], b=[4.0, None, 6.0]).lazy()
    out = drop_rows_any_nonfinite(lf).collect()
    assert out.height == 2
    assert out["a"].to_list() == pytest.approx([1.0, 3.0])


def test_drop_rows_any_nonfinite_keeps_all_finite_rows():
    lf = f32(a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0]).lazy()
    out = drop_rows_any_nonfinite(lf).collect()
    assert out.height == 3


def test_drop_rows_any_nonfinite_returns_lazy_frame():
    lf = f32(a=[1.0]).lazy()
    result = drop_rows_any_nonfinite(lf)
    assert isinstance(result, pl.LazyFrame)


def test_drop_rows_any_nonfinite_preserves_meta_columns():
    df = f32(feat=[1.0, None, 3.0])
    df = df.with_columns(pl.Series("_meta_label", ["A", "B", "C"]))
    out = drop_rows_any_nonfinite(df.lazy()).collect()
    assert out.height == 2
    assert "_meta_label" in out.columns
    assert out["_meta_label"].to_list() == ["A", "C"]


def test_drop_rows_any_nonfinite_does_not_drop_bad_meta_column():
    # null in a _meta column should NOT trigger row removal
    df = pl.DataFrame(
        {
            "feat": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
            "_meta_label": ["A", None, "C"],
        }
    )
    out = drop_rows_any_nonfinite(df.lazy()).collect()
    assert out.height == 3


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------


def test_clean_data_returns_lazy_frame():
    lf = f32(a=[1.0, 2.0]).lazy()
    result = clean_data(lf)
    assert isinstance(result, pl.LazyFrame)


def test_clean_data_default_pipeline_removes_null_col_and_null_row():
    lf = f32(
        good=[1.0, 2.0, 3.0],
        all_null=[None, None, None],
        partial=[None, 5.0, 6.0],
    ).lazy()
    out = clean_data(lf).collect()
    # all_null dropped by drop_cols_all_nonfinite
    assert "all_null" not in out.columns
    # row 0 has null in 'partial' → dropped by drop_rows_any_nonfinite
    assert out.height == 2
    assert out["good"].to_list() == pytest.approx([2.0, 3.0])


def test_clean_data_empty_stages_returns_input_unchanged():
    lf = f32(a=[1.0, None]).lazy()
    out = clean_data(lf, stages=[]).collect()
    # no filtering applied
    assert out.height == 2


def test_clean_data_only_drop_cols_stage():
    lf = f32(good=[1.0, None], bad=[None, None]).lazy()
    out = clean_data(lf, stages=["drop_cols_all_nonfinite"]).collect()
    assert "bad" not in out.columns
    # null row not removed because drop_rows stage wasn't run
    assert out.height == 2


def test_clean_data_only_drop_rows_stage():
    # b has one null; drop_cols stage not run so b is kept
    lf = f32(a=[1.0, None, 3.0], b=[0.0, None, 0.0]).lazy()
    out = clean_data(lf, stages=["drop_rows_any_nonfinite"]).collect()
    # all-null column kept (stage not run)
    assert "b" in out.columns
    # rows where any feature is null are dropped (rows 1)
    assert out.height == 2
    assert out["a"].to_list() == pytest.approx([1.0, 3.0])


def test_clean_data_unknown_stage_is_skipped_with_warning(caplog):
    lf = f32(a=[1.0, 2.0]).lazy()
    with caplog.at_level("WARNING"):
        out = clean_data(lf, stages=["no_such_stage"]).collect()
    # data unchanged
    assert out.height == 2
    assert any("no_such_stage" in rec.message for rec in caplog.records)


def test_clean_data_callable_stage_is_applied():
    calls = []

    def my_stage(lf: pl.LazyFrame) -> pl.LazyFrame:
        calls.append(True)
        return lf.filter(pl.col("a") > 1.0)

    lf = f32(a=[1.0, 2.0, 3.0]).lazy()
    out = clean_data(lf, stages=[my_stage]).collect()
    assert len(calls) == 1
    assert out.height == 2


def test_clean_data_stages_execute_in_order():
    """Column drop must happen before row drop to avoid operating on dead cols."""
    # all_null would cause is_finite() to error on non-float types if not dropped first,
    # but here we verify order by tracking which stage sees which columns.
    seen_cols = []

    def record_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
        seen_cols.append(set(lf.collect_schema().names()))
        return lf

    lf = f32(keep=[1.0, 2.0], drop_me=[None, None]).lazy()
    clean_data(lf, stages=["drop_cols_all_nonfinite", record_cols])

    assert "drop_me" not in seen_cols[0]
    assert "keep" in seen_cols[0]


def test_clean_data_mixed_string_and_callable_stages():
    dropped_cols = []

    def capture(lf: pl.LazyFrame) -> pl.LazyFrame:
        dropped_cols.extend(lf.collect_schema().names())
        return lf

    lf = f32(good=[1.0, 2.0], bad=[None, None]).lazy()
    clean_data(lf, stages=["drop_cols_all_nonfinite", capture])
    assert "bad" not in dropped_cols
    assert "good" in dropped_cols
