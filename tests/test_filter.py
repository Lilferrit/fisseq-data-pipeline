import polars as pl
import pytest
from polars.testing import assert_frame_equal

import fisseq_data_pipeline.filter as mod


class DummyConfig:
    def __init__(self, _cfg):
        self._cfg = _cfg


@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch):
    """
    Patch helper symbols that drop_feature_* depend on:
      - Config(...)
      - get_feature_selector(cfg) -> list[str]
      - get_feature_columns(lf, cfg) -> LazyFrame of just the feature cols
    """
    # make Config(...) construct our dummy
    monkeypatch.setattr(mod, "Config", DummyConfig, raising=True)

    # default feature set for tests; individual tests can monkeypatch again
    monkeypatch.setattr(
        mod, "get_feature_selector", lambda df, cfg: ["f1", "f2"], raising=True
    )

    def _get_feature_columns(lf: pl.LazyFrame, cfg: DummyConfig) -> pl.LazyFrame:
        return lf.select(["f1", "f2"])

    monkeypatch.setattr(mod, "get_feature_columns", _get_feature_columns, raising=True)


def test_drop_feature_null_drops_rows_with_nan_in_selected_features():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "f1": [1.0, 2.0, None, 4.0],  # NaN/Null in row 3
            "f2": [5.0, None, 7.0, 8.0],  # NaN/Null in row 2
            "meta": ["a", "b", "c", "d"],
        }
    )
    lf = df.lazy()

    out = mod.drop_feature_null(lf, config=None).collect()
    print(out)

    expected = pl.DataFrame(
        {
            "id": [1, 4],
            "f1": [1.0, 4.0],
            "f2": [5.0, 8.0],
            "meta": ["a", "d"],
        }
    )
    assert_frame_equal(out, expected)


def test_drop_feature_null_ignores_nans_outside_selected_features(monkeypatch):
    # Only f1 is a feature this time; NaN in f2 shouldn't trigger a drop.
    monkeypatch.setattr(
        mod, "get_feature_selector", lambda df, cfg: ["f1"], raising=True
    )

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "f1": [10.0, 11.0, 12.0],
            "f2": [None, 2.0, None],  # not part of the subset
        }
    )
    lf = df.lazy()

    out = mod.drop_feature_null(lf, config=None).collect()
    assert_frame_equal(out, df)  # unchanged


def test_drop_feature_zero_var_drops_constant_feature_columns(monkeypatch):
    # Features are f1, f2, f3
    monkeypatch.setattr(
        mod, "get_feature_selector", lambda cfg: ["f1", "f2", "f3"], raising=True
    )

    def _get_feature_columns(lf: pl.LazyFrame, cfg: DummyConfig) -> pl.LazyFrame:
        return lf.select(["f1", "f2", "f3"])

    monkeypatch.setattr(mod, "get_feature_columns", _get_feature_columns, raising=True)

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "f1": [1.0, 1.0, 1.0],  # constant → should be dropped
            "f2": [0.0, 1.0, 2.0],  # varying → should be kept
            "f3": [5.5, 5.5, 5.5],  # constant → should be dropped
            "meta": ["x", "y", "z"],  # non-feature → should be kept
        }
    )
    lf = df.lazy()

    out = mod.drop_feature_zero_var(lf, config=None).collect()

    # Expect only non-constant features + all non-feature columns
    expected = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "f2": [0.0, 1.0, 2.0],
            "meta": ["x", "y", "z"],
        }
    )
    # Column order can differ depending on implementation; align columns
    out = out.select(expected.columns)
    assert_frame_equal(out, expected)


def test_drop_feature_zero_var_only_checks_feature_columns(monkeypatch):
    # Only f1,f2 are features; const_meta is constant but not a feature → should remain.
    monkeypatch.setattr(
        mod, "get_feature_selector", lambda cfg: ["f1", "f2"], raising=True
    )

    def _get_feature_columns(lf: pl.LazyFrame, cfg: DummyConfig) -> pl.LazyFrame:
        return lf.select(["f1", "f2"])

    monkeypatch.setattr(mod, "get_feature_columns", _get_feature_columns, raising=True)

    df = pl.DataFrame(
        {
            "id": [10, 11, 12],
            "f1": [3.0, 3.0, 3.0],  # constant feature → drop
            "f2": [9.0, 8.0, 9.0],  # varying feature → keep
            "const_meta": ["m", "m", "m"],  # constant but NOT a feature → keep
        }
    )
    lf = df.lazy()

    out = mod.drop_feature_zero_var(lf, config=None).collect()

    print(out)

    expected = pl.DataFrame(
        {
            "id": [10, 11, 12],
            "f2": [9.0, 8.0, 9.0],
            "const_meta": ["m", "m", "m"],
        }
    )
    out = out.select(expected.columns)
    assert_frame_equal(out, expected)


def test_run_sequential_filters_applies_in_order():
    call_order = []

    def f1(df: pl.LazyFrame, cfg):
        call_order.append("null")
        return mod.drop_feature_null(df, cfg)

    def f2(df: pl.LazyFrame, cfg):
        call_order.append("zero")
        return mod.drop_feature_zero_var(df, cfg)

    lf = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "f1": [1.0, None, 3.0, 4.0],  # null so drop_feature_null does work
            "f2": [0.0, 1.0, 2.0, 3.0],
            "meta": ["a", "b", "c", "d"],
        }
    ).lazy()

    _ = mod.run_sequential_filters(lf, config=None, filter_funs=[f1, f2]).schema
    assert call_order == ["null", "zero"]


def test_run_sequential_filters_default_pipeline_runs_and_drops_nulls():
    lf = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "f1": [1.0, None, 3.0, 4.0],  # null → row 2 should be dropped
            "f2": [5.0, 6.0, 7.0, 8.0],
            "meta": ["a", "b", "c", "d"],
        }
    ).lazy()

    out = mod.run_sequential_filters(lf, config=None).collect()

    # No nulls in feature columns (f1, f2) after the default pipeline
    assert out["f1"].null_count() == 0
    assert out["f2"].null_count() == 0
    # Still a DataFrame with remaining rows
    assert isinstance(out, pl.DataFrame)


def test_run_sequential_filters_with_custom_filters():
    def add_flag(df: pl.LazyFrame, cfg):
        return df.with_columns(pl.lit(1).alias("flag"))

    def keep_even_ids(df: pl.LazyFrame, cfg):
        return df.filter((pl.col("id") % 2) == 0)

    lf = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [10.0, 20.0, 30.0, 40.0],
            "meta": ["a", "b", "c", "d"],
        }
    ).lazy()

    out = mod.run_sequential_filters(
        lf, config=None, filter_funs=[add_flag, keep_even_ids]
    ).collect()
    assert "flag" in out.columns
    assert out["id"].to_list() == [2, 4]


def test_run_sequential_filters_passes_config_through():
    seen = []

    def spy(df: pl.LazyFrame, cfg):
        seen.append(cfg)
        return df

    cfg = {"sentinel": 42}
    lf = pl.DataFrame(
        {
            "id": [1, 2],
            "f1": [1.0, 2.0],
            "f2": [10.0, 20.0],
        }
    ).lazy()

    _ = mod.run_sequential_filters(lf, cfg, [spy, spy, spy]).schema
    assert seen == [cfg, cfg, cfg]
