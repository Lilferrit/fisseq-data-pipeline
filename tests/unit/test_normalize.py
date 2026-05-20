import math
from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

from fisseq_data_pipeline.normalize import (
    NormalizeConfig,
    Normalizer,
    add_control_indicator_column,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_lf(
    f1: list,
    f2: list,
    control: list[bool] | None = None,
) -> pl.LazyFrame:
    """Build a Float64 LazyFrame with optional meta_is_control column."""
    data = {
        "f1": pl.Series("f1", f1, dtype=pl.Float64),
        "f2": pl.Series("f2", f2, dtype=pl.Float64),
    }
    if control is not None:
        data["meta_is_control"] = pl.Series("meta_is_control", control)
    return pl.DataFrame(data).lazy()


# ---------------------------------------------------------------------------
# Normalizer.from_lazyframe
# ---------------------------------------------------------------------------


def test_from_lazyframe_computes_mean():
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    assert n.means["f1"][0] == pytest.approx(2.0)
    assert n.means["f2"][0] == pytest.approx(20.0)


def test_from_lazyframe_computes_std():
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    assert n.stds["f1"][0] == pytest.approx(1.0)
    assert n.stds["f2"][0] == pytest.approx(10.0)


def test_from_lazyframe_fit_only_on_control_uses_control_rows_only():
    # non-control rows have extreme values that would shift the mean if included
    lf = make_lf(
        f1=[1.0, 999.0, 3.0, 999.0],
        f2=[10.0, 999.0, 30.0, 999.0],
        control=[True, False, True, False],
    )
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=True)
    assert n.means["f1"][0] == pytest.approx(2.0)
    assert n.means["f2"][0] == pytest.approx(20.0)


def test_from_lazyframe_fit_only_on_control_false_uses_all_rows():
    lf = make_lf(
        f1=[1.0, 3.0],
        f2=[10.0, 30.0],
        control=[True, False],
    )
    n_all = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    n_ctrl = Normalizer.from_lazyframe(lf, fit_only_on_control=True)
    # with only one control row std is undefined; means should differ
    assert n_all.means["f1"][0] == pytest.approx(2.0)
    assert n_ctrl.means["f1"][0] == pytest.approx(1.0)


def test_from_lazyframe_zero_variance_column_stored_as_none():
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[7.0, 7.0, 7.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    assert n.stds["f2"][0] is None


def test_from_lazyframe_nonzero_variance_column_not_none():
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    assert n.stds["f1"][0] is not None
    assert n.stds["f2"][0] is not None


def test_from_lazyframe_excludes_nan_from_statistics():
    # NaN in f1 row 1 should be ignored; mean of [1.0, 3.0] = 2.0
    lf = make_lf(f1=[1.0, math.nan, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    assert n.means["f1"][0] == pytest.approx(2.0)


def test_from_lazyframe_meta_columns_not_in_statistics():
    lf = make_lf(f1=[1.0, 2.0], f2=[3.0, 4.0], control=[True, False])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    assert "meta_is_control" not in n.means.columns
    assert "meta_is_control" not in n.stds.columns


# ---------------------------------------------------------------------------
# Normalizer.apply
# ---------------------------------------------------------------------------


def test_apply_returns_lazy_frame():
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    result = n.apply(lf)
    assert isinstance(result, pl.LazyFrame)


def test_apply_standardizes_features():
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    out = n.apply(lf).collect()
    assert out["f1"].to_list() == pytest.approx([-1.0, 0.0, 1.0])
    assert out["f2"].to_list() == pytest.approx([-1.0, 0.0, 1.0])


def test_apply_preserves_meta_columns():
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0], control=[True, False, True])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    out = n.apply(lf).collect()
    assert "meta_is_control" in out.columns
    assert out["meta_is_control"].to_list() == [True, False, True]


def test_apply_converts_nan_input_to_null():
    lf = make_lf(f1=[1.0, math.nan, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(
        make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0]),
        fit_only_on_control=False,
    )
    out = n.apply(lf).collect()
    assert out["f1"][1] is None


def test_apply_zero_variance_feature_produces_null():
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[7.0, 7.0, 7.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    out = n.apply(lf).collect()
    # f2 has None std → division produces NaN → converted to null
    assert all(v is None for v in out["f2"].to_list())


# ---------------------------------------------------------------------------
# Normalizer.save / Normalizer.load
# ---------------------------------------------------------------------------


def test_save_creates_file(tmp_path):
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    path = tmp_path / "normalizer.parquet"
    n.save(path)
    assert path.exists()


def test_load_restores_means(tmp_path):
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    path = tmp_path / "normalizer.parquet"
    n.save(path)
    loaded = Normalizer.load(path)
    assert loaded.means["f1"][0] == pytest.approx(n.means["f1"][0])
    assert loaded.means["f2"][0] == pytest.approx(n.means["f2"][0])


def test_load_restores_stds(tmp_path):
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    path = tmp_path / "normalizer.parquet"
    n.save(path)
    loaded = Normalizer.load(path)
    assert loaded.stds["f1"][0] == pytest.approx(n.stds["f1"][0])
    assert loaded.stds["f2"][0] == pytest.approx(n.stds["f2"][0])


def test_save_load_apply_roundtrip(tmp_path):
    lf = make_lf(f1=[1.0, 2.0, 3.0], f2=[10.0, 20.0, 30.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    path = tmp_path / "normalizer.parquet"
    n.save(path)
    loaded = Normalizer.load(path)
    original_out = n.apply(lf).collect()
    loaded_out = loaded.apply(lf).collect()
    assert original_out["f1"].to_list() == pytest.approx(loaded_out["f1"].to_list())
    assert original_out["f2"].to_list() == pytest.approx(loaded_out["f2"].to_list())


def test_load_does_not_include_stat_column(tmp_path):
    lf = make_lf(f1=[1.0, 2.0], f2=[3.0, 4.0])
    n = Normalizer.from_lazyframe(lf, fit_only_on_control=False)
    path = tmp_path / "normalizer.parquet"
    n.save(path)
    loaded = Normalizer.load(path)
    assert "_stat" not in loaded.means.columns
    assert "_stat" not in loaded.stds.columns


# ---------------------------------------------------------------------------
# Helpers for add_control_indicator_column / main
# ---------------------------------------------------------------------------


def make_control_lf(
    f1: list,
    f2: list,
    aa_changes: list[str],
) -> pl.LazyFrame:
    """Build a LazyFrame with feature columns and meta_aa_changes."""
    return pl.DataFrame(
        {
            "f1": pl.Series("f1", f1, dtype=pl.Float64),
            "f2": pl.Series("f2", f2, dtype=pl.Float64),
            "meta_aa_changes": pl.Series("meta_aa_changes", aa_changes),
        }
    ).lazy()


def make_cfg(tmp_path, *, output_root=None, save_normalizer=False) -> OmegaConf:
    """Return a DictConfig for NormalizeConfig with sensible test defaults."""
    return OmegaConf.structured(
        NormalizeConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            input_file=str(tmp_path / "input.parquet"),
            save_normalizer=save_normalizer,
        )
    )


def write_input_parquet(tmp_path) -> None:
    """Write a minimal test parquet to tmp_path/input.parquet."""
    pl.DataFrame(
        {
            "f1": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64),
            "f2": pl.Series([10.0, 20.0, 30.0, 40.0, 50.0], dtype=pl.Float64),
            "meta_aa_changes": ["WT", "WT", "WT", "M1A", "M1A"],
        }
    ).write_parquet(tmp_path / "input.parquet")


# ---------------------------------------------------------------------------
# add_control_indicator_column
# ---------------------------------------------------------------------------


def test_add_control_indicator_adds_column():
    lf = make_control_lf(f1=[1.0, 2.0], f2=[3.0, 4.0], aa_changes=["WT", "M1A"])
    cfg = NormalizeConfig(output_dir="/tmp", input_file="/tmp/x.parquet")
    result = add_control_indicator_column(lf, cfg).collect()
    assert "meta_is_control" in result.columns


def test_add_control_indicator_wt_rows_are_true():
    lf = make_control_lf(
        f1=[1.0, 2.0, 3.0],
        f2=[4.0, 5.0, 6.0],
        aa_changes=["WT", "M1A", "WT"],
    )
    cfg = NormalizeConfig(output_dir="/tmp", input_file="/tmp/x.parquet")
    result = add_control_indicator_column(lf, cfg).collect()
    assert result["meta_is_control"].to_list() == [True, False, True]


def test_add_control_indicator_non_wt_rows_are_false():
    lf = make_control_lf(f1=[1.0], f2=[2.0], aa_changes=["G12V"])
    cfg = NormalizeConfig(output_dir="/tmp", input_file="/tmp/x.parquet")
    result = add_control_indicator_column(lf, cfg).collect()
    assert result["meta_is_control"][0] is False


def test_add_control_indicator_column_is_boolean():
    lf = make_control_lf(f1=[1.0, 2.0], f2=[3.0, 4.0], aa_changes=["WT", "M1A"])
    cfg = NormalizeConfig(output_dir="/tmp", input_file="/tmp/x.parquet")
    result = add_control_indicator_column(lf, cfg).collect()
    assert result["meta_is_control"].dtype == pl.Boolean


def test_add_control_indicator_preserves_other_columns():
    lf = make_control_lf(f1=[1.0, 2.0], f2=[3.0, 4.0], aa_changes=["WT", "M1A"])
    cfg = NormalizeConfig(output_dir="/tmp", input_file="/tmp/x.parquet")
    result = add_control_indicator_column(lf, cfg).collect()
    assert "f1" in result.columns
    assert "f2" in result.columns
    assert "meta_aa_changes" in result.columns


def test_add_control_indicator_custom_query():
    lf = make_control_lf(f1=[1.0, 2.0], f2=[3.0, 4.0], aa_changes=["WT", "M1A"])
    cfg = NormalizeConfig(
        output_dir="/tmp",
        input_file="/tmp/x.parquet",
        control_sample_query="meta_aa_changes = 'M1A'",
    )
    result = add_control_indicator_column(lf, cfg).collect()
    assert result["meta_is_control"].to_list() == [False, True]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_creates_output_file(tmp_path):
    write_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.normalize.setup_logging"):
        main.__wrapped__(make_cfg(tmp_path))
    assert (tmp_path / "out" / "input.parquet").exists()


def test_main_output_values_are_normalized(tmp_path):
    write_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.normalize.setup_logging"):
        main.__wrapped__(make_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    # WT rows (indices 0-2) were the fit population; their z-scored mean ≈ 0
    assert result["f1"][:3].mean() == pytest.approx(0.0, abs=1e-6)


def test_main_preserves_meta_columns(tmp_path):
    write_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.normalize.setup_logging"):
        main.__wrapped__(make_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "meta_aa_changes" in result.columns
    assert "meta_is_control" in result.columns


def test_main_output_root_naming(tmp_path):
    write_input_parquet(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.normalize.setup_logging"):
        main.__wrapped__(make_cfg(tmp_path, output_root=root))
    assert (tmp_path / "run1.input.parquet").exists()


def test_main_saves_normalizer_when_configured(tmp_path):
    write_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.normalize.setup_logging"):
        main.__wrapped__(make_cfg(tmp_path, save_normalizer=True))
    assert (tmp_path / "out" / "normalizer.parquet").exists()
