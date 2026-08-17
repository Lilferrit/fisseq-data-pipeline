from __future__ import annotations

import logging
from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.wtnullaggregate as m


def write_control_pool(tmp_path, n_control: int = 40) -> None:
    """Cell-level parquet with a heterogeneous control pool (distinct f1
    values so a random split of the pool produces a non-trivial aggregator
    value between halves) and no non-control rows -- WT-null only ever
    consumes the control pool."""
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * n_control,
            "meta_is_control": [True] * n_control,
            "f1": [float(i) for i in range(n_control)],
        }
    ).write_parquet(tmp_path / "input.parquet")


def make_cfg(
    tmp_path,
    *,
    aggregator="KS",
    bootstrap_idx=1,
    downsample_wt=None,
    per_barcode=False,
    barcode_column="meta_barcode",
) -> OmegaConf:
    return OmegaConf.structured(
        m.WtNullAggregateConfig(
            output_dir=str(tmp_path / "out"),
            input_file=str(tmp_path / "input.parquet"),
            aggregator=aggregator,
            bootstrap_idx=bootstrap_idx,
            downsample_wt=downsample_wt,
            per_barcode=per_barcode,
            barcode_column=barcode_column,
        )
    )


def run_main(cfg) -> pl.DataFrame:
    with patch("fisseq_data_pipeline.wtnullaggregate.setup_logging"):
        m.main.__wrapped__(cfg)
    return pl.read_parquet(cfg.output_dir + "/wt_null.parquet")


# ---------------------------------------------------------------------------
# _NULL_STATISTIC_TRANSFORMS
# ---------------------------------------------------------------------------


def _apply_transform(name: str, value: float) -> float:
    expr = m._NULL_STATISTIC_TRANSFORMS[name](pl.col("x"))
    return pl.DataFrame({"x": [value]}).select(expr.alias("x"))["x"][0]


def test_ks_transform_is_identity() -> None:
    assert _apply_transform("KS", 0.3) == pytest.approx(0.3)


def test_signedks_transform_is_absolute_value() -> None:
    assert _apply_transform("signedKS", -0.4) == pytest.approx(0.4)
    assert _apply_transform("signedKS", 0.4) == pytest.approx(0.4)


def test_qq_transform_is_one_minus_value() -> None:
    assert _apply_transform("QQ", 1.0) == pytest.approx(0.0)
    assert _apply_transform("QQ", 0.7) == pytest.approx(0.3)
    assert _apply_transform("QQ", -0.2) == pytest.approx(1.2)


def test_auroc_transform_is_absolute_deviation_from_half() -> None:
    assert _apply_transform("AUROC", 0.5) == pytest.approx(0.0)
    assert _apply_transform("AUROC", 0.9) == pytest.approx(0.4)
    assert _apply_transform("AUROC", 0.1) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_output_has_feature_and_value_columns(tmp_path) -> None:
    write_control_pool(tmp_path)
    result = run_main(make_cfg(tmp_path, aggregator="KS"))
    assert set(result.columns) == {"feature", "value"}
    # The full stat-suffixed column name is kept as the feature identity --
    # must match AGGREGATE_FEATURE_TYPE's own column naming exactly, since
    # FINALIZE_FEATURE_SELECT drops blocked columns by this same name.
    assert result["feature"].to_list() == ["f1_KS"]


def test_main_ks_value_bounded_in_unit_interval(tmp_path) -> None:
    write_control_pool(tmp_path)
    result = run_main(make_cfg(tmp_path, aggregator="KS"))
    value = result["value"][0]
    assert 0.0 <= value <= 1.0


def test_main_auroc_value_bounded_by_transform(tmp_path) -> None:
    write_control_pool(tmp_path)
    result = run_main(make_cfg(tmp_path, aggregator="AUROC"))
    value = result["value"][0]
    # abs(auroc - 0.5) is always in [0, 0.5]; a raw (untransformed) AUROC
    # could exceed 0.5, so this bound is a structural check that the
    # transform actually ran.
    assert 0.0 <= value <= 0.5


def test_main_qq_value_non_negative(tmp_path) -> None:
    write_control_pool(tmp_path)
    result = run_main(make_cfg(tmp_path, aggregator="QQ"))
    value = result["value"][0]
    # 1 - QQ is always >= 0 since Pearson correlation is bounded by 1.
    assert value >= 0.0


def test_main_bootstrap_idx_is_deterministic(tmp_path) -> None:
    write_control_pool(tmp_path)
    result1 = run_main(make_cfg(tmp_path, bootstrap_idx=3))
    result2 = run_main(make_cfg(tmp_path, bootstrap_idx=3))
    assert result1["value"][0] == pytest.approx(result2["value"][0])


def test_main_different_bootstrap_idx_changes_split(tmp_path) -> None:
    write_control_pool(tmp_path)
    result1 = run_main(make_cfg(tmp_path, bootstrap_idx=1))
    result2 = run_main(make_cfg(tmp_path, bootstrap_idx=2))
    assert result1["value"][0] != pytest.approx(result2["value"][0])


def test_main_downsample_wt_changes_output(tmp_path) -> None:
    write_control_pool(tmp_path)
    full = run_main(make_cfg(tmp_path, bootstrap_idx=1))
    down = run_main(make_cfg(tmp_path, bootstrap_idx=1, downsample_wt=5))
    assert full["value"][0] != pytest.approx(down["value"][0])


def test_main_downsample_wt_overflow_warns(tmp_path, caplog) -> None:
    write_control_pool(tmp_path, n_control=10)
    with caplog.at_level(logging.WARNING):
        run_main(make_cfg(tmp_path, bootstrap_idx=1, downsample_wt=1000))
    assert any("exceeds" in r.message for r in caplog.records)


def test_main_downsample_wt_float_out_of_range_raises(tmp_path) -> None:
    write_control_pool(tmp_path)
    with pytest.raises(ValueError):
        run_main(make_cfg(tmp_path, downsample_wt=1.5))


def test_main_downsample_wt_nonpositive_int_raises(tmp_path) -> None:
    write_control_pool(tmp_path)
    with pytest.raises(ValueError):
        run_main(make_cfg(tmp_path, downsample_wt=-1))


def test_main_ineligible_aggregator_raises(tmp_path) -> None:
    write_control_pool(tmp_path)
    with pytest.raises(ValueError):
        run_main(make_cfg(tmp_path, aggregator="mean"))


def test_main_unknown_aggregator_raises(tmp_path) -> None:
    write_control_pool(tmp_path)
    with pytest.raises(ValueError):
        run_main(make_cfg(tmp_path, aggregator="not_a_real_aggregator"))
