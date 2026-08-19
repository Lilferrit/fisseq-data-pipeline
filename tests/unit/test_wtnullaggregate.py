from __future__ import annotations

import logging
from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.wtnullaggregate as m


def write_control_pool(
    tmp_path, n_control: int = 40, with_barcode: bool = False
) -> None:
    """Cell-level parquet with a heterogeneous control pool (distinct f1
    values so a random split of the pool produces a non-trivial aggregator
    value between halves) and no non-control rows -- WT-null only ever
    consumes the control pool."""
    data = {
        "meta_aa_changes": ["WT"] * n_control,
        "meta_is_control": [True] * n_control,
        "f1": [float(i) for i in range(n_control)],
    }
    if with_barcode:
        data["meta_barcode"] = [f"bc{i % 4}" for i in range(n_control)]
    pl.DataFrame(data).write_parquet(tmp_path / "input.parquet")


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
# main
# ---------------------------------------------------------------------------
# Note: per-aggregator null_statistic_transform behavior (identity for
# KS/mean/median/std, abs() for signedKS, 1-x for QQ, abs(x-0.5) for AUROC)
# used to be pinned here via a module-level _NULL_STATISTIC_TRANSFORMS dict.
# That logic now lives on the aggregator classes themselves
# (BaseAggregator.null_statistic_transform and its overrides) and is tested
# in test_aggregate.py, alongside is_null_eligible/null_eligible_aggregator_names.


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
    # MAD is registered but deliberately WT-null-ineligible (see
    # MADAggregator's docstring in aggregate.py) -- unlike "mean", which
    # became eligible as part of this refactor.
    with pytest.raises(ValueError):
        run_main(make_cfg(tmp_path, aggregator="MAD"))


def test_main_ksneglogp_raises(tmp_path) -> None:
    write_control_pool(tmp_path)
    with pytest.raises(ValueError):
        run_main(make_cfg(tmp_path, aggregator="KSnegLogP"))


def test_main_aurocneglogp_raises(tmp_path) -> None:
    write_control_pool(tmp_path)
    with pytest.raises(ValueError):
        run_main(make_cfg(tmp_path, aggregator="AUROCnegLogP"))


def test_main_unknown_aggregator_raises(tmp_path) -> None:
    write_control_pool(tmp_path)
    with pytest.raises(ValueError):
        run_main(make_cfg(tmp_path, aggregator="not_a_real_aggregator"))


# ---------------------------------------------------------------------------
# one-sample WT-null path (mean/median/std)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aggregator", ["mean", "median", "std"])
def test_main_succeeds_for_one_sample_aggregators(tmp_path, aggregator) -> None:
    write_control_pool(tmp_path)
    result = run_main(make_cfg(tmp_path, aggregator=aggregator))
    assert set(result.columns) == {"feature", "value"}
    assert result["feature"].to_list() == [f"f1_{aggregator}"]
    # The one-sample path's null_statistic_transform is identity over an
    # absolute difference, so the value is always non-negative.
    assert result["value"][0] >= 0.0


# ---------------------------------------------------------------------------
# per_barcode passthrough
# ---------------------------------------------------------------------------


def test_main_per_barcode_end_to_end_one_sample(tmp_path) -> None:
    write_control_pool(tmp_path, with_barcode=True)
    result = run_main(
        make_cfg(
            tmp_path,
            aggregator="mean",
            per_barcode=True,
            barcode_column="meta_barcode",
        )
    )
    assert set(result.columns) == {"feature", "value"}
    assert result["feature"].to_list() == ["f1_mean"]


def test_main_per_barcode_end_to_end_reference_based(tmp_path) -> None:
    write_control_pool(tmp_path, with_barcode=True)
    result = run_main(
        make_cfg(
            tmp_path,
            aggregator="KS",
            per_barcode=True,
            barcode_column="meta_barcode",
        )
    )
    assert set(result.columns) == {"feature", "value"}
    assert result["feature"].to_list() == ["f1_KS"]
