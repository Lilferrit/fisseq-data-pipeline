from __future__ import annotations

import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.passthroughblocklist as m


def make_cfg(tmp_path, aggregate_file) -> OmegaConf:
    return OmegaConf.structured(
        m.PassthroughBlocklistConfig(
            output_dir=str(tmp_path / "out"),
            aggregate_file=aggregate_file,
        )
    )


def run_main(tmp_path, aggregate_file) -> pl.DataFrame:
    m.main.__wrapped__(make_cfg(tmp_path, aggregate_file))
    return pl.read_parquet(tmp_path / "out" / "blocklist.parquet")


def test_main_output_schema(tmp_path) -> None:
    agg_path = tmp_path / "mean.parquet"
    pl.DataFrame(
        {"meta_aa_changes": ["WT", "A1B"], "f1_mean": [1.0, 2.0]}
    ).write_parquet(agg_path)
    result = run_main(tmp_path, str(agg_path))
    assert set(result.columns) == {
        "feature",
        "feature_ok",
        "null_mean",
        "threshold",
        "n_bootstraps",
    }


def test_main_every_feature_marked_ok(tmp_path) -> None:
    agg_path = tmp_path / "mean.parquet"
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "A1B"],
            "f1_mean": [1.0, 2.0],
            "f2_mean": [3.0, 4.0],
        }
    ).write_parquet(agg_path)
    result = run_main(tmp_path, str(agg_path))
    assert set(result["feature"].to_list()) == {"f1_mean", "f2_mean"}
    assert result["feature_ok"].to_list() == [True, True]


def test_main_audit_columns_are_null(tmp_path) -> None:
    agg_path = tmp_path / "mean.parquet"
    pl.DataFrame(
        {"meta_aa_changes": ["WT", "A1B"], "f1_mean": [1.0, 2.0]}
    ).write_parquet(agg_path)
    result = run_main(tmp_path, str(agg_path))
    assert result["null_mean"].null_count() == result.height
    assert result["threshold"].null_count() == result.height
    assert result["n_bootstraps"].null_count() == result.height


def test_main_excludes_meta_columns_from_feature_list(tmp_path) -> None:
    agg_path = tmp_path / "mean.parquet"
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "A1B"],
            "meta_batch": ["b1", "b1"],
            "f1_mean": [1.0, 2.0],
        }
    ).write_parquet(agg_path)
    result = run_main(tmp_path, str(agg_path))
    assert result["feature"].to_list() == ["f1_mean"]


def test_main_empty_feature_set_produces_empty_blocklist(tmp_path) -> None:
    agg_path = tmp_path / "mean.parquet"
    pl.DataFrame({"meta_aa_changes": ["WT", "A1B"]}).write_parquet(agg_path)
    result = run_main(tmp_path, str(agg_path))
    assert result.height == 0
    assert result["feature"].dtype == pl.String


def test_main_schema_concat_compatible_with_wtnullblocklist(tmp_path) -> None:
    import fisseq_data_pipeline.wtnullblocklist as wtnb

    agg_path = tmp_path / "mean.parquet"
    pl.DataFrame(
        {"meta_aa_changes": ["WT", "A1B"], "f1_mean": [1.0, 2.0]}
    ).write_parquet(agg_path)
    passthrough_result = run_main(tmp_path, str(agg_path))

    boot_dir = tmp_path / "boots"
    boot_dir.mkdir()
    pl.DataFrame({"feature": ["f2_KS"], "value": [0.1]}).write_parquet(
        boot_dir / "bootstrap_0.parquet"
    )
    wtnb_cfg = OmegaConf.structured(
        wtnb.WtNullBlocklistConfig(
            output_dir=str(tmp_path / "wtnb_out"),
            wt_null_files=str(boot_dir / "*.parquet"),
        )
    )
    wtnb.main.__wrapped__(wtnb_cfg)
    wtnb_result = pl.read_parquet(tmp_path / "wtnb_out" / "blocklist.parquet")

    combined = pl.concat([passthrough_result, wtnb_result])
    assert combined.height == 2


def test_main_raises_when_aggregate_file_missing(tmp_path) -> None:
    with pytest.raises(Exception):
        run_main(tmp_path, str(tmp_path / "does_not_exist.parquet"))
