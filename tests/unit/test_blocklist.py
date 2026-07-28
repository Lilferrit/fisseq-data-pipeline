from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.blocklist as m

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def make_bl_cfg(
    tmp_path, correlation_files, *, minimum_correlation: float = 0.5
) -> OmegaConf:
    return OmegaConf.structured(
        m.BlocklistConfig(
            output_dir=str(tmp_path / "bl_out"),
            correlation_files=correlation_files,
            minimum_correlation=minimum_correlation,
        )
    )


def test_main_computes_median_r_across_bootstraps(tmp_path) -> None:
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    for i, r in enumerate([0.9, 0.5, 0.7], start=1):
        pl.DataFrame(
            {"feature": ["f1_mean"], "r": [r], "r_squared": [r**2]}
        ).write_parquet(corr_dir / f"bootstrap_{i}.parquet")

    with patch("fisseq_data_pipeline.blocklist.setup_logging"):
        m.main.__wrapped__(make_bl_cfg(tmp_path, str(corr_dir / "*.parquet")))

    result = pl.read_parquet(tmp_path / "bl_out" / "blocklist.parquet")
    row = result.filter(pl.col("feature") == "f1_mean").to_dicts().pop()
    assert row["median_r"] == pytest.approx(0.7)


def test_main_feature_ok_thresholding(tmp_path) -> None:
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    pl.DataFrame(
        {
            "feature": ["f1_mean", "f2_mean"],
            "r": [0.9, 0.2],
            "r_squared": [0.81, 0.04],
        }
    ).write_parquet(corr_dir / "bootstrap_1.parquet")

    with patch("fisseq_data_pipeline.blocklist.setup_logging"):
        m.main.__wrapped__(
            make_bl_cfg(tmp_path, str(corr_dir / "*.parquet"), minimum_correlation=0.5)
        )

    result = pl.read_parquet(tmp_path / "bl_out" / "blocklist.parquet")
    ok = dict(zip(result["feature"].to_list(), result["feature_ok"].to_list()))
    assert ok["f1_mean"] is True
    assert ok["f2_mean"] is False


def test_main_raises_on_empty_glob(tmp_path) -> None:
    with patch("fisseq_data_pipeline.blocklist.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(
                make_bl_cfg(tmp_path, str(tmp_path / "nonexistent" / "*.parquet"))
            )
