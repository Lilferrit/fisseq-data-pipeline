from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.combineblocklists as m

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def make_cb_cfg(tmp_path, blocklist_files) -> OmegaConf:
    return OmegaConf.structured(
        m.CombineBlocklistsConfig(
            output_dir=str(tmp_path / "cb_out"),
            blocklist_files=blocklist_files,
        )
    )


def test_main_concatenates_disjoint_features(tmp_path) -> None:
    bl_dir = tmp_path / "bl"
    bl_dir.mkdir()
    pl.DataFrame(
        {
            "feature": ["f1_mean"],
            "r_est": [0.9],
            "se_z": [0.01],
            "n_replicates": [10],
            "feature_ok": [True],
        }
    ).write_parquet(bl_dir / "mean.parquet")
    pl.DataFrame(
        {
            "feature": ["f1_std"],
            "r_est": [0.3],
            "se_z": [0.05],
            "n_replicates": [10],
            "feature_ok": [False],
        }
    ).write_parquet(bl_dir / "std.parquet")

    with patch("fisseq_data_pipeline.combineblocklists.setup_logging"):
        m.main.__wrapped__(make_cb_cfg(tmp_path, str(bl_dir / "*.parquet")))

    result = pl.read_parquet(tmp_path / "cb_out" / "blocklist.parquet")
    assert set(result["feature"].to_list()) == {"f1_mean", "f1_std"}
    assert len(result) == 2


def test_main_raises_on_empty_glob(tmp_path) -> None:
    with patch("fisseq_data_pipeline.combineblocklists.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(
                make_cb_cfg(tmp_path, str(tmp_path / "nonexistent" / "*.parquet"))
            )
