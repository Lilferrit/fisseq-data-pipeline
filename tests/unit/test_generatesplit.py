from __future__ import annotations

from unittest.mock import patch

import polars as pl
from omegaconf import OmegaConf

import fisseq_data_pipeline.generatesplit as m
from fisseq_data_pipeline.utils.splits import TMP_IDX_COL

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def write_split_input_parquet(tmp_path) -> None:
    """Cell-level parquet with 4 label groups, 4 cells each (16 rows total)."""
    n = 4
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * n + ["A1A"] * n + ["A1B"] * n + ["A1C"] * n,
            "f1": list(range(4 * n)),
        }
    ).write_parquet(tmp_path / "split_input.parquet")


def make_split_cfg(tmp_path, *, random_state: int = 0) -> OmegaConf:
    return OmegaConf.structured(
        m.GenerateSplitConfig(
            output_dir=str(tmp_path / "split_out"),
            input_file=str(tmp_path / "split_input.parquet"),
            random_state=random_state,
        )
    )


def test_main_writes_both_halves(tmp_path) -> None:
    write_split_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.generatesplit.setup_logging"):
        m.main.__wrapped__(make_split_cfg(tmp_path))
    assert (tmp_path / "split_out" / "half1.parquet").exists()
    assert (tmp_path / "split_out" / "half2.parquet").exists()


def test_main_halves_carry_tmp_idx_col(tmp_path) -> None:
    write_split_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.generatesplit.setup_logging"):
        m.main.__wrapped__(make_split_cfg(tmp_path))
    half1 = pl.read_parquet(tmp_path / "split_out" / "half1.parquet")
    assert half1.columns == [TMP_IDX_COL]


def test_main_halves_disjoint_and_cover_all_rows(tmp_path) -> None:
    write_split_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.generatesplit.setup_logging"):
        m.main.__wrapped__(make_split_cfg(tmp_path))
    half1 = set(
        pl.read_parquet(tmp_path / "split_out" / "half1.parquet")[TMP_IDX_COL].to_list()
    )
    half2 = set(
        pl.read_parquet(tmp_path / "split_out" / "half2.parquet")[TMP_IDX_COL].to_list()
    )
    assert half1.isdisjoint(half2)
    assert half1 | half2 == set(range(16))


def test_main_random_state_is_deterministic(tmp_path) -> None:
    write_split_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.generatesplit.setup_logging"):
        m.main.__wrapped__(make_split_cfg(tmp_path, random_state=7))
    half1_first = sorted(
        pl.read_parquet(tmp_path / "split_out" / "half1.parquet")[TMP_IDX_COL].to_list()
    )

    with patch("fisseq_data_pipeline.generatesplit.setup_logging"):
        m.main.__wrapped__(make_split_cfg(tmp_path, random_state=7))
    half1_second = sorted(
        pl.read_parquet(tmp_path / "split_out" / "half1.parquet")[TMP_IDX_COL].to_list()
    )

    assert half1_first == half1_second
