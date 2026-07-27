from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.barcodeblocklist as m
from fisseq_data_pipeline.barcodeblocklist import (
    BarcodeBlocklistConfig,
    compute_barcode_blocklist,
)


def _write_results_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "results.parquet"
    schema = {
        "variant": pl.Utf8,
        "barcode": pl.Utf8,
        "group_mean": pl.Float64,
        "comparison_barcode": pl.Utf8,
        "comparison_group_mean": pl.Float64,
        "mean_diff": pl.Float64,
        "p_adj": pl.Float64,
        "reject": pl.Boolean,
    }
    pl.DataFrame(rows, schema=schema if rows else schema).write_parquet(path)
    return path


def _row(
    barcode: str,
    comparison_barcode: str,
    p_adj: float,
    variant: str = "V1",
) -> dict:
    return {
        "variant": variant,
        "barcode": barcode,
        "group_mean": 0.0,
        "comparison_barcode": comparison_barcode,
        "comparison_group_mean": 0.0,
        "mean_diff": 0.0,
        "p_adj": p_adj,
        "reject": p_adj < 0.05,
    }


def make_cfg(tmp_path: Path, check_barcodes_file, **overrides) -> OmegaConf:
    kwargs = dict(
        output_dir=str(tmp_path / "out"),
        check_barcodes_file=str(check_barcodes_file),
    )
    kwargs.update(overrides)
    return OmegaConf.structured(BarcodeBlocklistConfig(**kwargs))


# ---------------------------------------------------------------------------
# compute_barcode_blocklist
# ---------------------------------------------------------------------------


def test_compute_barcode_blocklist_unions_both_pair_columns() -> None:
    """A barcode's p_adj values are scattered across `barcode` and
    `comparison_barcode` depending on which side of the alphabetical dedup
    it lands on for a given pair -- the median must be computed over all of
    them, not just the subset from one column."""
    results_df = pl.DataFrame(
        [
            _row("bc1", "bc2", p_adj=0.10),  # bc1 as `barcode`
            _row("bc0", "bc1", p_adj=0.90),  # bc1 as `comparison_barcode`
            _row("bc1", "bc3", p_adj=0.50),  # bc1 as `barcode`
        ]
    )
    result = compute_barcode_blocklist(results_df, pvalue_threshold=0.05)
    bc1_p_adj = result.filter(pl.col("barcode") == "bc1")["p_adj"][0]
    assert bc1_p_adj == pytest.approx(0.50)  # median of [0.10, 0.90, 0.50]


def test_compute_barcode_blocklist_one_row_per_distinct_barcode() -> None:
    results_df = pl.DataFrame(
        [
            _row("bc1", "bc2", p_adj=0.10),
            _row("bc2", "bc3", p_adj=0.20),
        ]
    )
    result = compute_barcode_blocklist(results_df, pvalue_threshold=0.05)
    assert set(result["barcode"].to_list()) == {"bc1", "bc2", "bc3"}


def test_compute_barcode_blocklist_empty_input_has_correct_schema() -> None:
    results_df = pl.DataFrame(
        schema={
            "variant": pl.Utf8,
            "barcode": pl.Utf8,
            "group_mean": pl.Float64,
            "comparison_barcode": pl.Utf8,
            "comparison_group_mean": pl.Float64,
            "mean_diff": pl.Float64,
            "p_adj": pl.Float64,
            "reject": pl.Boolean,
        }
    )
    result = compute_barcode_blocklist(results_df, pvalue_threshold=0.05)
    assert result.height == 0
    assert result.schema == {
        "barcode": pl.Utf8,
        "p_adj": pl.Float64,
        "barcode_ok": pl.Boolean,
    }


def test_compute_barcode_blocklist_boundary_p_adj_equal_to_threshold_is_ok() -> None:
    """barcode_ok is non-strict '>=' -- median p_adj exactly equal to the
    threshold is kept."""
    results_df = pl.DataFrame([_row("bc1", "bc2", p_adj=0.05)])
    result = compute_barcode_blocklist(results_df, pvalue_threshold=0.05)
    assert result.filter(pl.col("barcode") == "bc1")["barcode_ok"][0] is True


def test_compute_barcode_blocklist_below_threshold_is_blocked() -> None:
    results_df = pl.DataFrame([_row("bc1", "bc2", p_adj=0.01)])
    result = compute_barcode_blocklist(results_df, pvalue_threshold=0.05)
    assert result.filter(pl.col("barcode") == "bc1")["barcode_ok"][0] is False


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_creates_output_file(tmp_path: Path) -> None:
    check_barcodes_file = _write_results_parquet(
        tmp_path, [_row("bc1", "bc2", p_adj=0.5)]
    )
    with patch("fisseq_data_pipeline.barcodeblocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, check_barcodes_file))
    assert (tmp_path / "out" / "barcode_blocklist.parquet").exists()


def test_main_output_has_correct_columns(tmp_path: Path) -> None:
    check_barcodes_file = _write_results_parquet(
        tmp_path, [_row("bc1", "bc2", p_adj=0.5)]
    )
    with patch("fisseq_data_pipeline.barcodeblocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, check_barcodes_file))
    result = pl.read_parquet(tmp_path / "out" / "barcode_blocklist.parquet")
    assert set(result.columns) == {"barcode", "p_adj", "barcode_ok"}


def test_main_blocks_barcodes_below_threshold(tmp_path: Path) -> None:
    check_barcodes_file = _write_results_parquet(
        tmp_path,
        [
            _row("significant", "other", p_adj=0.01),
            _row("not_significant", "other2", p_adj=0.9),
        ],
    )
    with patch("fisseq_data_pipeline.barcodeblocklist.setup_logging"):
        m.main.__wrapped__(
            make_cfg(tmp_path, check_barcodes_file, pvalue_threshold=0.05)
        )
    result = pl.read_parquet(tmp_path / "out" / "barcode_blocklist.parquet")
    ok = dict(zip(result["barcode"].to_list(), result["barcode_ok"].to_list()))
    assert ok["significant"] is False
    assert ok["not_significant"] is True
    # "other"/"other2" barcodes get their own rows too (median over their
    # one comparison each), and are not below threshold.
    assert ok["other"] is False
    assert ok["other2"] is True


def test_main_default_threshold_is_point_zero_five(tmp_path: Path) -> None:
    check_barcodes_file = _write_results_parquet(
        tmp_path, [_row("bc1", "bc2", p_adj=0.049)]
    )
    with patch("fisseq_data_pipeline.barcodeblocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, check_barcodes_file))
    result = pl.read_parquet(tmp_path / "out" / "barcode_blocklist.parquet")
    assert result.filter(pl.col("barcode") == "bc1")["barcode_ok"][0] is False
