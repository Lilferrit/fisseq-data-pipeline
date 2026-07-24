from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import fisseq_data_pipeline.checkbarcodes as m
from fisseq_data_pipeline.checkbarcodes import (
    CheckBarcodesConfig,
    compute_barcode_tukey,
)

_LABEL_COL = "meta_aa_changes"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scores_df(seed: int = 0) -> pl.DataFrame:
    """Wide-format cell scores for two variants (V1 with three barcodes of
    unequal size and different means, V2 with a single barcode) plus WT
    cells that have no matching model column."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    bc_means = {"bc1": 0.2, "bc2": 0.25, "bc3": 0.6}
    bc_n = {"bc1": 12, "bc2": 20, "bc3": 8}
    for bc, mean in bc_means.items():
        for s in rng.normal(mean, 0.1, bc_n[bc]):
            rows.append(
                {"V1": s, "V2": rng.random(), _LABEL_COL: "V1", "meta_barcode": bc}
            )

    for s in rng.normal(0.5, 0.1, 15):
        rows.append(
            {
                "V1": rng.random(),
                "V2": s,
                _LABEL_COL: "V2",
                "meta_barcode": "bc_v2_only",
            }
        )

    for _ in range(10):
        rows.append(
            {
                "V1": rng.random(),
                "V2": rng.random(),
                _LABEL_COL: "WT",
                "meta_barcode": "bc_wt",
            }
        )

    return pl.DataFrame(rows)


def _make_cfg(tmp_path: Path, input_file: str, **overrides) -> OmegaConf:
    kwargs = dict(
        output_dir=str(tmp_path),
        input_file=input_file,
        min_cells=5,
        alpha=0.05,
    )
    kwargs.update(overrides)
    return OmegaConf.structured(CheckBarcodesConfig(**kwargs))


# ---------------------------------------------------------------------------
# compute_barcode_tukey — matches statsmodels' pairwise_tukeyhsd
# ---------------------------------------------------------------------------


def test_matches_statsmodels_pairwise_tukeyhsd():
    scores_df = _make_scores_df()
    result = compute_barcode_tukey(
        scores_df.lazy(), _LABEL_COL, min_cells=5, alpha=0.05
    ).sort("barcode", "comparison_barcode")

    v1 = scores_df.filter(pl.col(_LABEL_COL) == "V1")
    ref = pairwise_tukeyhsd(
        v1["V1"].to_numpy(), v1["meta_barcode"].to_list(), alpha=0.05
    )

    # Build a lookup from statsmodels' own results table rather than relying
    # on internal attributes.
    table_rows = ref._results_table.data[1:]
    ref_by_pair = {
        (row[0], row[1]): (float(row[2]), float(row[3]), bool(row[6]))
        for row in table_rows
    }

    result_v1 = result.filter(pl.col("variant") == "V1")
    assert result_v1.height == 3
    for row in result_v1.iter_rows(named=True):
        key = (row["barcode"], row["comparison_barcode"])
        assert key in ref_by_pair, f"pair {key} not found in statsmodels output"
        ref_meandiff, ref_p, ref_reject = ref_by_pair[key]
        assert row["mean_diff"] == pytest.approx(ref_meandiff, abs=1e-3)
        assert row["p_adj"] == pytest.approx(ref_p, abs=1e-3)
        assert row["reject"] == ref_reject


# ---------------------------------------------------------------------------
# compute_barcode_tukey — skip/filter behavior
# ---------------------------------------------------------------------------


def test_single_barcode_variant_is_skipped():
    scores_df = _make_scores_df()
    result = compute_barcode_tukey(
        scores_df.lazy(), _LABEL_COL, min_cells=5, alpha=0.05
    )
    assert "V2" not in result["variant"].to_list()


def test_wt_cells_excluded():
    scores_df = _make_scores_df()
    result = compute_barcode_tukey(
        scores_df.lazy(), _LABEL_COL, min_cells=5, alpha=0.05
    )
    assert "WT" not in result["variant"].to_list()


def test_barcode_below_min_cells_excluded():
    scores_df = _make_scores_df()
    # bc3 has 8 cells; raising min_cells above that drops it, leaving V1 with
    # only bc1/bc2 -- one pair instead of three.
    result = compute_barcode_tukey(
        scores_df.lazy(), _LABEL_COL, min_cells=9, alpha=0.05
    )
    v1 = result.filter(pl.col("variant") == "V1")
    assert v1.height == 1
    assert "bc3" not in v1["barcode"].to_list()
    assert "bc3" not in v1["comparison_barcode"].to_list()


def test_variant_with_no_qualifying_barcodes_after_min_cells_is_dropped():
    scores_df = _make_scores_df()
    # min_cells above every barcode's size drops everything.
    result = compute_barcode_tukey(
        scores_df.lazy(), _LABEL_COL, min_cells=1000, alpha=0.05
    )
    assert result.height == 0
    assert set(result.columns) == {
        "variant",
        "barcode",
        "group_mean",
        "comparison_barcode",
        "comparison_group_mean",
        "mean_diff",
        "p_adj",
        "reject",
    }


def test_reject_reflects_alpha():
    scores_df = _make_scores_df()
    strict = compute_barcode_tukey(
        scores_df.lazy(), _LABEL_COL, min_cells=5, alpha=1e-12
    )
    lenient = compute_barcode_tukey(
        scores_df.lazy(), _LABEL_COL, min_cells=5, alpha=0.999999
    )
    assert strict.filter(pl.col("variant") == "V1")["reject"].sum() == 0
    assert lenient.filter(pl.col("variant") == "V1")["reject"].sum() == 3


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_creates_output_file(tmp_path: Path):
    scores_df = _make_scores_df()
    input_path = tmp_path / "cell_scores.parquet"
    scores_df.write_parquet(input_path)
    cfg = _make_cfg(tmp_path, str(input_path))
    with patch("fisseq_data_pipeline.checkbarcodes.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "results.parquet").exists()


def test_main_output_has_expected_columns(tmp_path: Path):
    scores_df = _make_scores_df()
    input_path = tmp_path / "cell_scores.parquet"
    scores_df.write_parquet(input_path)
    cfg = _make_cfg(tmp_path, str(input_path))
    with patch("fisseq_data_pipeline.checkbarcodes.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "results.parquet")
    expected = {
        "variant",
        "barcode",
        "group_mean",
        "comparison_barcode",
        "comparison_group_mean",
        "mean_diff",
        "p_adj",
        "reject",
    }
    assert set(result.columns) == expected
    assert result.height > 0
    assert result["p_adj"].is_between(0.0, 1.0, closed="both").all()
