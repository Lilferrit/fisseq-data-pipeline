"""Unit tests for GLOBAL_OVWT's cross-experiment score correction + aggregation."""

import pathlib
import subprocess
import sys

import polars as pl
import pytest

from fisseq_data_pipeline.globalovwt import (
    GlobalOvwtConfig,
    global_variant_distinguishability,
)

LABEL = "meta_aa_changes"


def _results(
    rows: list[tuple[str, float, float]],
    label_column: str = LABEL,
) -> pl.DataFrame:
    """One OVWT_BATCHWISE results.parquet: (variant, auroc_pooled, auroc_median_barcode)."""
    return pl.DataFrame(
        {
            label_column: [r[0] for r in rows],
            "auroc_pooled": [r[1] for r in rows],
            "auroc_median_barcode": [r[2] for r in rows],
            "meta_n_barcodes": [2] * len(rows),
            "meta_n_cells": [100] * len(rows),
        }
    )


# "A1A"/"L5L" are synonymous (same first and last residue) and so act as the
# control population the z-score is fit on; "M1K" is a real missense variant.
_SYNONYMOUS = [("A1A", 0.50, 0.50), ("L5L", 0.60, 0.60)]


def test_config_defaults():
    cfg = GlobalOvwtConfig(output_dir="o", batch_stems=["a"])
    assert cfg.label_column == "meta_aa_changes"
    assert cfg.random_seed == 0


def test_empty_input_raises():
    with pytest.raises(ValueError, match="non-empty"):
        global_variant_distinguishability([], LABEL)


def test_output_schema():
    out = global_variant_distinguishability(
        [_results(_SYNONYMOUS + [("M1K", 0.95, 0.93)])], LABEL
    )
    assert set(out.columns) == {
        LABEL,
        "meta_median_auroc_pooled",
        "meta_median_auroc_median_barcode",
        "meta_num_experiments",
    }


def test_one_row_per_variant():
    out = global_variant_distinguishability(
        [
            _results(_SYNONYMOUS + [("M1K", 0.95, 0.93)]),
            _results(_SYNONYMOUS + [("M1K", 0.90, 0.88)]),
        ],
        LABEL,
    )
    assert len(out) == 3
    assert set(out.get_column(LABEL).to_list()) == {"A1A", "L5L", "M1K"}


def test_num_experiments_counts_contributors():
    out = global_variant_distinguishability(
        [
            _results(_SYNONYMOUS + [("M1K", 0.95, 0.93)]),
            _results(_SYNONYMOUS + [("M1K", 0.90, 0.88), ("R2H", 0.80, 0.79)]),
        ],
        LABEL,
    )
    counts = dict(zip(out.get_column(LABEL), out.get_column("meta_num_experiments")))
    assert counts["M1K"] == 2
    assert counts["R2H"] == 1


def test_scores_are_zscored_not_raw_medians():
    """
    The whole point of the stage: a variant far above its experiment's own
    synonymous population lands well above 0, not at its raw AUROC.
    """
    out = global_variant_distinguishability(
        [_results(_SYNONYMOUS + [("M1K", 0.95, 0.93)])], LABEL
    )
    m1k = out.filter(pl.col(LABEL) == "M1K")
    score = m1k.get_column("meta_median_auroc_pooled")[0]
    assert score != pytest.approx(0.95)
    assert score > 1.0  # several synonymous SDs above the control mean


def test_per_experiment_recentering_makes_experiments_comparable():
    """
    Two experiments whose raw AUROCs sit on different scales but whose variant
    is equally far above each one's own synonymous baseline must agree after
    correction -- a plain median of raw AUROC would not.
    """
    low = _results([("A1A", 0.30, 0.30), ("L5L", 0.40, 0.40), ("M1K", 0.60, 0.60)])
    high = _results([("A1A", 0.60, 0.60), ("L5L", 0.70, 0.70), ("M1K", 0.90, 0.90)])
    both = global_variant_distinguishability([low, high], LABEL)
    only_low = global_variant_distinguishability([low], LABEL)
    only_high = global_variant_distinguishability([high], LABEL)

    def m1k(df):
        return df.filter(pl.col(LABEL) == "M1K").get_column("meta_median_auroc_pooled")[
            0
        ]

    assert m1k(only_low) == pytest.approx(m1k(only_high))
    assert m1k(both) == pytest.approx(m1k(only_low))


def test_metadata_columns_are_not_zscored_away():
    """FEATURE_SELECTOR must pick up only the two AUROC columns."""
    out = global_variant_distinguishability(
        [_results(_SYNONYMOUS + [("M1K", 0.95, 0.93)])], LABEL
    )
    # meta_n_barcodes / meta_n_cells are excluded from the aggregation entirely
    assert "meta_n_barcodes" not in out.columns
    assert "meta_n_cells" not in out.columns


def test_cli_end_to_end_single_experiment(tmp_path: pathlib.Path):
    """Exercises reconstruct_staged_paths' n == 1 naming (res_input_.parquet)."""
    _results(_SYNONYMOUS + [("M1K", 0.95, 0.93)]).write_parquet(
        tmp_path / "res_input_.parquet"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_data_pipeline.globalovwt",
            "output_dir=.",
            "batch_stems=[plate1]",
            f"label_column={LABEL}",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(tmp_path / "global_scores.parquet")
    assert set(out.get_column(LABEL).to_list()) == {"A1A", "L5L", "M1K"}


def test_cli_end_to_end_multiple_experiments(tmp_path: pathlib.Path):
    _results(_SYNONYMOUS + [("M1K", 0.95, 0.93)]).write_parquet(
        tmp_path / "res_input_1.parquet"
    )
    _results(_SYNONYMOUS + [("M1K", 0.90, 0.88)]).write_parquet(
        tmp_path / "res_input_2.parquet"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_data_pipeline.globalovwt",
            "output_dir=.",
            "batch_stems=[plate1,plate2]",
            f"label_column={LABEL}",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(tmp_path / "global_scores.parquet")
    m1k = out.filter(pl.col(LABEL) == "M1K")
    assert m1k.get_column("meta_num_experiments")[0] == 2
