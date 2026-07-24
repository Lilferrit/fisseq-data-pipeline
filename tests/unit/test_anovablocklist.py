from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.anovablocklist as m
from fisseq_data_pipeline.anovablocklist import AnovaBlocklistConfig


def _write_anova_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "anova.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def make_cfg(tmp_path: Path, anova_file, **overrides) -> OmegaConf:
    kwargs = dict(
        output_dir=str(tmp_path / "out"),
        anova_file=str(anova_file),
    )
    kwargs.update(overrides)
    return OmegaConf.structured(AnovaBlocklistConfig(**kwargs))


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_creates_output_file(tmp_path: Path) -> None:
    anova_file = _write_anova_parquet(
        tmp_path,
        [{"feature": "f1", "f_value": 1.0, "p_value": 0.5}],
    )
    with patch("fisseq_data_pipeline.anovablocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, anova_file))
    assert (tmp_path / "out" / "anova_blocklist.parquet").exists()


def test_main_output_has_correct_columns(tmp_path: Path) -> None:
    anova_file = _write_anova_parquet(
        tmp_path,
        [{"feature": "f1", "f_value": 1.0, "p_value": 0.5}],
    )
    with patch("fisseq_data_pipeline.anovablocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, anova_file))
    result = pl.read_parquet(tmp_path / "out" / "anova_blocklist.parquet")
    assert set(result.columns) == {"feature", "p_value", "feature_ok"}


def test_main_row_count_matches_input(tmp_path: Path) -> None:
    anova_file = _write_anova_parquet(
        tmp_path,
        [
            {"feature": "f1", "f_value": 1.0, "p_value": 0.01},
            {"feature": "f2", "f_value": 2.0, "p_value": 0.5},
            {"feature": "f3", "f_value": 3.0, "p_value": 0.99},
        ],
    )
    with patch("fisseq_data_pipeline.anovablocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, anova_file))
    result = pl.read_parquet(tmp_path / "out" / "anova_blocklist.parquet")
    assert len(result) == 3
    assert set(result["feature"].to_list()) == {"f1", "f2", "f3"}


def test_main_blocks_features_below_threshold(tmp_path: Path) -> None:
    """p_value < threshold (a significant batch effect) -> feature_ok = False."""
    anova_file = _write_anova_parquet(
        tmp_path,
        [
            {"feature": "significant", "f_value": 10.0, "p_value": 0.01},
            {"feature": "not_significant", "f_value": 0.1, "p_value": 0.9},
        ],
    )
    with patch("fisseq_data_pipeline.anovablocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, anova_file, pvalue_threshold=0.05))
    result = pl.read_parquet(tmp_path / "out" / "anova_blocklist.parquet")
    ok = dict(zip(result["feature"].to_list(), result["feature_ok"].to_list()))
    assert ok["significant"] is False
    assert ok["not_significant"] is True


def test_main_threshold_boundary_p_value_equal_to_threshold_not_blocked(
    tmp_path: Path,
) -> None:
    """Blocking is strict '<' -- p_value exactly equal to the threshold is kept."""
    anova_file = _write_anova_parquet(
        tmp_path,
        [{"feature": "boundary", "f_value": 5.0, "p_value": 0.05}],
    )
    with patch("fisseq_data_pipeline.anovablocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, anova_file, pvalue_threshold=0.05))
    result = pl.read_parquet(tmp_path / "out" / "anova_blocklist.parquet")
    assert result.filter(pl.col("feature") == "boundary")["feature_ok"][0] is True


def test_main_empty_blocklist_when_all_features_pass(tmp_path: Path) -> None:
    anova_file = _write_anova_parquet(
        tmp_path,
        [
            {"feature": "f1", "f_value": 0.1, "p_value": 0.4},
            {"feature": "f2", "f_value": 0.2, "p_value": 0.8},
        ],
    )
    with patch("fisseq_data_pipeline.anovablocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, anova_file, pvalue_threshold=0.05))
    result = pl.read_parquet(tmp_path / "out" / "anova_blocklist.parquet")
    assert result["feature_ok"].all()


def test_main_all_features_blocked(tmp_path: Path) -> None:
    anova_file = _write_anova_parquet(
        tmp_path,
        [
            {"feature": "f1", "f_value": 10.0, "p_value": 0.001},
            {"feature": "f2", "f_value": 20.0, "p_value": 0.002},
        ],
    )
    with patch("fisseq_data_pipeline.anovablocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, anova_file, pvalue_threshold=0.05))
    result = pl.read_parquet(tmp_path / "out" / "anova_blocklist.parquet")
    assert not result["feature_ok"].any()


def test_main_p_value_column_preserved(tmp_path: Path) -> None:
    anova_file = _write_anova_parquet(
        tmp_path,
        [{"feature": "f1", "f_value": 1.0, "p_value": 0.1234}],
    )
    with patch("fisseq_data_pipeline.anovablocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, anova_file))
    result = pl.read_parquet(tmp_path / "out" / "anova_blocklist.parquet")
    assert result["p_value"][0] == pytest.approx(0.1234)


def test_main_default_threshold_is_point_zero_five(tmp_path: Path) -> None:
    anova_file = _write_anova_parquet(
        tmp_path,
        [{"feature": "f1", "f_value": 1.0, "p_value": 0.049}],
    )
    with patch("fisseq_data_pipeline.anovablocklist.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, anova_file))
    result = pl.read_parquet(tmp_path / "out" / "anova_blocklist.parquet")
    assert result["feature_ok"][0] is False
