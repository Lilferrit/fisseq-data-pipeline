from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.permanova as m
from fisseq_data_pipeline.utils.constants import META_BATCH_COL
from fisseq_data_pipeline.permanova import PermanovaConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_feature_df(n_per_batch: int = 30, seed: int = 0) -> pl.DataFrame:
    """Two clearly separated batches with CellProfiler-style feature names."""
    rng = np.random.default_rng(seed)
    batch_a = rng.normal(loc=0.0, scale=0.1, size=(n_per_batch, 3))
    batch_b = rng.normal(loc=5.0, scale=0.1, size=(n_per_batch, 3))
    data = np.vstack([batch_a, batch_b])
    batches = ["batch_a"] * n_per_batch + ["batch_b"] * n_per_batch
    labels = ["WT"] * (n_per_batch * 2)
    return pl.DataFrame(
        {
            "meta_aa_changes": labels,
            META_BATCH_COL: batches,
            "Intensity_mean": data[:, 0].tolist(),
            "Texture_std": data[:, 1].tolist(),
            "Shape_area": data[:, 2].tolist(),
        }
    )


def _make_random_df(n_per_batch: int = 30, seed: int = 1) -> pl.DataFrame:
    """Two batches drawn from the same distribution (no batch effect)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_per_batch * 2, 3))
    batches = ["batch_a"] * n_per_batch + ["batch_b"] * n_per_batch
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * (n_per_batch * 2),
            META_BATCH_COL: batches,
            "Intensity_mean": data[:, 0].tolist(),
            "Texture_std": data[:, 1].tolist(),
            "Shape_area": data[:, 2].tolist(),
        }
    )


def make_perm_cfg(tmp_path: Path, input_file: str, **overrides) -> OmegaConf:
    kwargs = dict(
        output_dir=str(tmp_path),
        input_file=input_file,
        n_permutations=99,
        seed=42,
    )
    kwargs.update(overrides)
    return OmegaConf.structured(PermanovaConfig(**kwargs))


FEATURE_COLS = ["Intensity_mean", "Texture_std", "Shape_area"]


# ---------------------------------------------------------------------------
# _f_statistic
# ---------------------------------------------------------------------------


def test_f_statistic_hand_computed() -> None:
    # 4 samples, 2 batches of 2. Pairs: (0,1) same batch 0, (0,2) cross,
    # (0,3) cross, (1,2) cross, (1,3) cross, (2,3) same batch 1.
    idx_a = np.array([0, 0, 0, 1, 1, 2])
    idx_b = np.array([1, 2, 3, 2, 3, 3])
    d2 = np.array([1.0, 4.0, 9.0, 9.0, 16.0, 1.0])
    group_of_sample = np.array([0, 0, 1, 1])
    group_sizes = np.array([2, 2])
    n, a = 4, 2

    f = m._f_statistic(d2, idx_a, idx_b, group_of_sample, group_sizes, n, a)

    ss_total = d2.sum() / n
    ss_within = 1.0 / 2 + 1.0 / 2  # pairs (0,1) and (2,3), weight 1/n_g each
    ss_between = ss_total - ss_within
    expected = (ss_between / (a - 1)) / (ss_within / (n - a))
    assert f == pytest.approx(expected)


def test_f_statistic_separated_groups_larger_than_mixed() -> None:
    df_sep = _make_feature_df()
    df_mix = _make_random_df()

    def _f_for(df: pl.DataFrame) -> float:
        result = m.compute_variant_permanova(
            df.lazy(), FEATURE_COLS, META_BATCH_COL, n_permutations=0, seed=0
        )
        return result["f_statistic"]

    assert _f_for(df_sep) > _f_for(df_mix)


# ---------------------------------------------------------------------------
# compute_variant_permanova
# ---------------------------------------------------------------------------


def test_compute_variant_permanova_output_keys() -> None:
    df = _make_feature_df()
    result = m.compute_variant_permanova(
        df.lazy(), FEATURE_COLS, META_BATCH_COL, n_permutations=10, seed=0
    )
    assert set(result.keys()) == {"f_statistic", "p_value"}


def test_compute_variant_permanova_no_permutations_p_value_none() -> None:
    df = _make_feature_df()
    result = m.compute_variant_permanova(
        df.lazy(), FEATURE_COLS, META_BATCH_COL, n_permutations=0, seed=0
    )
    assert result["p_value"] is None


def test_compute_variant_permanova_p_value_in_range() -> None:
    df = _make_random_df()
    result = m.compute_variant_permanova(
        df.lazy(), FEATURE_COLS, META_BATCH_COL, n_permutations=50, seed=0
    )
    assert 0.0 < result["p_value"] <= 1.0


def test_compute_variant_permanova_separated_batches_significant() -> None:
    df = _make_feature_df()
    result = m.compute_variant_permanova(
        df.lazy(), FEATURE_COLS, META_BATCH_COL, n_permutations=99, seed=0
    )
    assert result["f_statistic"] > 10.0
    assert result["p_value"] < 0.05


def test_compute_variant_permanova_single_batch_returns_none() -> None:
    df = _make_feature_df().filter(pl.col(META_BATCH_COL) == "batch_a")
    result = m.compute_variant_permanova(
        df.lazy(), FEATURE_COLS, META_BATCH_COL, n_permutations=10, seed=0
    )
    assert result is None


def test_compute_variant_permanova_too_few_samples_returns_none() -> None:
    df = _make_feature_df().head(1)
    result = m.compute_variant_permanova(
        df.lazy(), FEATURE_COLS, META_BATCH_COL, n_permutations=10, seed=0
    )
    assert result is None


def test_compute_variant_permanova_drops_non_finite_rows() -> None:
    df = _make_feature_df()
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 3)
        .then(float("inf"))
        .otherwise(pl.col("Intensity_mean"))
        .alias("Intensity_mean")
    )
    result = m.compute_variant_permanova(
        df.lazy(), FEATURE_COLS, META_BATCH_COL, n_permutations=10, seed=0
    )
    assert result is not None


def test_compute_variant_permanova_uses_cross_batch_pairs() -> None:
    """If the self-join only paired same-batch samples, F would be undefined
    (SS_between would need cross-batch information to be meaningful)."""
    df = _make_feature_df()
    result = m.compute_variant_permanova(
        df.lazy(), FEATURE_COLS, META_BATCH_COL, n_permutations=0, seed=0
    )
    assert np.isfinite(result["f_statistic"])
    assert result["f_statistic"] > 0


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _write_batch_parquets(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for name, loc in [("batch_a", 0.0), ("batch_b", 5.0)]:
        data = {
            "meta_aa_changes": ["WT"] * 20,
            "Intensity_mean": rng.normal(loc, 0.1, 20).tolist(),
            "Texture_std": rng.normal(loc, 0.1, 20).tolist(),
        }
        pl.DataFrame(data).write_parquet(tmp_path / f"{name}.parquet")


def test_main_creates_output_file(tmp_path: Path) -> None:
    _write_batch_parquets(tmp_path)
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "permanova.parquet").exists()


def test_main_output_has_correct_columns(tmp_path: Path) -> None:
    _write_batch_parquets(tmp_path)
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "permanova.parquet")
    assert {"meta_aa_changes", "f_statistic", "p_value", "meta_num_cells"}.issubset(
        set(result.columns)
    )


def test_main_output_row_count_one_per_variant(tmp_path: Path) -> None:
    _write_batch_parquets(tmp_path)
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "permanova.parquet")
    assert len(result) == 1  # only one variant ("WT") present


def test_main_excludes_single_batch_variants(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 20,
            "Intensity_mean": rng.normal(0.0, 0.1, 20).tolist(),
            "Texture_std": rng.normal(0.0, 0.1, 20).tolist(),
        }
    ).write_parquet(tmp_path / "batch_a.parquet")
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1B"] * 20,
            "Intensity_mean": rng.normal(5.0, 0.1, 20).tolist(),
            "Texture_std": rng.normal(5.0, 0.1, 20).tolist(),
        }
    ).write_parquet(tmp_path / "batch_b.parquet")

    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "permanova.parquet")
    # "WT" only appears in batch_a and "A1B" only in batch_b -> neither has >1 batch
    assert len(result) == 0


def test_main_output_root_naming(tmp_path: Path) -> None:
    _write_batch_parquets(tmp_path)
    root = str(tmp_path / "run1")
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"), output_root=root)
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "run1.permanova.parquet").exists()


def test_main_n_permutations_zero_gives_null_p_value(tmp_path: Path) -> None:
    _write_batch_parquets(tmp_path)
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"), n_permutations=0)
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "permanova.parquet")
    assert result["p_value"][0] is None
