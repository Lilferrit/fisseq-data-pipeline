from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.permanova as m
from fisseq_data_pipeline.constants import META_BATCH_COL
from fisseq_data_pipeline.permanova import PermanovaConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_feature_df(n_per_batch: int = 60, seed: int = 0) -> pl.DataFrame:
    """Two clearly separated batches with CellProfiler-style feature names."""
    rng = np.random.default_rng(seed)
    batch_a = rng.normal(loc=0.0, scale=0.1, size=(n_per_batch, 3))
    batch_b = rng.normal(loc=5.0, scale=0.1, size=(n_per_batch, 3))
    data = np.vstack([batch_a, batch_b])
    batches = ["batch_a"] * n_per_batch + ["batch_b"] * n_per_batch
    labels = ["WT"] * n_per_batch + ["WT"] * n_per_batch
    return pl.DataFrame(
        {
            "meta_aa_changes": labels,
            META_BATCH_COL: batches,
            "Intensity_mean": data[:, 0].tolist(),
            "Texture_std": data[:, 1].tolist(),
            "Shape_area": data[:, 2].tolist(),
        }
    )


def _make_random_df(n_per_batch: int = 60, seed: int = 1) -> pl.DataFrame:
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
        n_bootstraps=3,
        sample_size=30,
        seed=42,
        parallel=False,
        variant_class_filter=None,
    )
    kwargs.update(overrides)
    return OmegaConf.structured(PermanovaConfig(**kwargs))


# ---------------------------------------------------------------------------
# cosine_dists_matrix
# ---------------------------------------------------------------------------


def test_cosine_dists_matrix_shape() -> None:
    x = np.random.default_rng(0).random((10, 4))
    d = m.cosine_dists_matrix(x)
    assert d.shape == (10, 10)


def test_cosine_dists_matrix_diagonal_is_zero() -> None:
    x = np.random.default_rng(0).random((8, 3))
    d = m.cosine_dists_matrix(x)
    np.testing.assert_array_equal(np.diag(d), 0.0)


def test_cosine_dists_matrix_is_symmetric() -> None:
    x = np.random.default_rng(0).random((8, 3))
    d = m.cosine_dists_matrix(x)
    np.testing.assert_allclose(d, d.T, atol=1e-12)


def test_cosine_dists_matrix_non_negative() -> None:
    x = np.random.default_rng(0).random((10, 4))
    d = m.cosine_dists_matrix(x)
    assert np.all(d >= 0.0)


def test_cosine_dists_matrix_zero_norm_row_no_nan() -> None:
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    d = m.cosine_dists_matrix(x)
    assert not np.any(np.isnan(d))


def test_cosine_dists_matrix_is_exactly_symmetric() -> None:
    x = np.random.default_rng(0).random((12, 5))
    d = m.cosine_dists_matrix(x)
    assert np.array_equal(d, d.T)


def test_cosine_dists_matrix_identical_rows_zero_distance() -> None:
    v = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    d = m.cosine_dists_matrix(v)
    assert d[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_cosine_dists_matrix_orthogonal_rows_distance_one() -> None:
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = m.cosine_dists_matrix(v)
    assert d[0, 1] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# _compute_f_stat
# ---------------------------------------------------------------------------


def test_compute_f_stat_returns_float() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((20, 3))
    labels = np.array(["a"] * 10 + ["b"] * 10)
    dist = m.cosine_dists_matrix(x)
    result = m._compute_f_stat(dist, labels)
    assert isinstance(result, float)


def test_compute_f_stat_separated_groups_larger_than_mixed() -> None:
    rng = np.random.default_rng(42)
    x_sep = np.vstack(
        [
            rng.normal(0.0, 0.1, (20, 3)),
            rng.normal(5.0, 0.1, (20, 3)),
        ]
    )
    x_mix = rng.normal(0.0, 1.0, (40, 3))
    labels = np.array(["a"] * 20 + ["b"] * 20)

    f_sep = m._compute_f_stat(m.cosine_dists_matrix(x_sep), labels)
    f_mix = m._compute_f_stat(m.cosine_dists_matrix(x_mix), labels)
    assert f_sep > f_mix


# ---------------------------------------------------------------------------
# compute_permanova_sample
# ---------------------------------------------------------------------------


def test_compute_permanova_sample_output_columns() -> None:
    df = _make_feature_df()
    result = m.compute_permanova_sample(df.sample(n=30, seed=0, shuffle=True), META_BATCH_COL, seed=0)
    assert set(result.columns) == {"f_value", "f_value_shuffled"}


def test_compute_permanova_sample_single_row() -> None:
    df = _make_feature_df()
    result = m.compute_permanova_sample(df.sample(n=30, seed=0, shuffle=True), META_BATCH_COL, seed=0)
    assert len(result) == 1


def test_compute_permanova_sample_deterministic() -> None:
    df = _make_feature_df()
    sampled = df.sample(n=30, seed=7, shuffle=True)
    r1 = m.compute_permanova_sample(sampled, META_BATCH_COL, seed=7)
    r2 = m.compute_permanova_sample(sampled, META_BATCH_COL, seed=7)
    assert r1["f_value"][0] == pytest.approx(r2["f_value"][0])
    assert r1["f_value_shuffled"][0] == pytest.approx(r2["f_value_shuffled"][0])


def test_compute_permanova_sample_different_seeds_differ() -> None:
    df = _make_feature_df()
    r1 = m.compute_permanova_sample(df.sample(n=30, seed=0, shuffle=True), META_BATCH_COL, seed=0)
    r2 = m.compute_permanova_sample(df.sample(n=30, seed=99, shuffle=True), META_BATCH_COL, seed=99)
    assert r1["f_value"][0] != pytest.approx(r2["f_value"][0])


def test_compute_permanova_sample_separated_batches_high_f() -> None:
    df = _make_feature_df()
    result = m.compute_permanova_sample(df.sample(n=50, seed=0, shuffle=True), META_BATCH_COL, seed=0)
    assert result["f_value"][0] > 10.0


def test_compute_permanova_sample_null_features_no_crash() -> None:
    """Null feature values must be dropped, not crash DistanceMatrix."""
    df = _make_feature_df()
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 5)
        .then(None)
        .otherwise(pl.col("Intensity_mean"))
        .alias("Intensity_mean")
    )
    result = m.compute_permanova_sample(df.sample(n=30, seed=0, shuffle=True), META_BATCH_COL, seed=0)
    assert set(result.columns) == {"f_value", "f_value_shuffled"}


def test_compute_permanova_sample_inf_features_no_crash() -> None:
    """Inf feature values must be dropped, not crash DistanceMatrix."""
    df = _make_feature_df()
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 5)
        .then(float("inf"))
        .otherwise(pl.col("Intensity_mean"))
        .alias("Intensity_mean")
    )
    result = m.compute_permanova_sample(df.sample(n=30, seed=0, shuffle=True), META_BATCH_COL, seed=0)
    assert set(result.columns) == {"f_value", "f_value_shuffled"}


def test_compute_permanova_sample_excludes_meta_columns() -> None:
    """meta_ columns must not be treated as features."""
    df = _make_feature_df().with_columns(pl.lit("extra").alias("meta_extra"))
    result = m.compute_permanova_sample(df.sample(n=30, seed=0, shuffle=True), META_BATCH_COL, seed=0)
    assert result["f_value"][0] is not None


# ---------------------------------------------------------------------------
# bootstrap_permanova
# ---------------------------------------------------------------------------


def test_bootstrap_permanova_row_count() -> None:
    lf = _make_feature_df().lazy()
    result = m.bootstrap_permanova(
        lf,
        META_BATCH_COL,
        n_bootstraps=5,
        sample_size=30,
        seed=42,
        n_jobs=1,
        parallel=False,
    )
    assert len(result) == 5


def test_bootstrap_permanova_columns() -> None:
    lf = _make_feature_df().lazy()
    result = m.bootstrap_permanova(
        lf,
        META_BATCH_COL,
        n_bootstraps=3,
        sample_size=30,
        seed=42,
        n_jobs=1,
        parallel=False,
    )
    assert set(result.columns) == {"f_value", "f_value_shuffled"}


def test_bootstrap_permanova_parallel_matches_sequential() -> None:
    lf = _make_feature_df().lazy()
    seq = m.bootstrap_permanova(
        lf,
        META_BATCH_COL,
        n_bootstraps=3,
        sample_size=30,
        seed=0,
        n_jobs=1,
        parallel=False,
    )
    par = m.bootstrap_permanova(
        lf,
        META_BATCH_COL,
        n_bootstraps=3,
        sample_size=30,
        seed=0,
        n_jobs=2,
        parallel=True,
    )
    np.testing.assert_allclose(
        sorted(seq["f_value"].to_list()),
        sorted(par["f_value"].to_list()),
        rtol=1e-9,
    )


def test_bootstrap_permanova_separated_batches_median_f_high() -> None:
    lf = _make_feature_df(n_per_batch=80).lazy()
    result = m.bootstrap_permanova(
        lf,
        META_BATCH_COL,
        n_bootstraps=5,
        sample_size=50,
        seed=42,
        n_jobs=1,
        parallel=False,
    )
    assert result["f_value"].median() > 5.0


def test_bootstrap_permanova_shuffled_f_lower_than_observed() -> None:
    lf = _make_feature_df(n_per_batch=80).lazy()
    result = m.bootstrap_permanova(
        lf,
        META_BATCH_COL,
        n_bootstraps=5,
        sample_size=50,
        seed=42,
        n_jobs=1,
        parallel=False,
    )
    assert result["f_value"].mean() > result["f_value_shuffled"].mean()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _write_batch_parquets(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for name, loc in [("batch_a", 0.0), ("batch_b", 5.0)]:
        pl.DataFrame(
            {
                "meta_aa_changes": ["WT"] * 40,
                "Intensity_mean": rng.normal(loc, 0.1, 40).tolist(),
                "Texture_std": rng.normal(loc, 0.1, 40).tolist(),
            }
        ).write_parquet(tmp_path / f"{name}.parquet")


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
    assert set(result.columns) == {"f_value", "f_value_shuffled"}


def test_main_output_row_count_matches_n_bootstraps(tmp_path: Path) -> None:
    _write_batch_parquets(tmp_path)
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"), n_bootstraps=4)
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "permanova.parquet")
    assert len(result) == 4


def test_main_output_root_naming(tmp_path: Path) -> None:
    _write_batch_parquets(tmp_path)
    root = str(tmp_path / "run1")
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"), output_root=root)
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "run1.permanova.parquet").exists()


def test_main_variant_class_filter_wt_only(tmp_path: Path) -> None:
    """When filter is set, non-WT rows are excluded before bootstrapping."""
    rng = np.random.default_rng(0)
    for name, loc in [("b1", 0.0), ("b2", 5.0)]:
        pl.DataFrame(
            {
                "meta_aa_changes": ["WT"] * 30 + ["A1B"] * 10,
                "Intensity_mean": rng.normal(loc, 0.1, 40).tolist(),
                "Texture_std": rng.normal(loc, 0.1, 40).tolist(),
            }
        ).write_parquet(tmp_path / f"{name}.parquet")

    cfg = make_perm_cfg(
        tmp_path,
        str(tmp_path / "*.parquet"),
        variant_class_filter="WT",
        sample_size=20,
    )
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "permanova.parquet").exists()
