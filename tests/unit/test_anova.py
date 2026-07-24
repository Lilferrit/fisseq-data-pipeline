from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import scipy.stats
from omegaconf import OmegaConf

import fisseq_data_pipeline.anova as m
from fisseq_data_pipeline.anova import AnovaConfig
from fisseq_data_pipeline.utils.constants import META_BATCH_COL

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
    labels = ["A1A"] * (n_per_batch * 2)
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
            "meta_aa_changes": ["A1A"] * (n_per_batch * 2),
            META_BATCH_COL: batches,
            "Intensity_mean": data[:, 0].tolist(),
            "Texture_std": data[:, 1].tolist(),
            "Shape_area": data[:, 2].tolist(),
        }
    )


def make_anova_cfg(tmp_path: Path, input_file: str, **overrides) -> OmegaConf:
    kwargs = dict(
        output_dir=str(tmp_path),
        input_file=input_file,
    )
    kwargs.update(overrides)
    return OmegaConf.structured(AnovaConfig(**kwargs))


FEATURE_COLS = ["Intensity_mean", "Texture_std", "Shape_area"]


# ---------------------------------------------------------------------------
# _f_statistic
# ---------------------------------------------------------------------------


def test_f_statistic_hand_computed() -> None:
    # Group 0: values [1, 3] (n=2). Group 1: values [10, 12, 14] (n=3).
    sum_g = np.array([4.0, 36.0])
    sumsq_g = np.array([10.0, 440.0])
    n_g = np.array([2, 3])
    n, a = 5, 2

    f = m._f_statistic(sum_g, sumsq_g, n_g, n, a)

    ss_within = (sumsq_g - sum_g**2 / n_g).sum()
    ss_total = sumsq_g.sum() - sum_g.sum() ** 2 / n
    ss_between = ss_total - ss_within
    expected = (ss_between / (a - 1)) / (ss_within / (n - a))
    assert f == pytest.approx(expected)
    assert f == pytest.approx(36.0)


def test_f_statistic_separated_groups_larger_than_mixed() -> None:
    df_sep = _make_feature_df()
    df_mix = _make_random_df()

    def _f_for(df: pl.DataFrame) -> float:
        result = m.compute_feature_anova(df.lazy(), FEATURE_COLS[0], META_BATCH_COL)
        return result["f_statistic"]

    assert _f_for(df_sep) > _f_for(df_mix)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_f_statistic_matches_old_pairwise_distance_computation(seed: int) -> None:
    """Regression test: the sufficient-statistics F-statistic this module now
    computes must be algebraically identical to the OLD O(n^2) pairwise
    absolute-distance Anderson (2001) sum-of-squares decomposition it
    replaces -- not merely correlated with it. The old computation is
    reimplemented inline here since _pairwise_abs_distance no longer exists
    in fisseq_data_pipeline.anova."""
    rng = np.random.default_rng(seed)
    n_groups = int(rng.integers(2, 5))
    group_sizes = rng.integers(2, 6, size=n_groups)
    batch_labels = np.concatenate(
        [np.full(size, f"batch_{i}") for i, size in enumerate(group_sizes)]
    )
    group_locs = np.repeat(rng.uniform(-5, 5, size=n_groups), group_sizes)
    x = rng.normal(loc=group_locs, scale=1.0)
    n = x.shape[0]

    # --- OLD algorithm: pairwise absolute distance, Anderson (2001) SS decomposition ---
    _, group_of_sample, group_sizes_arr = np.unique(
        batch_labels, return_inverse=True, return_counts=True
    )
    a = group_sizes_arr.shape[0]
    abs_dist = np.abs(x[:, None] - x[None, :])
    idx_a, idx_b = np.triu_indices(n, k=1)
    d2 = abs_dist[idx_a, idx_b] ** 2
    ss_total_old = d2.sum() / n
    same = group_of_sample[idx_a] == group_of_sample[idx_b]
    weights = 1.0 / group_sizes_arr[group_of_sample[idx_a]]
    ss_within_old = (d2[same] * weights[same]).sum()
    ss_between_old = ss_total_old - ss_within_old
    f_old = (ss_between_old / (a - 1)) / (ss_within_old / (n - a))

    # --- NEW algorithm: per-group sufficient statistics ---
    result = m._compute_anova_stats(x, batch_labels)

    assert result["f_statistic"] == pytest.approx(f_old, rel=1e-9)


# ---------------------------------------------------------------------------
# _compute_anova_stats
# ---------------------------------------------------------------------------


def test_compute_anova_stats_p_value_closed_form() -> None:
    """The F-statistic and p-value must match an independent, dead-simple
    derivation via per-group means/sums-of-squares and a direct call to
    scipy.stats.f.sf with explicit degrees of freedom -- this also pins
    down that dfn=a-1 and dfd=n-a are wired correctly."""
    x = np.array([0.0, 1.0, -1.0, 0.0, 5.0, 6.0, 4.0, 5.0, 10.0, 11.0, 9.0, 10.0])
    batch_labels = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    n, a = 12, 3

    result = m._compute_anova_stats(x, batch_labels)

    groups = [x[batch_labels == g] for g in ("a", "b", "c")]
    grand_mean = x.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    f_expected = (ss_between / (a - 1)) / (ss_within / (n - a))
    assert result["f_statistic"] == pytest.approx(f_expected)

    p_expected = scipy.stats.f.sf(f_expected, dfn=a - 1, dfd=n - a)
    assert result["p_value"] == pytest.approx(p_expected)


# ---------------------------------------------------------------------------
# compute_feature_anova
# ---------------------------------------------------------------------------


def test_compute_feature_anova_output_keys() -> None:
    df = _make_feature_df()
    result = m.compute_feature_anova(df.lazy(), FEATURE_COLS[0], META_BATCH_COL)
    assert set(result.keys()) == {"f_statistic", "p_value"}


def test_compute_feature_anova_p_value_in_range() -> None:
    df = _make_random_df()
    result = m.compute_feature_anova(df.lazy(), FEATURE_COLS[0], META_BATCH_COL)
    # Unlike the old permutation-based p-value (always > 0), the closed-form
    # p-value can legitimately land exactly on 0.0 or 1.0.
    assert 0.0 <= result["p_value"] <= 1.0


def test_compute_feature_anova_separated_batches_significant() -> None:
    df = _make_feature_df()
    result = m.compute_feature_anova(df.lazy(), FEATURE_COLS[0], META_BATCH_COL)
    assert result["f_statistic"] > 10.0
    assert result["p_value"] < 0.05


def test_compute_feature_anova_single_batch_returns_none() -> None:
    df = _make_feature_df().filter(pl.col(META_BATCH_COL) == "batch_a")
    result = m.compute_feature_anova(df.lazy(), FEATURE_COLS[0], META_BATCH_COL)
    assert result is None


def test_compute_feature_anova_too_few_samples_returns_none() -> None:
    df = _make_feature_df().head(1)
    result = m.compute_feature_anova(df.lazy(), FEATURE_COLS[0], META_BATCH_COL)
    assert result is None


def test_compute_feature_anova_non_finite_values_propagate_to_nan_f_statistic() -> None:
    """A non-finite value in the tested feature makes that sample's group sum
    and sum-of-squares non-finite, which propagates through the
    sufficient-statistics decomposition to a non-finite F-statistic and, via
    scipy.stats.f.sf, a non-finite p-value."""
    df = _make_feature_df()
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 3)
        .then(float("inf"))
        .otherwise(pl.col("Intensity_mean"))
        .alias("Intensity_mean")
    )
    result = m.compute_feature_anova(df.lazy(), "Intensity_mean", META_BATCH_COL)
    assert result is not None
    assert not np.isfinite(result["f_statistic"])
    assert not np.isfinite(result["p_value"])


def test_compute_feature_anova_between_group_variation_detected() -> None:
    """Sanity check that the F-statistic reflects between-group variation
    (not just within-group noise) -- i.e. the between/within decomposition
    is wired correctly end to end."""
    df = _make_feature_df()
    result = m.compute_feature_anova(df.lazy(), FEATURE_COLS[0], META_BATCH_COL)
    assert np.isfinite(result["f_statistic"])
    assert result["f_statistic"] > 0


# ---------------------------------------------------------------------------
# _iter_feature_batches
# ---------------------------------------------------------------------------


def _make_batches_lf(
    n_features: int = 5, n_rows: int = 3
) -> tuple[pl.LazyFrame, list[str]]:
    """Each row i's value for feature f{k} is i*10+k, and its batch label is
    "batch_{i}" -- both uniquely decodable from a row's data, so alignment
    between batch_col and a feature column can be verified per-row."""
    feature_names = [f"f{k}" for k in range(n_features)]
    data = {META_BATCH_COL: [f"batch_{i}" for i in range(n_rows)]}
    for k, name in enumerate(feature_names):
        data[name] = [float(i * 10 + k) for i in range(n_rows)]
    return pl.DataFrame(data).lazy(), feature_names


def test_iter_feature_batches_none_yields_single_chunk_all_features() -> None:
    lf, feature_names = _make_batches_lf(n_features=3)
    chunks = list(m._iter_feature_batches(lf, feature_names, META_BATCH_COL, None))
    assert len(chunks) == 1
    collected, chunk = chunks[0]
    assert chunk == feature_names
    assert len(collected) == 3
    assert set(collected.columns) == {META_BATCH_COL, *feature_names}


def test_iter_feature_batches_splits_into_chunks_in_order() -> None:
    lf, feature_names = _make_batches_lf(n_features=5)
    chunks = list(m._iter_feature_batches(lf, feature_names, META_BATCH_COL, 2))
    assert [len(chunk) for _, chunk in chunks] == [2, 2, 1]
    assert [f for _, chunk in chunks for f in chunk] == feature_names


def test_iter_feature_batches_batch_col_aligned_with_chunk_features() -> None:
    lf, feature_names = _make_batches_lf(n_features=3, n_rows=4)
    for collected, chunk in m._iter_feature_batches(
        lf, feature_names, META_BATCH_COL, 1
    ):
        feature_col = chunk[0]
        k = int(feature_col[1:])
        for row in collected.iter_rows(named=True):
            expected_i = (int(row[feature_col]) - k) // 10
            assert row[META_BATCH_COL] == f"batch_{expected_i}"


@pytest.mark.parametrize("bad_size", [0, -1, -5])
def test_iter_feature_batches_non_positive_size_raises(bad_size: int) -> None:
    lf, feature_names = _make_batches_lf(n_features=3)
    with pytest.raises(ValueError):
        list(m._iter_feature_batches(lf, feature_names, META_BATCH_COL, bad_size))


@pytest.mark.parametrize("feature_batch_size", [None, 2])
def test_iter_feature_batches_empty_feature_cols_yields_nothing(
    feature_batch_size,
) -> None:
    lf, _ = _make_batches_lf(n_features=0)
    assert (
        list(m._iter_feature_batches(lf, [], META_BATCH_COL, feature_batch_size)) == []
    )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _write_anova_batch_parquets(tmp_path: Path) -> None:
    """Two batches, each with rows across three label classes: Synonymous
    ("A1A", included), Single Missense ("A1B", excluded), and
    Synonymous-but-downsampled ("A1A:downsampled", excluded)."""
    rng = np.random.default_rng(0)
    for name, loc in [("batch_a", 0.0), ("batch_b", 5.0)]:
        n = 20
        labels = ["A1A"] * n + ["A1B"] * n + ["A1A:downsampled"] * n
        total = len(labels)
        data = {
            "meta_aa_changes": labels,
            "Intensity_mean": rng.normal(loc, 0.1, total).tolist(),
            "Texture_std": rng.normal(loc, 0.1, total).tolist(),
        }
        pl.DataFrame(data).write_parquet(tmp_path / f"{name}.parquet")


def test_main_creates_output_file(tmp_path: Path) -> None:
    _write_anova_batch_parquets(tmp_path)
    cfg = make_anova_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.anova.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "anova.parquet").exists()


def test_main_output_has_correct_columns(tmp_path: Path) -> None:
    _write_anova_batch_parquets(tmp_path)
    cfg = make_anova_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.anova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "anova.parquet")
    assert set(result.columns) == {"feature", "f_value", "p_value"}


def test_main_output_row_count_one_per_feature(tmp_path: Path) -> None:
    _write_anova_batch_parquets(tmp_path)
    cfg = make_anova_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.anova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "anova.parquet")
    # Two feature columns (Intensity_mean, Texture_std), both qualifying
    # since "A1A" appears in both batches.
    assert len(result) == 2
    assert set(result["feature"]) == {"Intensity_mean", "Texture_std"}


def test_main_excludes_features_seen_in_only_one_batch(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    # All Synonymous, non-downsampled rows land in batch_a only; batch_b's
    # rows are all non-synonymous, so after filtering every feature's
    # surviving rows come from a single batch.
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A"] * 20,
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

    cfg = make_anova_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.anova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "anova.parquet")
    assert len(result) == 0


def test_main_output_root_naming(tmp_path: Path) -> None:
    _write_anova_batch_parquets(tmp_path)
    root = str(tmp_path / "run1")
    cfg = make_anova_cfg(tmp_path, str(tmp_path / "*.parquet"), output_root=root)
    with patch("fisseq_data_pipeline.anova.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "run1.anova.parquet").exists()


# ---------------------------------------------------------------------------
# main() -- feature_batch_size
# ---------------------------------------------------------------------------


def _write_anova_batch_parquets_wide(tmp_path: Path, n_features: int = 5) -> list[str]:
    """Like _write_anova_batch_parquets but with n_features feature
    columns (feat_0..feat_{n-1}), for exercising feature_batch_size chunking
    that does not evenly divide the feature count."""
    rng = np.random.default_rng(0)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    for name, loc in [("batch_a", 0.0), ("batch_b", 5.0)]:
        n = 20
        labels = ["A1A"] * n + ["A1B"] * n + ["A1A:downsampled"] * n
        total = len(labels)
        data = {"meta_aa_changes": labels}
        for feat in feature_names:
            data[feat] = rng.normal(loc, 0.1, total).tolist()
        pl.DataFrame(data).write_parquet(tmp_path / f"{name}.parquet")
    return feature_names


def test_main_feature_batch_size_matches_unbatched_output(tmp_path: Path) -> None:
    _write_anova_batch_parquets_wide(tmp_path, n_features=5)
    input_glob = str(tmp_path / "*.parquet")

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    cfg = make_anova_cfg(baseline_dir, input_glob)
    with patch("fisseq_data_pipeline.anova.setup_logging"):
        m.main.__wrapped__(cfg)
    baseline = pl.read_parquet(baseline_dir / "anova.parquet").sort("feature")

    for size in [1, 2, 3, 5, 100]:
        out_dir = tmp_path / f"batched_{size}"
        out_dir.mkdir()
        cfg = make_anova_cfg(out_dir, input_glob, feature_batch_size=size)
        with patch("fisseq_data_pipeline.anova.setup_logging"):
            m.main.__wrapped__(cfg)
        batched = pl.read_parquet(out_dir / "anova.parquet").sort("feature")
        assert batched["feature"].to_list() == baseline["feature"].to_list()
        np.testing.assert_allclose(
            batched["f_value"].to_numpy(), baseline["f_value"].to_numpy()
        )
        np.testing.assert_allclose(
            batched["p_value"].to_numpy(), baseline["p_value"].to_numpy()
        )


def test_main_feature_batch_size_smaller_than_feature_count_no_features_dropped(
    tmp_path: Path,
) -> None:
    feature_names = _write_anova_batch_parquets_wide(tmp_path, n_features=5)
    cfg = make_anova_cfg(tmp_path, str(tmp_path / "*.parquet"), feature_batch_size=2)
    with patch("fisseq_data_pipeline.anova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "anova.parquet")
    assert len(result) == 5
    assert set(result["feature"]) == set(feature_names)


@pytest.mark.parametrize("bad_size", [0, -1])
def test_main_feature_batch_size_non_positive_raises(
    tmp_path: Path, bad_size: int
) -> None:
    _write_anova_batch_parquets(tmp_path)
    cfg = make_anova_cfg(
        tmp_path, str(tmp_path / "*.parquet"), feature_batch_size=bad_size
    )
    with patch("fisseq_data_pipeline.anova.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(cfg)
