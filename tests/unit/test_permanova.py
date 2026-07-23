from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.permanova as m
from fisseq_data_pipeline.permanova import PermanovaConfig
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
        result = m.compute_feature_permanova(
            df.lazy(), FEATURE_COLS[0], META_BATCH_COL, n_permutations=0, seed=0
        )
        return result["f_statistic"]

    assert _f_for(df_sep) > _f_for(df_mix)


# ---------------------------------------------------------------------------
# _pairwise_abs_distance
# ---------------------------------------------------------------------------


def test_pairwise_abs_distance_hand_computed() -> None:
    x = np.array([1.0, 4.0, -2.0])
    result = m._pairwise_abs_distance(x)
    expected = np.array(
        [
            [0.0, 3.0, 3.0],
            [3.0, 0.0, 6.0],
            [3.0, 6.0, 0.0],
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_pairwise_abs_distance_symmetric_and_zero_diagonal() -> None:
    x = np.array([2.0, 5.0, -1.0, 0.0])
    result = m._pairwise_abs_distance(x)
    assert np.allclose(np.diag(result), 0.0)
    assert np.allclose(result, result.T)


def test_pairwise_abs_distance_nonfinite_value_propagates() -> None:
    x = np.array([1.0, 2.0, np.nan])
    result = m._pairwise_abs_distance(x)
    assert np.isnan(result[0, 2])
    assert np.isnan(result[1, 2])
    assert np.isnan(result[2, 0])
    assert np.isnan(result[2, 1])
    assert result[0, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_feature_permanova
# ---------------------------------------------------------------------------


def test_compute_feature_permanova_output_keys() -> None:
    df = _make_feature_df()
    result = m.compute_feature_permanova(
        df.lazy(), FEATURE_COLS[0], META_BATCH_COL, n_permutations=10, seed=0
    )
    assert set(result.keys()) == {"f_statistic", "p_value"}


def test_compute_feature_permanova_no_permutations_p_value_none() -> None:
    df = _make_feature_df()
    result = m.compute_feature_permanova(
        df.lazy(), FEATURE_COLS[0], META_BATCH_COL, n_permutations=0, seed=0
    )
    assert result["p_value"] is None


def test_compute_feature_permanova_p_value_in_range() -> None:
    df = _make_random_df()
    result = m.compute_feature_permanova(
        df.lazy(), FEATURE_COLS[0], META_BATCH_COL, n_permutations=50, seed=0
    )
    assert 0.0 < result["p_value"] <= 1.0


def test_compute_feature_permanova_separated_batches_significant() -> None:
    df = _make_feature_df()
    result = m.compute_feature_permanova(
        df.lazy(), FEATURE_COLS[0], META_BATCH_COL, n_permutations=99, seed=0
    )
    assert result["f_statistic"] > 10.0
    assert result["p_value"] < 0.05


def test_compute_feature_permanova_single_batch_returns_none() -> None:
    df = _make_feature_df().filter(pl.col(META_BATCH_COL) == "batch_a")
    result = m.compute_feature_permanova(
        df.lazy(), FEATURE_COLS[0], META_BATCH_COL, n_permutations=10, seed=0
    )
    assert result is None


def test_compute_feature_permanova_too_few_samples_returns_none() -> None:
    df = _make_feature_df().head(1)
    result = m.compute_feature_permanova(
        df.lazy(), FEATURE_COLS[0], META_BATCH_COL, n_permutations=10, seed=0
    )
    assert result is None


def test_compute_feature_permanova_non_finite_values_propagate_to_nan_f_statistic() -> (
    None
):
    """With a single feature dimension there is no other dimension to fall
    back on, so a non-finite value in the tested feature makes every pair
    involving that sample non-finite, and that propagates through to the
    F-statistic (unlike the old per-pair-masked cosine-distance path)."""
    df = _make_feature_df()
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 3)
        .then(float("inf"))
        .otherwise(pl.col("Intensity_mean"))
        .alias("Intensity_mean")
    )
    result = m.compute_feature_permanova(
        df.lazy(), "Intensity_mean", META_BATCH_COL, n_permutations=10, seed=0
    )
    assert result is not None
    assert not np.isfinite(result["f_statistic"])


def test_compute_feature_permanova_uses_cross_batch_pairs() -> None:
    """If the self-join only paired same-batch samples, F would be undefined
    (SS_between would need cross-batch information to be meaningful)."""
    df = _make_feature_df()
    result = m.compute_feature_permanova(
        df.lazy(), FEATURE_COLS[0], META_BATCH_COL, n_permutations=0, seed=0
    )
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


def _write_permanova_batch_parquets(tmp_path: Path) -> None:
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
    _write_permanova_batch_parquets(tmp_path)
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "permanova.parquet").exists()


def test_main_output_has_correct_columns(tmp_path: Path) -> None:
    _write_permanova_batch_parquets(tmp_path)
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "permanova.parquet")
    assert set(result.columns) == {"feature", "f_value", "p_value"}


def test_main_output_row_count_one_per_feature(tmp_path: Path) -> None:
    _write_permanova_batch_parquets(tmp_path)
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "permanova.parquet")
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

    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"))
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "permanova.parquet")
    assert len(result) == 0


def test_main_output_root_naming(tmp_path: Path) -> None:
    _write_permanova_batch_parquets(tmp_path)
    root = str(tmp_path / "run1")
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"), output_root=root)
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    assert (tmp_path / "run1.permanova.parquet").exists()


def test_main_n_permutations_zero_gives_null_p_value(tmp_path: Path) -> None:
    _write_permanova_batch_parquets(tmp_path)
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"), n_permutations=0)
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "permanova.parquet")
    assert result["p_value"][0] is None


# ---------------------------------------------------------------------------
# main() -- feature_batch_size
# ---------------------------------------------------------------------------


def _write_permanova_batch_parquets_wide(
    tmp_path: Path, n_features: int = 5
) -> list[str]:
    """Like _write_permanova_batch_parquets but with n_features feature
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
    _write_permanova_batch_parquets_wide(tmp_path, n_features=5)
    input_glob = str(tmp_path / "*.parquet")

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    cfg = make_perm_cfg(baseline_dir, input_glob)
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    baseline = pl.read_parquet(baseline_dir / "permanova.parquet").sort("feature")

    for size in [1, 2, 3, 5, 100]:
        out_dir = tmp_path / f"batched_{size}"
        out_dir.mkdir()
        cfg = make_perm_cfg(out_dir, input_glob, feature_batch_size=size)
        with patch("fisseq_data_pipeline.permanova.setup_logging"):
            m.main.__wrapped__(cfg)
        batched = pl.read_parquet(out_dir / "permanova.parquet").sort("feature")
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
    feature_names = _write_permanova_batch_parquets_wide(tmp_path, n_features=5)
    cfg = make_perm_cfg(tmp_path, str(tmp_path / "*.parquet"), feature_batch_size=2)
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        m.main.__wrapped__(cfg)
    result = pl.read_parquet(tmp_path / "permanova.parquet")
    assert len(result) == 5
    assert set(result["feature"]) == set(feature_names)


@pytest.mark.parametrize("bad_size", [0, -1])
def test_main_feature_batch_size_non_positive_raises(
    tmp_path: Path, bad_size: int
) -> None:
    _write_permanova_batch_parquets(tmp_path)
    cfg = make_perm_cfg(
        tmp_path, str(tmp_path / "*.parquet"), feature_batch_size=bad_size
    )
    with patch("fisseq_data_pipeline.permanova.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(cfg)
