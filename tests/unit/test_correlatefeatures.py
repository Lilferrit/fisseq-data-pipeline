from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import scipy.stats
from omegaconf import OmegaConf

import fisseq_data_pipeline.correlatefeatures as m

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corr_df_pair() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two aggregate DataFrames with matching label columns and two features."""
    df1 = pl.DataFrame(
        {
            "meta_aa_changes": ["A", "B", "C", "D"],
            "f1": [1.0, 4.0, 2.0, 3.0],
            "f2": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df2 = pl.DataFrame(
        {
            "meta_aa_changes": ["A", "B", "C", "D"],
            "f1": [2.0, 5.0, 1.0, 4.0],
            "f2": [15.0, 25.0, 35.0, 45.0],
        }
    )
    return df1, df2


# ---------------------------------------------------------------------------
# compute_feature_correlations
# ---------------------------------------------------------------------------


def test_compute_feature_correlations_output_columns(
    corr_df_pair: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df1, df2 = corr_df_pair
    result = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    assert set(result.columns) == {"feature", "r", "r_squared"}


def test_compute_feature_correlations_one_row_per_feature(
    corr_df_pair: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df1, df2 = corr_df_pair
    result = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    assert set(result["feature"].to_list()) == {"f1", "f2"}


def test_compute_feature_correlations_label_col_not_in_features(
    corr_df_pair: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df1, df2 = corr_df_pair
    result = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    assert "meta_aa_changes" not in result["feature"].to_list()


def test_compute_feature_correlations_identical_dfs_gives_r_one() -> None:
    df = pl.DataFrame(
        {"meta_aa_changes": ["A", "B", "C", "D"], "f1": [1.0, 2.0, 4.0, 8.0]}
    )
    result = m.compute_feature_correlations(df, df, "meta_aa_changes")
    row = result.filter(pl.col("feature") == "f1").to_dicts().pop()
    assert row["r"] == pytest.approx(1.0)


def test_compute_feature_correlations_r_squared_equals_r_squared(
    corr_df_pair: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df1, df2 = corr_df_pair
    result = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    for row in result.to_dicts():
        assert row["r_squared"] == pytest.approx(row["r"] ** 2)


def test_compute_feature_correlations_matches_scipy(
    corr_df_pair: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df1, df2 = corr_df_pair
    result = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    row = result.filter(pl.col("feature") == "f1").to_dicts().pop()
    expected_r, _ = scipy.stats.pearsonr(df1["f1"].to_numpy(), df2["f1"].to_numpy())
    assert row["r"] == pytest.approx(expected_r)


# ---------------------------------------------------------------------------
# compute_feature_correlations: bootstrap_variant_downsample
# ---------------------------------------------------------------------------


@pytest.fixture
def corr_df_pair_many_variants() -> tuple[pl.DataFrame, pl.DataFrame]:
    """~50 variants with seeded pseudo-random per-variant values in both
    halves, so a downsampled subset's r generically differs from the
    full-set r and different seeds generically pick different subsets."""
    rng = np.random.default_rng(0)
    n = 50
    labels = [f"V{i}" for i in range(n)]
    df1 = pl.DataFrame({"meta_aa_changes": labels, "f1": rng.normal(size=n).tolist()})
    df2 = pl.DataFrame({"meta_aa_changes": labels, "f1": rng.normal(size=n).tolist()})
    return df1, df2


def test_compute_feature_correlations_downsample_none_matches_baseline(
    corr_df_pair: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df1, df2 = corr_df_pair
    baseline = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    explicit_none = m.compute_feature_correlations(
        df1, df2, "meta_aa_changes", bootstrap_variant_downsample=None
    )
    assert baseline.equals(explicit_none)


def test_compute_feature_correlations_downsample_keeps_feature_count_changes_r(
    corr_df_pair_many_variants: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df1, df2 = corr_df_pair_many_variants
    full = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    down = m.compute_feature_correlations(
        df1, df2, "meta_aa_changes", bootstrap_variant_downsample=10, seed=1
    )
    assert set(down["feature"].to_list()) == set(full["feature"].to_list())
    assert down["r"][0] != pytest.approx(full["r"][0])


def test_compute_feature_correlations_downsample_deterministic_same_seed(
    corr_df_pair_many_variants: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df1, df2 = corr_df_pair_many_variants
    r1 = m.compute_feature_correlations(
        df1, df2, "meta_aa_changes", bootstrap_variant_downsample=10, seed=5
    )
    r2 = m.compute_feature_correlations(
        df1, df2, "meta_aa_changes", bootstrap_variant_downsample=10, seed=5
    )
    assert r1.equals(r2)


def test_compute_feature_correlations_downsample_different_seed_can_differ(
    corr_df_pair_many_variants: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df1, df2 = corr_df_pair_many_variants
    r1 = m.compute_feature_correlations(
        df1, df2, "meta_aa_changes", bootstrap_variant_downsample=10, seed=1
    )
    r2 = m.compute_feature_correlations(
        df1, df2, "meta_aa_changes", bootstrap_variant_downsample=10, seed=2
    )
    assert r1["r"][0] != pytest.approx(r2["r"][0])


def test_compute_feature_correlations_downsample_n_greater_than_available_uses_all_and_warns_once(
    corr_df_pair: tuple[pl.DataFrame, pl.DataFrame], caplog: pytest.LogCaptureFixture
) -> None:
    df1, df2 = corr_df_pair
    baseline = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    with caplog.at_level("WARNING"):
        result = m.compute_feature_correlations(
            df1, df2, "meta_aa_changes", bootstrap_variant_downsample=1000, seed=1
        )
    assert result.equals(baseline)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "bootstrap_variant_downsample" in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def make_corr_cfg(
    tmp_path,
    half1_file,
    half2_file,
    *,
    label_column="meta_aa_changes",
    bootstrap_variant_downsample=None,
    bootstrap_idx=0,
    seed=0,
) -> OmegaConf:
    return OmegaConf.structured(
        m.CorrelateFeaturesConfig(
            output_dir=str(tmp_path / "corr_out"),
            half1_file=str(half1_file),
            half2_file=str(half2_file),
            label_column=label_column,
            bootstrap_variant_downsample=bootstrap_variant_downsample,
            bootstrap_idx=bootstrap_idx,
            seed=seed,
        )
    )


def test_main_writes_correlations_file(tmp_path) -> None:
    df1 = pl.DataFrame({"meta_aa_changes": ["A", "B"], "f1_mean": [1.0, 2.0]})
    df2 = pl.DataFrame({"meta_aa_changes": ["A", "B"], "f1_mean": [1.1, 2.1]})
    p1, p2 = tmp_path / "half1.parquet", tmp_path / "half2.parquet"
    df1.write_parquet(p1)
    df2.write_parquet(p2)

    with patch("fisseq_data_pipeline.correlatefeatures.setup_logging"):
        m.main.__wrapped__(make_corr_cfg(tmp_path, p1, p2))

    result = pl.read_parquet(tmp_path / "corr_out" / "correlations.parquet")
    assert set(result.columns) == {"feature", "r", "r_squared"}


def test_main_matches_compute_feature_correlations(tmp_path) -> None:
    df1 = pl.DataFrame({"meta_aa_changes": ["A", "B", "C"], "f1_mean": [1.0, 2.0, 4.0]})
    df2 = pl.DataFrame({"meta_aa_changes": ["A", "B", "C"], "f1_mean": [2.0, 5.0, 1.0]})
    p1, p2 = tmp_path / "half1.parquet", tmp_path / "half2.parquet"
    df1.write_parquet(p1)
    df2.write_parquet(p2)

    with patch("fisseq_data_pipeline.correlatefeatures.setup_logging"):
        m.main.__wrapped__(make_corr_cfg(tmp_path, p1, p2))

    result = pl.read_parquet(tmp_path / "corr_out" / "correlations.parquet")
    expected = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    assert result["r"][0] == pytest.approx(expected["r"][0])


# ---------------------------------------------------------------------------
# main: bootstrap_variant_downsample seed derivation
# ---------------------------------------------------------------------------


def _write_many_variant_halves(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    labels = [f"V{i}" for i in range(n)]
    df1 = pl.DataFrame({"meta_aa_changes": labels, "f1_mean": rng.normal(size=n).tolist()})
    df2 = pl.DataFrame({"meta_aa_changes": labels, "f1_mean": rng.normal(size=n).tolist()})
    p1, p2 = tmp_path / "half1.parquet", tmp_path / "half2.parquet"
    df1.write_parquet(p1)
    df2.write_parquet(p2)
    return p1, p2


def test_main_bootstrap_idx_and_seed_derive_per_replicate_seed_deterministically(
    tmp_path,
) -> None:
    p1, p2 = _write_many_variant_halves(tmp_path)
    with patch("fisseq_data_pipeline.correlatefeatures.setup_logging"):
        m.main.__wrapped__(
            make_corr_cfg(
                tmp_path / "run1",
                p1,
                p2,
                bootstrap_variant_downsample=10,
                bootstrap_idx=3,
                seed=7,
            )
        )
        m.main.__wrapped__(
            make_corr_cfg(
                tmp_path / "run2",
                p1,
                p2,
                bootstrap_variant_downsample=10,
                bootstrap_idx=3,
                seed=7,
            )
        )
    result1 = pl.read_parquet(tmp_path / "run1" / "corr_out" / "correlations.parquet")
    result2 = pl.read_parquet(tmp_path / "run2" / "corr_out" / "correlations.parquet")
    assert result1.sort("feature").equals(result2.sort("feature"))


def test_main_different_bootstrap_idx_changes_sampled_variants_for_same_base_seed(
    tmp_path,
) -> None:
    p1, p2 = _write_many_variant_halves(tmp_path)
    with patch("fisseq_data_pipeline.correlatefeatures.setup_logging"):
        m.main.__wrapped__(
            make_corr_cfg(
                tmp_path / "idx1",
                p1,
                p2,
                bootstrap_variant_downsample=10,
                bootstrap_idx=1,
                seed=7,
            )
        )
        m.main.__wrapped__(
            make_corr_cfg(
                tmp_path / "idx2",
                p1,
                p2,
                bootstrap_variant_downsample=10,
                bootstrap_idx=2,
                seed=7,
            )
        )
    result1 = pl.read_parquet(tmp_path / "idx1" / "corr_out" / "correlations.parquet")
    result2 = pl.read_parquet(tmp_path / "idx2" / "corr_out" / "correlations.parquet")
    assert result1["r"][0] != pytest.approx(result2["r"][0])
