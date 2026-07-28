from __future__ import annotations

from unittest.mock import patch

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
# main
# ---------------------------------------------------------------------------


def make_corr_cfg(
    tmp_path, half1_file, half2_file, *, label_column="meta_aa_changes"
) -> OmegaConf:
    return OmegaConf.structured(
        m.CorrelateFeaturesConfig(
            output_dir=str(tmp_path / "corr_out"),
            half1_file=str(half1_file),
            half2_file=str(half2_file),
            label_column=label_column,
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
