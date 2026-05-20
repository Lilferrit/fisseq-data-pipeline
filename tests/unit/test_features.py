from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import scipy.stats
from omegaconf import OmegaConf

import fisseq_data_pipeline.features as m
from fisseq_data_pipeline.features import TMP_IDX_COL, FeatureSelectConfig
from fisseq_data_pipeline.constants import IMPACT_SCORE_COL

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


@pytest.fixture
def pseudo_rep_df() -> pl.DataFrame:
    """Cell-level dataset with constant feature values per variant.

    Both pseudo-replicates will produce identical aggregates, giving r = 1.0.
    Four cells per group ensures each replicate half gets at least 2 cells for
    the stratified split.
    """
    n = 4
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * n + ["A1A"] * n + ["A1B"] * n + ["A1C"] * n,
            "meta_is_control": [True] * n + [False] * (3 * n),
            "f1": [0.0] * n + [1.0] * n + [5.0] * n + [10.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# get_replicate_lf
# ---------------------------------------------------------------------------


def test_get_replicate_lf_includes_matching_rows() -> None:
    lf = pl.DataFrame({TMP_IDX_COL: [0, 1, 2, 3], "f1": [1.0, 2.0, 3.0, 4.0]}).lazy()
    result = m.get_replicate_lf(lf, [0, 2]).collect()
    assert sorted(result[TMP_IDX_COL].to_list()) == [0, 2]


def test_get_replicate_lf_excludes_non_matching_rows() -> None:
    lf = pl.DataFrame({TMP_IDX_COL: [0, 1, 2, 3], "f1": [1.0, 2.0, 3.0, 4.0]}).lazy()
    result = m.get_replicate_lf(lf, [0, 2]).collect()
    assert 1 not in result[TMP_IDX_COL].to_list()
    assert 3 not in result[TMP_IDX_COL].to_list()


def test_get_replicate_lf_empty_idx_returns_empty() -> None:
    lf = pl.DataFrame({TMP_IDX_COL: [0, 1, 2], "f1": [1.0, 2.0, 3.0]}).lazy()
    result = m.get_replicate_lf(lf, []).collect()
    assert len(result) == 0


def test_get_replicate_lf_preserves_all_columns() -> None:
    lf = pl.DataFrame(
        {TMP_IDX_COL: [0, 1], "f1": [1.0, 2.0], "meta_foo": ["a", "b"]}
    ).lazy()
    result = m.get_replicate_lf(lf, [0]).collect()
    assert set(result.columns) == {TMP_IDX_COL, "f1", "meta_foo"}


# ---------------------------------------------------------------------------
# compute_feature_correlations
# ---------------------------------------------------------------------------


def test_compute_feature_correlations_output_columns(
    corr_df_pair: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    df1, df2 = corr_df_pair
    result = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    assert set(result.columns) == {"feature", "r", "r_squared", "p_value"}


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
    expected_r, expected_p = scipy.stats.pearsonr(
        df1["f1"].to_numpy(), df2["f1"].to_numpy()
    )
    assert row["r"] == pytest.approx(expected_r)
    assert row["p_value"] == pytest.approx(expected_p)


# ---------------------------------------------------------------------------
# pseudo_replicate_correlation
# ---------------------------------------------------------------------------


def test_pseudo_replicate_correlation_output_columns(
    pseudo_rep_df: pl.DataFrame,
) -> None:
    result = m.pseudo_replicate_correlation(
        pseudo_rep_df.lazy(), "meta_aa_changes", "mean", random_state=0
    )
    assert set(result.columns) == {"feature", "r", "r_squared", "p_value"}


def test_pseudo_replicate_correlation_one_row_per_aggregated_feature(
    pseudo_rep_df: pl.DataFrame,
) -> None:
    result = m.pseudo_replicate_correlation(
        pseudo_rep_df.lazy(), "meta_aa_changes", "mean", random_state=0
    )
    assert len(result) == 1
    assert result["feature"][0] == "f1_mean"


def test_pseudo_replicate_correlation_constant_features_give_r_one(
    pseudo_rep_df: pl.DataFrame,
) -> None:
    result = m.pseudo_replicate_correlation(
        pseudo_rep_df.lazy(), "meta_aa_changes", "mean", random_state=0
    )
    assert result["r"][0] == pytest.approx(1.0, abs=1e-6)


def test_pseudo_replicate_correlation_r_squared_equals_r_squared(
    pseudo_rep_df: pl.DataFrame,
) -> None:
    result = m.pseudo_replicate_correlation(
        pseudo_rep_df.lazy(), "meta_aa_changes", "mean", random_state=0
    )
    r = result["r"][0]
    assert result["r_squared"][0] == pytest.approx(r**2)


def test_pseudo_replicate_correlation_random_state_is_deterministic(
    pseudo_rep_df: pl.DataFrame,
) -> None:
    r1 = m.pseudo_replicate_correlation(
        pseudo_rep_df.lazy(), "meta_aa_changes", "mean", random_state=7
    )["r"][0]
    r2 = m.pseudo_replicate_correlation(
        pseudo_rep_df.lazy(), "meta_aa_changes", "mean", random_state=7
    )["r"][0]
    assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# pyc_feature_select
# ---------------------------------------------------------------------------


@pytest.fixture
def agg_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "B", "C"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
        }
    )


def test_pyc_feature_select_returns_polars_dataframe(
    agg_df: pl.DataFrame,
) -> None:
    with patch("pycytominer.feature_select") as mock_fs:
        mock_fs.return_value = agg_df.to_pandas()
        result = m.pyc_feature_select(agg_df)
    assert isinstance(result, pl.DataFrame)


def test_pyc_feature_select_passes_feature_columns(agg_df: pl.DataFrame) -> None:
    with patch("pycytominer.feature_select") as mock_fs:
        mock_fs.return_value = agg_df.to_pandas()
        m.pyc_feature_select(agg_df)
    features_arg = mock_fs.call_args.kwargs["features"]
    assert "f1" in features_arg
    assert "f2" in features_arg
    assert "meta_aa_changes" not in features_arg


def test_pyc_feature_select_passes_correct_operations(agg_df: pl.DataFrame) -> None:
    with patch("pycytominer.feature_select") as mock_fs:
        mock_fs.return_value = agg_df.to_pandas()
        m.pyc_feature_select(agg_df)
    ops = mock_fs.call_args.kwargs["operation"]
    assert "variance_threshold" in ops
    assert "blocklist" in ops
    assert "correlation_threshold" in ops


def test_pyc_feature_select_dropped_features_absent_from_output(
    agg_df: pl.DataFrame,
) -> None:
    with patch("pycytominer.feature_select") as mock_fs:
        mock_fs.return_value = agg_df.drop("f2").to_pandas()
        result = m.pyc_feature_select(agg_df)
    assert "f1" in result.columns
    assert "f2" not in result.columns


def test_pyc_feature_select_meta_columns_preserved(agg_df: pl.DataFrame) -> None:
    with patch("pycytominer.feature_select") as mock_fs:
        mock_fs.return_value = agg_df.to_pandas()
        result = m.pyc_feature_select(agg_df)
    assert "meta_aa_changes" in result.columns


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def make_feat_cfg(
    tmp_path,
    *,
    output_root=None,
    minimum_correlation: float = 0.5,
    aggregator: str = "mean",
    random_state: int = 0,
    compute_impact_score: bool = True,
) -> OmegaConf:
    """Return a DictConfig for FeatureSelectConfig with sensible test defaults."""
    return OmegaConf.structured(
        FeatureSelectConfig(
            output_dir=str(tmp_path),
            output_root=output_root,
            input_file=str(tmp_path / "input.parquet"),
            aggregator=aggregator,
            minimum_correlation=minimum_correlation,
            random_state=random_state,
            compute_impact_score=compute_impact_score,
        )
    )


def write_feat_input_parquet(tmp_path) -> None:
    """Write cell-level test parquet: WT controls + 3 variants, 4 cells each.

    Constant feature values per group mean pseudo-replicates produce r = 1.0.
    """
    n = 4
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * n + ["A1A"] * n + ["A1B"] * n + ["A1C"] * n,
            "meta_is_control": [True] * n + [False] * (3 * n),
            "f1": [0.0] * n + [1.0] * n + [5.0] * n + [10.0] * n,
            "f2": [0.0] * n + [2.0] * n + [6.0] * n + [12.0] * n,
        }
    ).write_parquet(tmp_path / "input.parquet")


# Correlation DataFrame with one high-r and one low-r feature.
_KNOWN_CORR = pl.DataFrame(
    {
        "feature": ["f1_mean", "f2_mean"],
        "r": [0.9, 0.2],
        "r_squared": [0.81, 0.04],
        "p_value": [0.01, 0.8],
    }
)


def test_main_creates_output_file(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    assert (tmp_path / "input.parquet").exists()


def test_main_creates_feature_correlations_file(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    assert (tmp_path / "feature_correlations.parquet").exists()


def test_main_output_contains_label_column(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert "meta_aa_changes" in result.columns


def test_main_output_root_names_output_file(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path, output_root=root))
    assert (tmp_path / "run1.input.parquet").exists()


def test_main_output_root_names_correlations_file(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path, output_root=root))
    assert (tmp_path / "run1.feature_correlations.parquet").exists()


def test_main_feature_correlations_has_feature_ok_column(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    corr = pl.read_parquet(tmp_path / "feature_correlations.parquet")
    assert "feature_ok" in corr.columns


def test_main_high_correlation_feature_marked_ok(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "fisseq_data_pipeline.features.pseudo_replicate_correlation",
            return_value=_KNOWN_CORR,
        ):
            with patch(
                "pycytominer.feature_select",
                side_effect=lambda profiles, **_kw: profiles,
            ):
                m.main.__wrapped__(make_feat_cfg(tmp_path, minimum_correlation=0.5))
    corr = pl.read_parquet(tmp_path / "feature_correlations.parquet")
    row = corr.filter(pl.col("feature") == "f1_mean").to_dicts().pop()
    assert row["feature_ok"] is True


def test_main_low_correlation_feature_marked_not_ok(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "fisseq_data_pipeline.features.pseudo_replicate_correlation",
            return_value=_KNOWN_CORR,
        ):
            with patch(
                "pycytominer.feature_select",
                side_effect=lambda profiles, **_kw: profiles,
            ):
                m.main.__wrapped__(make_feat_cfg(tmp_path, minimum_correlation=0.5))
    corr = pl.read_parquet(tmp_path / "feature_correlations.parquet")
    row = corr.filter(pl.col("feature") == "f2_mean").to_dicts().pop()
    assert row["feature_ok"] is False


def test_main_blocked_feature_absent_from_output(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "fisseq_data_pipeline.features.pseudo_replicate_correlation",
            return_value=_KNOWN_CORR,
        ):
            with patch(
                "pycytominer.feature_select",
                side_effect=lambda profiles, **_kw: profiles,
            ):
                m.main.__wrapped__(make_feat_cfg(tmp_path, minimum_correlation=0.5))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert "f2_mean" not in result.columns


def test_main_unblocked_feature_present_in_output(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "fisseq_data_pipeline.features.pseudo_replicate_correlation",
            return_value=_KNOWN_CORR,
        ):
            with patch(
                "pycytominer.feature_select",
                side_effect=lambda profiles, **_kw: profiles,
            ):
                m.main.__wrapped__(make_feat_cfg(tmp_path, minimum_correlation=0.5))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert "f1_mean" in result.columns


def test_main_pyc_feature_select_called(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch("pycytominer.feature_select") as mock_fs:
            mock_fs.side_effect = lambda profiles, **kw: profiles
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    mock_fs.assert_called_once()


def test_main_pyc_feature_select_dropped_feature_absent(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)

    def drop_f1_mean(profiles, **_kwargs):
        return profiles.drop(columns=["f1_mean"])

    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch("pycytominer.feature_select", side_effect=drop_f1_mean):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert "f1_mean" not in result.columns


# ---------------------------------------------------------------------------
# compute_impact_score — main() integration
# ---------------------------------------------------------------------------


def test_main_impact_score_column_present_by_default(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert IMPACT_SCORE_COL in result.columns


def test_main_impact_score_column_absent_when_disabled(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path, compute_impact_score=False))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert IMPACT_SCORE_COL not in result.columns


def test_main_impact_score_values_are_finite(tmp_path) -> None:
    # A1A is the only synonymous control; its aggregated feature vector is
    # non-zero, so compute_impact_score produces finite scores for all rows.
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert result[IMPACT_SCORE_COL].is_finite().all()


def test_main_synonymous_control_has_zero_impact_score(tmp_path) -> None:
    # After variant_classification the sole synonymous variant (A1A) becomes
    # the control, so its own impact score should be 0.
    write_feat_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "input.parquet")
    ctrl_row = result.filter(pl.col("meta_aa_changes") == "A1A")
    assert ctrl_row[IMPACT_SCORE_COL][0] == pytest.approx(0.0, abs=1e-9)
