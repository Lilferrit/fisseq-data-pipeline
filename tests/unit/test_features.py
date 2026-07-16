from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest
import scipy.stats
from omegaconf import OmegaConf

import fisseq_data_pipeline.features as m
from fisseq_data_pipeline.utils.constants import (
    IMPACT_SCORE_COL,
    META_BARCODE_COL,
    META_BATCH_COL,
)
from fisseq_data_pipeline.utils.splits import TMP_IDX_COL

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
# generate_split_main
# ---------------------------------------------------------------------------


def write_split_input_parquet(tmp_path) -> None:
    """Cell-level parquet with 4 label groups, 4 cells each (16 rows total)."""
    n = 4
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * n + ["A1A"] * n + ["A1B"] * n + ["A1C"] * n,
            "f1": list(range(4 * n)),
        }
    ).write_parquet(tmp_path / "split_input.parquet")


def make_split_cfg(tmp_path, *, random_state: int = 0) -> OmegaConf:
    return OmegaConf.structured(
        m.GenerateSplitConfig(
            output_dir=str(tmp_path / "split_out"),
            input_file=str(tmp_path / "split_input.parquet"),
            random_state=random_state,
        )
    )


def test_generate_split_main_writes_both_halves(tmp_path) -> None:
    write_split_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        m.generate_split_main.__wrapped__(make_split_cfg(tmp_path))
    assert (tmp_path / "split_out" / "half1.parquet").exists()
    assert (tmp_path / "split_out" / "half2.parquet").exists()


def test_generate_split_main_halves_carry_tmp_idx_col(tmp_path) -> None:
    write_split_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        m.generate_split_main.__wrapped__(make_split_cfg(tmp_path))
    half1 = pl.read_parquet(tmp_path / "split_out" / "half1.parquet")
    assert half1.columns == [TMP_IDX_COL]


def test_generate_split_main_halves_disjoint_and_cover_all_rows(tmp_path) -> None:
    write_split_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        m.generate_split_main.__wrapped__(make_split_cfg(tmp_path))
    half1 = set(
        pl.read_parquet(tmp_path / "split_out" / "half1.parquet")[TMP_IDX_COL].to_list()
    )
    half2 = set(
        pl.read_parquet(tmp_path / "split_out" / "half2.parquet")[TMP_IDX_COL].to_list()
    )
    assert half1.isdisjoint(half2)
    assert half1 | half2 == set(range(16))


def test_generate_split_main_random_state_is_deterministic(tmp_path) -> None:
    write_split_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        m.generate_split_main.__wrapped__(make_split_cfg(tmp_path, random_state=7))
    half1_first = sorted(
        pl.read_parquet(tmp_path / "split_out" / "half1.parquet")[TMP_IDX_COL].to_list()
    )

    with patch("fisseq_data_pipeline.features.setup_logging"):
        m.generate_split_main.__wrapped__(make_split_cfg(tmp_path, random_state=7))
    half1_second = sorted(
        pl.read_parquet(tmp_path / "split_out" / "half1.parquet")[TMP_IDX_COL].to_list()
    )

    assert half1_first == half1_second


# ---------------------------------------------------------------------------
# correlate_features_main
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


def test_correlate_features_main_writes_correlations_file(tmp_path) -> None:
    df1 = pl.DataFrame({"meta_aa_changes": ["A", "B"], "f1_mean": [1.0, 2.0]})
    df2 = pl.DataFrame({"meta_aa_changes": ["A", "B"], "f1_mean": [1.1, 2.1]})
    p1, p2 = tmp_path / "half1.parquet", tmp_path / "half2.parquet"
    df1.write_parquet(p1)
    df2.write_parquet(p2)

    with patch("fisseq_data_pipeline.features.setup_logging"):
        m.correlate_features_main.__wrapped__(make_corr_cfg(tmp_path, p1, p2))

    result = pl.read_parquet(tmp_path / "corr_out" / "correlations.parquet")
    assert set(result.columns) == {"feature", "r", "r_squared"}


def test_correlate_features_main_matches_compute_feature_correlations(tmp_path) -> None:
    df1 = pl.DataFrame({"meta_aa_changes": ["A", "B", "C"], "f1_mean": [1.0, 2.0, 4.0]})
    df2 = pl.DataFrame({"meta_aa_changes": ["A", "B", "C"], "f1_mean": [2.0, 5.0, 1.0]})
    p1, p2 = tmp_path / "half1.parquet", tmp_path / "half2.parquet"
    df1.write_parquet(p1)
    df2.write_parquet(p2)

    with patch("fisseq_data_pipeline.features.setup_logging"):
        m.correlate_features_main.__wrapped__(make_corr_cfg(tmp_path, p1, p2))

    result = pl.read_parquet(tmp_path / "corr_out" / "correlations.parquet")
    expected = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    assert result["r"][0] == pytest.approx(expected["r"][0])


# ---------------------------------------------------------------------------
# blocklist_main
# ---------------------------------------------------------------------------


def make_bl_cfg(
    tmp_path, correlation_files, *, minimum_correlation: float = 0.5
) -> OmegaConf:
    return OmegaConf.structured(
        m.BlocklistConfig(
            output_dir=str(tmp_path / "bl_out"),
            correlation_files=correlation_files,
            minimum_correlation=minimum_correlation,
        )
    )


def test_blocklist_main_computes_median_r_across_bootstraps(tmp_path) -> None:
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    for i, r in enumerate([0.9, 0.5, 0.7], start=1):
        pl.DataFrame(
            {"feature": ["f1_mean"], "r": [r], "r_squared": [r**2]}
        ).write_parquet(corr_dir / f"bootstrap_{i}.parquet")

    with patch("fisseq_data_pipeline.features.setup_logging"):
        m.blocklist_main.__wrapped__(make_bl_cfg(tmp_path, str(corr_dir / "*.parquet")))

    result = pl.read_parquet(tmp_path / "bl_out" / "blocklist.parquet")
    row = result.filter(pl.col("feature") == "f1_mean").to_dicts().pop()
    assert row["median_r"] == pytest.approx(0.7)


def test_blocklist_main_feature_ok_thresholding(tmp_path) -> None:
    corr_dir = tmp_path / "corr"
    corr_dir.mkdir()
    pl.DataFrame(
        {
            "feature": ["f1_mean", "f2_mean"],
            "r": [0.9, 0.2],
            "r_squared": [0.81, 0.04],
        }
    ).write_parquet(corr_dir / "bootstrap_1.parquet")

    with patch("fisseq_data_pipeline.features.setup_logging"):
        m.blocklist_main.__wrapped__(
            make_bl_cfg(tmp_path, str(corr_dir / "*.parquet"), minimum_correlation=0.5)
        )

    result = pl.read_parquet(tmp_path / "bl_out" / "blocklist.parquet")
    ok = dict(zip(result["feature"].to_list(), result["feature_ok"].to_list()))
    assert ok["f1_mean"] is True
    assert ok["f2_mean"] is False


def test_blocklist_main_raises_on_empty_glob(tmp_path) -> None:
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with pytest.raises(ValueError):
            m.blocklist_main.__wrapped__(
                make_bl_cfg(tmp_path, str(tmp_path / "nonexistent" / "*.parquet"))
            )


# ---------------------------------------------------------------------------
# combine_blocklists_main
# ---------------------------------------------------------------------------


def make_cb_cfg(tmp_path, blocklist_files) -> OmegaConf:
    return OmegaConf.structured(
        m.CombineBlocklistsConfig(
            output_dir=str(tmp_path / "cb_out"),
            blocklist_files=blocklist_files,
        )
    )


def test_combine_blocklists_main_concatenates_disjoint_features(tmp_path) -> None:
    bl_dir = tmp_path / "bl"
    bl_dir.mkdir()
    pl.DataFrame(
        {"feature": ["f1_mean"], "median_r": [0.9], "feature_ok": [True]}
    ).write_parquet(bl_dir / "mean.parquet")
    pl.DataFrame(
        {"feature": ["f1_std"], "median_r": [0.3], "feature_ok": [False]}
    ).write_parquet(bl_dir / "std.parquet")

    with patch("fisseq_data_pipeline.features.setup_logging"):
        m.combine_blocklists_main.__wrapped__(
            make_cb_cfg(tmp_path, str(bl_dir / "*.parquet"))
        )

    result = pl.read_parquet(tmp_path / "cb_out" / "blocklist.parquet")
    assert set(result["feature"].to_list()) == {"f1_mean", "f1_std"}
    assert len(result) == 2


def test_combine_blocklists_main_raises_on_empty_glob(tmp_path) -> None:
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with pytest.raises(ValueError):
            m.combine_blocklists_main.__wrapped__(
                make_cb_cfg(tmp_path, str(tmp_path / "nonexistent" / "*.parquet"))
            )


# ---------------------------------------------------------------------------
# main() — stage 4, final feature selection
# ---------------------------------------------------------------------------


def write_feat_input_parquet(tmp_path) -> None:
    """Raw cell-level parquet used only for the metadata join in main()."""
    n = 4
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * n + ["A1A"] * n + ["A1B"] * n + ["A1C"] * n,
            "meta_is_control": [True] * n + [False] * (3 * n),
            META_BARCODE_COL: ["bc_0", "bc_1"] * (2 * n),
        }
    ).write_parquet(tmp_path / "input.parquet")


def write_feature_type_aggregate(tmp_path) -> None:
    """Per-feature-type aggregate fixture matching write_feat_input_parquet's
    cell-level means exactly (mean aggregator over f1=[0]*4,[1]*4,[5]*4,[10]*4
    and f2=[0]*4,[2]*4,[6]*4,[12]*4 for WT/A1A/A1B/A1C respectively)."""
    ft_dir = tmp_path / "ft"
    ft_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A1B", "A1C"],
            "f1_mean": [1.0, 5.0, 10.0],
            "f2_mean": [2.0, 6.0, 12.0],
        }
    ).write_parquet(ft_dir / "mean.parquet")


def write_blocklist(tmp_path, *, block_f2: bool = False) -> None:
    pl.DataFrame(
        {"feature": ["f1_mean", "f2_mean"], "feature_ok": [True, not block_f2]}
    ).write_parquet(tmp_path / "blocklist.parquet")


def make_feat_cfg(
    tmp_path,
    *,
    output_root=None,
    feature_type_files=None,
    block_list_file=None,
    compute_impact_score: bool = True,
) -> OmegaConf:
    """Return a DictConfig for FinalizeFeatureSelectConfig with test defaults."""
    if feature_type_files is None:
        feature_type_files = str(tmp_path / "ft" / "*.parquet")
    if block_list_file is None:
        block_list_file = str(tmp_path / "blocklist.parquet")
    return OmegaConf.structured(
        m.FinalizeFeatureSelectConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            input_file=str(tmp_path / "input.parquet"),
            feature_type_files=feature_type_files,
            block_list_file=block_list_file,
            compute_impact_score=compute_impact_score,
        )
    )


def _write_default_fixtures(tmp_path, *, block_f2: bool = False) -> None:
    write_feat_input_parquet(tmp_path)
    write_feature_type_aggregate(tmp_path)
    write_blocklist(tmp_path, block_f2=block_f2)


def test_main_creates_output_file(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    assert (tmp_path / "out" / "input.parquet").exists()


def test_main_output_contains_label_column(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "meta_aa_changes" in result.columns


def test_main_output_root_names_output_file(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path, output_root=root))
    assert (tmp_path / "run1.input.parquet").exists()


def test_main_blocked_feature_absent_from_output(tmp_path) -> None:
    _write_default_fixtures(tmp_path, block_f2=True)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "f2_mean" not in result.columns


def test_main_unblocked_feature_present_in_output(tmp_path) -> None:
    _write_default_fixtures(tmp_path, block_f2=True)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "f1_mean" in result.columns


def test_main_joins_multiple_feature_type_files(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    write_feature_type_aggregate(tmp_path)
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A1B", "A1C"],
            "f1_std": [0.0, 0.0, 0.0],
        }
    ).write_parquet(tmp_path / "ft" / "std.parquet")
    write_blocklist(tmp_path)
    pl.DataFrame(
        {"feature": ["f1_mean", "f2_mean", "f1_std"], "feature_ok": [True, True, True]}
    ).write_parquet(tmp_path / "blocklist.parquet")

    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert {"f1_mean", "f2_mean", "f1_std"}.issubset(set(result.columns))


def test_main_raises_on_empty_feature_type_glob(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(
                make_feat_cfg(
                    tmp_path,
                    feature_type_files=str(tmp_path / "nonexistent" / "*.parquet"),
                )
            )


def test_main_pyc_feature_select_called(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch("pycytominer.feature_select") as mock_fs:
            mock_fs.side_effect = lambda profiles, **kw: profiles
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    mock_fs.assert_called_once()


def test_main_pyc_feature_select_dropped_feature_absent(tmp_path) -> None:
    _write_default_fixtures(tmp_path)

    def drop_f1_mean(profiles, **_kwargs):
        return profiles.drop(columns=["f1_mean"])

    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch("pycytominer.feature_select", side_effect=drop_f1_mean):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "f1_mean" not in result.columns


# ---------------------------------------------------------------------------
# compute_impact_score — main() integration
# ---------------------------------------------------------------------------


def test_main_impact_score_column_present_by_default(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert IMPACT_SCORE_COL in result.columns


def test_main_impact_score_column_absent_when_disabled(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path, compute_impact_score=False))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert IMPACT_SCORE_COL not in result.columns


def test_main_impact_score_values_are_finite(tmp_path) -> None:
    # A1A is the only synonymous control; its aggregated feature vector is
    # non-zero, so compute_impact_score produces finite scores for all rows.
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert result[IMPACT_SCORE_COL].is_finite().all()


def test_main_synonymous_control_has_zero_impact_score(tmp_path) -> None:
    # After variant_classification the sole synonymous variant (A1A) becomes
    # the control, so its own impact score should be 0.
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    ctrl_row = result.filter(pl.col("meta_aa_changes") == "A1A")
    assert ctrl_row[IMPACT_SCORE_COL][0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# aggregate meta data — main() integration
# ---------------------------------------------------------------------------


def _run_main(tmp_path, **kwargs) -> pl.DataFrame:
    """Run main() and return the output parquet."""
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.features.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path, **kwargs))
    return pl.read_parquet(tmp_path / "out" / "input.parquet")


def test_main_output_contains_meta_num_cells(tmp_path) -> None:
    result = _run_main(tmp_path)
    assert "meta_num_cells" in result.columns


def test_main_meta_num_cells_correct(tmp_path) -> None:
    result = _run_main(tmp_path)
    assert (result["meta_num_cells"] == 4).all()


def test_main_output_contains_barcode_num_unique(tmp_path) -> None:
    result = _run_main(tmp_path)
    assert f"{META_BARCODE_COL}_num_unique" in result.columns


def test_main_meta_barcode_num_unique_correct(tmp_path) -> None:
    # write_feat_input_parquet alternates bc_0 / bc_1 → 2 unique per variant
    result = _run_main(tmp_path)
    assert (result[f"{META_BARCODE_COL}_num_unique"] == 2).all()


def test_main_output_contains_batch_num_unique(tmp_path) -> None:
    # meta_batch is added by load_batches from the filename stem
    result = _run_main(tmp_path)
    assert f"{META_BATCH_COL}_num_unique" in result.columns


def test_main_meta_batch_num_unique_is_one_for_single_file(tmp_path) -> None:
    # single input file → all cells share the same batch label
    result = _run_main(tmp_path)
    assert (result[f"{META_BATCH_COL}_num_unique"] == 1).all()
