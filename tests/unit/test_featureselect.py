from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.featureselect as m
from fisseq_data_pipeline.utils.constants import (
    IMPACT_SCORE_COL,
    META_BARCODE_COL,
    META_BATCH_COL,
)

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
# main() — stage 4, final feature selection
# ---------------------------------------------------------------------------


def write_feat_input_parquet(tmp_path) -> None:
    """Raw cell-level parquet used only for the metadata join in main()."""
    n = 4
    labels = (
        ["WT"] * n + ["A1A"] * n + ["A2A"] * n + ["A3A"] * n + ["A1B"] * n + ["A1C"] * n
    )
    pl.DataFrame(
        {
            "meta_aa_changes": labels,
            "meta_is_control": [label == "WT" for label in labels],
            META_BARCODE_COL: ["bc_0", "bc_1"] * (len(labels) // 2),
        }
    ).write_parquet(tmp_path / "input.parquet")


def write_feature_type_aggregate(tmp_path) -> None:
    """Per-feature-type aggregate fixture matching write_feat_input_parquet's
    cell-level means exactly. A1A, A2A, and A3A are all Synonymous
    (classify_variant), so together they form the synonymous-baseline
    normalization reference; their non-uniformly-spaced values keep the
    control group's median (used by compute_impact_score) non-degenerate."""
    ft_dir = tmp_path / "ft"
    ft_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
            "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
            "f2_mean": [0.0, 2.0, 8.0, 6.0, 12.0],
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
    run_pca: bool = False,
    pca_n_components: int = 2,
    run_umap: bool = False,
    umap_n_components: int = 2,
    umap_n_neighbors: int = 2,
    umap_metric: str = "cosine",
    umap_min_dist: float = 0.1,
    umap_random_state=42,
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
            run_pca=run_pca,
            pca_n_components=pca_n_components,
            run_umap=run_umap,
            umap_n_components=umap_n_components,
            umap_n_neighbors=umap_n_neighbors,
            umap_metric=umap_metric,
            umap_min_dist=umap_min_dist,
            umap_random_state=umap_random_state,
        )
    )


def _write_default_fixtures(tmp_path, *, block_f2: bool = False) -> None:
    write_feat_input_parquet(tmp_path)
    write_feature_type_aggregate(tmp_path)
    write_blocklist(tmp_path, block_f2=block_f2)


def test_main_creates_output_file(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    assert (tmp_path / "out" / "input.parquet").exists()


def test_main_output_contains_label_column(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "meta_aa_changes" in result.columns


def test_main_output_root_names_output_file(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path, output_root=root))
    assert (tmp_path / "run1.input.parquet").exists()


def test_main_blocked_feature_absent_from_output(tmp_path) -> None:
    _write_default_fixtures(tmp_path, block_f2=True)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "f2_mean" not in result.columns


def test_main_unblocked_feature_present_in_output(tmp_path) -> None:
    _write_default_fixtures(tmp_path, block_f2=True)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
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

    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert {"f1_mean", "f2_mean", "f1_std"}.issubset(set(result.columns))


def test_main_raises_on_empty_feature_type_glob(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(
                make_feat_cfg(
                    tmp_path,
                    feature_type_files=str(tmp_path / "nonexistent" / "*.parquet"),
                )
            )


def test_main_pyc_feature_select_called(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch("pycytominer.feature_select") as mock_fs:
            mock_fs.side_effect = lambda profiles, **kw: profiles
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    mock_fs.assert_called_once()


def test_main_pyc_feature_select_dropped_feature_absent(tmp_path) -> None:
    _write_default_fixtures(tmp_path)

    def drop_f1_mean(profiles, **_kwargs):
        return profiles.drop(columns=["f1_mean"])

    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch("pycytominer.feature_select", side_effect=drop_f1_mean):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "f1_mean" not in result.columns


# ---------------------------------------------------------------------------
# compute_impact_score — main() integration
# ---------------------------------------------------------------------------


def test_main_impact_score_column_present_by_default(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert IMPACT_SCORE_COL in result.columns


def test_main_impact_score_column_absent_when_disabled(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path, compute_impact_score=False))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert IMPACT_SCORE_COL not in result.columns


def test_main_impact_score_values_are_finite(tmp_path) -> None:
    # A1A/A2A/A3A are the synonymous control group; their normalized feature
    # vectors are non-zero, so compute_impact_score produces finite scores
    # for all rows.
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert result[IMPACT_SCORE_COL].is_finite().all()


def test_main_synonymous_median_has_zero_impact_score(tmp_path) -> None:
    # variant_classification marks A1A/A2A/A3A (all Synonymous) as the
    # control group. compute_impact_score's reference vector is their
    # *median*, not their mean — with three non-uniformly-spaced values
    # (raw f1_mean = 0.0, 1.0, 4.0), A2A (raw f1_mean=1.0) is the actual
    # middle value, so its normalized vector exactly equals the control
    # median and its impact score should be 0.
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    median_row = result.filter(pl.col("meta_aa_changes") == "A2A")
    assert median_row[IMPACT_SCORE_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_main_output_features_are_synonymous_normalized(tmp_path) -> None:
    # Output feature values should be z-scored against the synonymous
    # (A1A/A2A/A3A) control group's mean/std, not left as raw aggregate
    # values. Verify against an independent numpy computation from the
    # fixture's known raw values (write_feature_type_aggregate).
    result = _run_main(tmp_path)

    control_f1 = np.array([0.0, 1.0, 4.0])
    control_f2 = np.array([0.0, 2.0, 8.0])
    f1_mean, f1_std = control_f1.mean(), control_f1.std(ddof=1)
    f2_mean, f2_std = control_f2.mean(), control_f2.std(ddof=1)

    raw_values = {
        "A1A": (0.0, 0.0),
        "A2A": (1.0, 2.0),
        "A3A": (4.0, 8.0),
        "A1B": (5.0, 6.0),
        "A1C": (10.0, 12.0),
    }
    for label, (raw_f1, raw_f2) in raw_values.items():
        row = result.filter(pl.col("meta_aa_changes") == label)
        assert row["f1_mean"][0] == pytest.approx((raw_f1 - f1_mean) / f1_std, abs=1e-9)
        assert row["f2_mean"][0] == pytest.approx((raw_f2 - f2_mean) / f2_std, abs=1e-9)


# ---------------------------------------------------------------------------
# aggregate meta data — main() integration
# ---------------------------------------------------------------------------


def _run_main(tmp_path, **kwargs) -> pl.DataFrame:
    """Run main() and return the output parquet."""
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
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


# ---------------------------------------------------------------------------
# PCA / UMAP — main() integration
# ---------------------------------------------------------------------------


def test_main_pca_off_by_default_no_pc_columns(tmp_path) -> None:
    result = _run_main(tmp_path)
    assert not any(c.startswith("meta_pc_") for c in result.columns)
    assert not (tmp_path / "out" / "pca_components.parquet").exists()


def test_main_umap_off_by_default_no_umap_columns(tmp_path) -> None:
    result = _run_main(tmp_path)
    assert not any(c.startswith("meta_umap_") for c in result.columns)


def test_main_run_pca_adds_pc_columns(tmp_path) -> None:
    result = _run_main(tmp_path, run_pca=True, pca_n_components=2)
    assert "meta_pc_1" in result.columns
    assert "meta_pc_2" in result.columns
    assert not any(c.startswith("meta_umap_") for c in result.columns)


def test_main_run_pca_writes_components_file_with_expected_schema(tmp_path) -> None:
    _run_main(tmp_path, run_pca=True, pca_n_components=2)
    components = pl.read_parquet(tmp_path / "out" / "pca_components.parquet")
    assert components["meta_component_idx"].to_list() == [1, 2]
    assert "meta_variance_explained" in components.columns
    assert "meta_cumulative_variance_explained" in components.columns
    other_cols = set(components.columns) - {
        "meta_component_idx",
        "meta_variance_explained",
        "meta_cumulative_variance_explained",
    }
    assert other_cols == {"f1_mean", "f2_mean"}


def test_main_run_pca_components_file_absent_when_pca_off(tmp_path) -> None:
    _run_main(tmp_path, run_pca=False)
    assert not (tmp_path / "out" / "pca_components.parquet").exists()


def test_main_run_pca_respects_output_root_naming(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            m.main.__wrapped__(
                make_feat_cfg(
                    tmp_path, output_root=root, run_pca=True, pca_n_components=2
                )
            )
    assert (tmp_path / "run1.pca_components.parquet").exists()
    assert not (tmp_path / "out" / "pca_components.parquet").exists()


def test_main_run_umap_adds_umap_columns(tmp_path) -> None:
    result = _run_main(tmp_path, run_umap=True, umap_n_components=2, umap_n_neighbors=2)
    assert "meta_umap_1" in result.columns
    assert "meta_umap_2" in result.columns
    assert not any(c.startswith("meta_pc_") for c in result.columns)


def test_main_umap_metric_passed_to_compute_umap(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            with patch("fisseq_data_pipeline.featureselect.compute_umap") as mock_umap:
                mock_umap.return_value = pl.DataFrame(
                    {
                        "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
                        "meta_umap_1": [0.0] * 5,
                        "meta_umap_2": [0.0] * 5,
                    }
                )
                m.main.__wrapped__(
                    make_feat_cfg(
                        tmp_path,
                        run_umap=True,
                        umap_metric="euclidean",
                        umap_n_neighbors=2,
                    )
                )
    # compute_umap(df, label_column, n_components, n_neighbors, metric, ...)
    assert mock_umap.call_args.args[4] == "euclidean"


def test_main_pca_umap_join_by_label_not_position(tmp_path) -> None:
    # compute_pca/compute_umap return scores in a different row order than
    # the aggregate's own order; main() must join by label_column, not by
    # position.
    _write_default_fixtures(tmp_path)
    shuffled_labels = ["A1C", "A1A", "A3A", "A1B", "A2A"]
    pc_values = {label: float(i) for i, label in enumerate(shuffled_labels)}
    umap_values = {label: float(-i) for i, label in enumerate(shuffled_labels)}

    fake_scores_df = pl.DataFrame(
        {
            "meta_aa_changes": shuffled_labels,
            "meta_pc_1": [pc_values[label] for label in shuffled_labels],
        }
    )
    fake_components_df = pl.DataFrame(
        {
            "meta_component_idx": [1],
            "f1_mean": [0.1],
            "f2_mean": [0.2],
            "meta_variance_explained": [1.0],
            "meta_cumulative_variance_explained": [1.0],
        }
    )
    fake_umap_df = pl.DataFrame(
        {
            "meta_aa_changes": shuffled_labels,
            "meta_umap_1": [umap_values[label] for label in shuffled_labels],
        }
    )

    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with patch(
            "pycytominer.feature_select", side_effect=lambda profiles, **_kw: profiles
        ):
            with patch(
                "fisseq_data_pipeline.featureselect.compute_pca",
                return_value=(fake_scores_df, fake_components_df),
            ):
                with patch(
                    "fisseq_data_pipeline.featureselect.compute_umap",
                    return_value=fake_umap_df,
                ):
                    m.main.__wrapped__(
                        make_feat_cfg(
                            tmp_path,
                            run_pca=True,
                            pca_n_components=1,
                            run_umap=True,
                            umap_n_components=1,
                        )
                    )
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    for label, expected in pc_values.items():
        row = result.filter(pl.col("meta_aa_changes") == label)
        assert row["meta_pc_1"][0] == pytest.approx(expected)
    for label, expected in umap_values.items():
        row = result.filter(pl.col("meta_aa_changes") == label)
        assert row["meta_umap_1"][0] == pytest.approx(expected)


def test_main_pca_all_null_feature_column_dropped_with_warning(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    # f3_mean is constant across the synonymous control group (A1A/A2A/A3A),
    # so the Normalizer stores std=None for it and it normalizes to
    # entirely null -- compute_pca must drop it (with a warning) rather than
    # fail.
    write_feat_input_parquet(tmp_path)
    ft_dir = tmp_path / "ft"
    ft_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
            "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
            "f2_mean": [0.0, 2.0, 8.0, 6.0, 12.0],
            "f3_mean": [5.0, 5.0, 5.0, 9.0, 3.0],
        }
    ).write_parquet(ft_dir / "mean.parquet")
    pl.DataFrame(
        {
            "feature": ["f1_mean", "f2_mean", "f3_mean"],
            "feature_ok": [True, True, True],
        }
    ).write_parquet(tmp_path / "blocklist.parquet")

    with caplog.at_level(logging.WARNING):
        with patch("fisseq_data_pipeline.featureselect.setup_logging"):
            with patch(
                "pycytominer.feature_select",
                side_effect=lambda profiles, **_kw: profiles,
            ):
                m.main.__wrapped__(
                    make_feat_cfg(tmp_path, run_pca=True, pca_n_components=2)
                )
    assert "f3_mean" in caplog.text
    components = pl.read_parquet(tmp_path / "out" / "pca_components.parquet")
    assert "f3_mean" not in components.columns
    assert {"f1_mean", "f2_mean"}.issubset(set(components.columns))


def test_main_impact_score_unaffected_by_pca_or_umap(tmp_path) -> None:
    baseline = _run_main(tmp_path)
    with_embeddings = _run_main(
        tmp_path,
        run_pca=True,
        pca_n_components=2,
        run_umap=True,
        umap_n_components=2,
        umap_n_neighbors=2,
    )
    baseline_sorted = baseline.sort("meta_aa_changes")
    with_embeddings_sorted = with_embeddings.sort("meta_aa_changes")
    assert baseline_sorted[IMPACT_SCORE_COL].to_list() == pytest.approx(
        with_embeddings_sorted[IMPACT_SCORE_COL].to_list()
    )
