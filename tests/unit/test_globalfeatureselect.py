from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.globalfeatureselect as m
from fisseq_data_pipeline.aggregate import _AGGREGATORS
from fisseq_data_pipeline.utils.constants import IMPACT_SCORE_COL

# ---------------------------------------------------------------------------
# combine_batch_blocklists
# ---------------------------------------------------------------------------


def _write_blocklist(path, feature_ok: dict) -> str:
    pl.DataFrame(
        {"feature": list(feature_ok.keys()), "feature_ok": list(feature_ok.values())}
    ).write_parquet(path)
    return str(path)


def test_combine_batch_blocklists_default_requires_unanimity(tmp_path) -> None:
    p1 = _write_blocklist(tmp_path / "b1.parquet", {"f1": True, "f2": True})
    p2 = _write_blocklist(tmp_path / "b2.parquet", {"f1": True, "f2": False})
    p3 = _write_blocklist(tmp_path / "b3.parquet", {"f1": True, "f2": True})
    result = m.combine_batch_blocklists([p1, p2, p3], min_batches_ok=None)
    f1 = result.filter(pl.col("feature") == "f1")
    f2 = result.filter(pl.col("feature") == "f2")
    assert f1["feature_ok"][0] is True
    assert f2["feature_ok"][0] is False


def test_combine_batch_blocklists_counts_correct(tmp_path) -> None:
    p1 = _write_blocklist(tmp_path / "b1.parquet", {"f1": True})
    p2 = _write_blocklist(tmp_path / "b2.parquet", {"f1": False})
    p3 = _write_blocklist(tmp_path / "b3.parquet", {"f1": True})
    result = m.combine_batch_blocklists([p1, p2, p3], min_batches_ok=None)
    row = result.filter(pl.col("feature") == "f1")
    assert row["n_batches"][0] == 3
    assert row["n_ok"][0] == 2


def test_combine_batch_blocklists_explicit_threshold(tmp_path) -> None:
    p1 = _write_blocklist(tmp_path / "b1.parquet", {"f2": True})
    p2 = _write_blocklist(tmp_path / "b2.parquet", {"f2": False})
    p3 = _write_blocklist(tmp_path / "b3.parquet", {"f2": True})
    result = m.combine_batch_blocklists([p1, p2, p3], min_batches_ok=2)
    f2 = result.filter(pl.col("feature") == "f2")
    assert f2["feature_ok"][0] is True


def test_combine_batch_blocklists_raises_on_empty_paths() -> None:
    with pytest.raises(ValueError):
        m.combine_batch_blocklists([], min_batches_ok=None)


# ---------------------------------------------------------------------------
# _reconstruct_staged_paths / group_paths_by_batch
#
# These back main()'s translation from Nextflow's stageAs auto-numbering
# (agg_input_1.parquet, agg_input_2.parquet, ... -- see
# modules/local/global_feature_select.nf) back into per-batch file lists.
# ---------------------------------------------------------------------------


def test_reconstruct_staged_paths_numbers_from_one() -> None:
    assert m._reconstruct_staged_paths(3, "agg_input") == [
        "agg_input_1.parquet",
        "agg_input_2.parquet",
        "agg_input_3.parquet",
    ]


def test_reconstruct_staged_paths_empty() -> None:
    assert m._reconstruct_staged_paths(0, "agg_input") == []


def test_group_paths_by_batch_preserves_order_and_groups() -> None:
    grouped = m.group_paths_by_batch(
        ["batchA", "batchB", "batchA"], ["a1.parquet", "b1.parquet", "a2.parquet"]
    )
    assert list(grouped.keys()) == ["batchA", "batchB"]
    assert grouped["batchA"] == ["a1.parquet", "a2.parquet"]
    assert grouped["batchB"] == ["b1.parquet"]


def test_group_paths_by_batch_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        m.group_paths_by_batch(["batchA"], ["a1.parquet", "a2.parquet"])


# ---------------------------------------------------------------------------
# normalize_batch_aggregate
# ---------------------------------------------------------------------------


def _write_batch_aggregate(dir_path, filename: str = "mean.parquet") -> str:
    """A1A/A2A/A3A are Synonymous (classify_variant) and form the
    normalization reference; A1B/A1C are not."""
    path = dir_path / filename
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
            "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
            "f2_mean": [0.0, 2.0, 8.0, 6.0, 12.0],
        }
    ).write_parquet(path)
    return str(path)


def test_normalize_batch_aggregate_normalizes_to_synonymous_baseline(tmp_path) -> None:
    path = _write_batch_aggregate(tmp_path)
    result = m.normalize_batch_aggregate([path], "meta_aa_changes").collect()

    control_f1 = np.array([0.0, 1.0, 4.0])
    f1_mean, f1_std = control_f1.mean(), control_f1.std(ddof=1)
    row = result.filter(pl.col("meta_aa_changes") == "A1B")
    assert row["f1_mean"][0] == pytest.approx((5.0 - f1_mean) / f1_std, abs=1e-9)


def test_normalize_batch_aggregate_raises_on_empty_paths() -> None:
    with pytest.raises(ValueError):
        m.normalize_batch_aggregate([], "meta_aa_changes")


def test_normalize_batch_aggregate_drops_blocked_features(tmp_path) -> None:
    path = _write_batch_aggregate(tmp_path)
    result = m.normalize_batch_aggregate(
        [path], "meta_aa_changes", blocked_features={"f2_mean"}
    ).collect()
    assert "f2_mean" not in result.columns
    assert "f1_mean" in result.columns


# ---------------------------------------------------------------------------
# median_across_batches
# ---------------------------------------------------------------------------


def test_median_across_batches_medians_shared_variant() -> None:
    batch1 = pl.LazyFrame(
        {"meta_aa_changes": ["shared", "only_in_1"], "f1_mean": [1.0, 100.0]}
    )
    batch2 = pl.LazyFrame(
        {"meta_aa_changes": ["shared", "only_in_2"], "f1_mean": [3.0, 200.0]}
    )
    batch3 = pl.LazyFrame({"meta_aa_changes": ["shared"], "f1_mean": [5.0]})
    result = m.median_across_batches([batch1, batch2, batch3], "meta_aa_changes")
    row = result.filter(pl.col("meta_aa_changes") == "shared")
    assert row["f1_mean"][0] == 3.0


def test_median_across_batches_single_batch_variant_unchanged() -> None:
    batch1 = pl.LazyFrame(
        {"meta_aa_changes": ["shared", "only_in_1"], "f1_mean": [1.0, 42.0]}
    )
    batch2 = pl.LazyFrame({"meta_aa_changes": ["shared"], "f1_mean": [3.0]})
    result = m.median_across_batches([batch1, batch2], "meta_aa_changes")
    row = result.filter(pl.col("meta_aa_changes") == "only_in_1")
    assert row["f1_mean"][0] == 42.0


def test_median_across_batches_drops_feature_columns_not_common_to_all_batches() -> (
    None
):
    batch1 = pl.LazyFrame(
        {"meta_aa_changes": ["shared"], "f1_mean": [1.0], "f2_mean": [10.0]}
    )
    batch2 = pl.LazyFrame({"meta_aa_changes": ["shared"], "f1_mean": [3.0]})
    result = m.median_across_batches([batch1, batch2], "meta_aa_changes")
    assert "f1_mean" in result.columns
    assert "f2_mean" not in result.columns


def test_median_across_batches_warns_on_dropped_columns(caplog) -> None:
    batch1 = pl.LazyFrame(
        {"meta_aa_changes": ["shared"], "f1_mean": [1.0], "f2_mean": [10.0]}
    )
    batch2 = pl.LazyFrame({"meta_aa_changes": ["shared"], "f1_mean": [3.0]})
    with caplog.at_level(logging.WARNING):
        m.median_across_batches(
            [batch1, batch2], "meta_aa_changes", batch_labels=["batchA", "batchB"]
        )
    assert any(
        "batchA" in rec.message and "f2_mean" in rec.message for rec in caplog.records
    )


def test_median_across_batches_drops_all_meta_columns_except_label() -> None:
    batch1 = pl.LazyFrame(
        {
            "meta_aa_changes": ["shared"],
            "meta_is_control": [False],
            "f1_mean": [1.0],
        }
    )
    batch2 = pl.LazyFrame(
        {"meta_aa_changes": ["shared"], "meta_batch": ["batchB"], "f1_mean": [3.0]}
    )
    result = m.median_across_batches([batch1, batch2], "meta_aa_changes")
    assert set(result.columns) == {"meta_aa_changes", "f1_mean"}


def test_median_across_batches_raises_on_empty_batch_lfs() -> None:
    with pytest.raises(ValueError):
        m.median_across_batches([], "meta_aa_changes")


def test_median_across_batches_raises_on_empty_feature_intersection() -> None:
    batch1 = pl.LazyFrame({"meta_aa_changes": ["shared"], "f1_mean": [1.0]})
    batch2 = pl.LazyFrame({"meta_aa_changes": ["shared"], "f2_mean": [3.0]})
    with pytest.raises(ValueError):
        m.median_across_batches([batch1, batch2], "meta_aa_changes")


# ---------------------------------------------------------------------------
# select_global_aggregate
# ---------------------------------------------------------------------------


def test_select_global_aggregate_drops_blocked_columns() -> None:
    agg_df = pl.DataFrame(
        {"meta_aa_changes": ["A", "B"], "f1_mean": [1.0, 2.0], "f2_mean": [3.0, 4.0]}
    )
    bl_df = pl.DataFrame(
        {"feature": ["f1_mean", "f2_mean"], "feature_ok": [True, False]}
    )
    with patch(
        "fisseq_data_pipeline.featureselect.pycytominer.feature_select",
        side_effect=lambda profiles, **_kw: profiles,
    ):
        result = m.select_global_aggregate(agg_df, bl_df)
    assert "f1_mean" in result.columns
    assert "f2_mean" not in result.columns


# ---------------------------------------------------------------------------
# classify_features_by_type
# ---------------------------------------------------------------------------


def test_classify_features_by_type_longest_match_KS_vs_KSnegLogP() -> None:
    result = m.classify_features_by_type(
        ["foo_bar_KSnegLogP", "foo_KS"], ["_KS", "_KSnegLogP"]
    )
    assert result == {"KSnegLogP": ["foo_bar_KSnegLogP"], "KS": ["foo_KS"]}


def test_classify_features_by_type_longest_match_AUROC_vs_AUROCnegLogP() -> None:
    result = m.classify_features_by_type(
        ["f1_AUROCnegLogP", "f1_AUROC"], ["_AUROC", "_AUROCnegLogP"]
    )
    assert result == {"AUROCnegLogP": ["f1_AUROCnegLogP"], "AUROC": ["f1_AUROC"]}


def test_classify_features_by_type_excludes_unmatched_columns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        result = m.classify_features_by_type(["f1_mean", "meta_something"], ["_mean"])
    assert result == {"mean": ["f1_mean"]}
    assert "meta_something" in caplog.text


def test_classify_features_by_type_empty_columns_returns_empty_dict() -> None:
    assert m.classify_features_by_type([], ["_mean", "_KS"]) == {}


def test_classify_features_by_type_empty_suffixes_returns_empty_dict() -> None:
    assert m.classify_features_by_type(["f1_mean"], []) == {}


def test_classify_features_by_type_omits_suffixes_with_no_matches() -> None:
    result = m.classify_features_by_type(["f1_mean"], ["_mean", "_KS", "_AUROC"])
    assert list(result.keys()) == ["mean"]


def test_classify_features_by_type_preserves_column_order_within_bucket() -> None:
    result = m.classify_features_by_type(["f2_mean", "f1_mean", "f3_mean"], ["_mean"])
    assert result["mean"] == ["f2_mean", "f1_mean", "f3_mean"]


def test_classify_features_by_type_duplicate_suffixes_deduped() -> None:
    assert m.classify_features_by_type(["f1_mean"], ["_mean", "_mean"]) == {
        "mean": ["f1_mean"]
    }


# ---------------------------------------------------------------------------
# main() — end to end
#
# main() reads its aggregate/blocklist inputs as bare, cwd-relative
# filenames (agg_input_1.parquet, bl_input_1.parquet, ...) -- reconstructing
# the names Nextflow's stageAs auto-numbering stages them under (see
# _reconstruct_staged_paths) -- so these tests stage files directly into
# tmp_path and chdir into it, mirroring a real task's work directory, rather
# than building a pipeline_dir tree for main() to glob.
# ---------------------------------------------------------------------------


def _stage_batches(work_dir, *, block_f2: bool = False):
    """Stage one "mean" aggregate file and one blocklist file per batch
    (batchA, batchB) into work_dir, named the way Nextflow's stageAs
    auto-numbering would name them, and return the kwargs make_gfs_cfg needs
    to describe that staging. Both batches carry identical aggregate data.
    """
    batch_stems = ["batchA", "batchB"]
    for i, _batch_stem in enumerate(batch_stems, start=1):
        _write_batch_aggregate(work_dir, filename=f"agg_input_{i}.parquet")
    for i, _batch_stem in enumerate(batch_stems, start=1):
        pl.DataFrame(
            {"feature": ["f1_mean", "f2_mean"], "feature_ok": [True, not block_f2]}
        ).write_parquet(work_dir / f"bl_input_{i}.parquet")
    return dict(
        agg_batch_stems=list(batch_stems),
        n_agg_files=len(batch_stems),
        bl_batch_stems=list(batch_stems),
        n_blocklist_files=len(batch_stems),
    )


def _write_batch_aggregate_multi_type(dir_path, filename: str = "multi.parquet") -> str:
    """Like _write_batch_aggregate, but spans multiple aggregate feature
    types -- including both suffix-prefix-collision pairs (_KS/_KSnegLogP,
    _AUROC/_AUROCnegLogP) -- to exercise classify_features_by_type's
    longest-match logic through the full main() path. A1A/A2A/A3A are
    Synonymous (classify_variant) and form the normalization reference;
    A1B/A1C are not. Every column's control triple has nonzero std, so
    normalization never hits the all-null-column path."""
    path = dir_path / filename
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
            "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
            "f2_mean": [0.0, 2.0, 8.0, 6.0, 12.0],
            "f1_KS": [0.1, 0.2, 0.3, 0.9, 0.8],
            "f1_KSnegLogP": [1.0, 1.5, 2.0, 5.0, 6.0],
            "f1_AUROC": [0.5, 0.6, 0.55, 0.9, 0.95],
            "f1_AUROCnegLogP": [0.0, 0.1, 0.2, 3.0, 4.0],
        }
    ).write_parquet(path)
    return str(path)


def _stage_batches_multi_type(work_dir, *, blocked: tuple = ()) -> dict:
    """Like _stage_batches, but with per-batch aggregates spanning multiple
    aggregate feature types (see _write_batch_aggregate_multi_type). `blocked`
    names feature columns to mark feature_ok=False in every batch's
    blocklist, so select_global_aggregate drops them from aggregate.parquet
    while the per-type files -- captured before the blocklist is
    (re-)applied -- must still retain them."""
    batch_stems = ["batchA", "batchB"]
    all_features = [
        "f1_mean",
        "f2_mean",
        "f1_KS",
        "f1_KSnegLogP",
        "f1_AUROC",
        "f1_AUROCnegLogP",
    ]
    for i, _batch_stem in enumerate(batch_stems, start=1):
        _write_batch_aggregate_multi_type(work_dir, filename=f"agg_input_{i}.parquet")
    for i, _batch_stem in enumerate(batch_stems, start=1):
        pl.DataFrame(
            {
                "feature": all_features,
                "feature_ok": [f not in blocked for f in all_features],
            }
        ).write_parquet(work_dir / f"bl_input_{i}.parquet")
    return dict(
        agg_batch_stems=list(batch_stems),
        n_agg_files=len(batch_stems),
        bl_batch_stems=list(batch_stems),
        n_blocklist_files=len(batch_stems),
    )


def make_gfs_cfg(
    tmp_path,
    staged: dict,
    *,
    min_batches_ok=None,
    compute_impact_score=None,
    run_pca: bool = False,
    pca_n_components: int = 2,
    run_umap: bool = False,
    umap_n_components: int = 2,
    umap_n_neighbors: int = 2,
    umap_metric: str = "cosine",
    umap_min_dist: float = 0.1,
    umap_random_state=42,
) -> OmegaConf:
    kwargs = dict(
        output_dir=str(tmp_path / "out"),
        min_batches_ok=min_batches_ok,
        run_pca=run_pca,
        pca_n_components=pca_n_components,
        run_umap=run_umap,
        umap_n_components=umap_n_components,
        umap_n_neighbors=umap_n_neighbors,
        umap_metric=umap_metric,
        umap_min_dist=umap_min_dist,
        umap_random_state=umap_random_state,
        **staged,
    )
    if compute_impact_score is not None:
        kwargs["compute_impact_score"] = compute_impact_score
    return OmegaConf.structured(m.GlobalFeatureSelectConfig(**kwargs))


def _run_gfs_main(tmp_path, monkeypatch, staged: dict, **kwargs) -> pl.DataFrame:
    """chdir into tmp_path (where staged input files live), run main(), and
    return the written aggregate parquet."""
    monkeypatch.chdir(tmp_path)
    with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
        with patch(
            "fisseq_data_pipeline.featureselect.pycytominer.feature_select",
            side_effect=lambda profiles, **_kw: profiles,
        ):
            m.main.__wrapped__(make_gfs_cfg(tmp_path, staged, **kwargs))
    return pl.read_parquet(tmp_path / "out" / "aggregate.parquet")


def test_main_writes_both_outputs(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    _run_gfs_main(tmp_path, monkeypatch, staged)
    assert (tmp_path / "out" / "aggregate.parquet").exists()
    assert (tmp_path / "out" / "blocklist.parquet").exists()


def test_main_blocked_feature_absent_from_aggregate(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path, block_f2=True)
    result = _run_gfs_main(tmp_path, monkeypatch, staged)
    assert "f2_mean" not in result.columns
    assert "f1_mean" in result.columns


def test_main_aggregate_has_one_row_per_variant(tmp_path, monkeypatch) -> None:
    # Both batches contribute the same 5 variants (A1A/A2A/A3A/A1B/A1C) ->
    # median_across_batches collapses each to a single row.
    staged = _stage_batches(tmp_path)
    result = _run_gfs_main(tmp_path, monkeypatch, staged)
    assert len(result) == 5


def test_main_raises_on_empty_agg_batch_stems(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    staged["agg_batch_stems"] = []
    staged["n_agg_files"] = 0
    monkeypatch.chdir(tmp_path)
    with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(make_gfs_cfg(tmp_path, staged))


def test_main_raises_on_empty_bl_batch_stems(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    staged["bl_batch_stems"] = []
    staged["n_blocklist_files"] = 0
    monkeypatch.chdir(tmp_path)
    with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(make_gfs_cfg(tmp_path, staged))


def test_main_tolerates_batch_present_in_agg_but_not_blocklist(
    tmp_path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    # batchC contributes an aggregate but (e.g. because its own
    # blocklist-combination chain failed upstream, tolerated under
    # errorStrategy 'ignore') no blocklist -- main() must warn and proceed
    # using only the batches available on each side, not crash.
    staged = _stage_batches(tmp_path)
    _write_batch_aggregate(tmp_path, filename="agg_input_3.parquet")
    staged["agg_batch_stems"].append("batchC")
    staged["n_agg_files"] = 3
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING):
        with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
            with patch(
                "fisseq_data_pipeline.featureselect.pycytominer.feature_select",
                side_effect=lambda profiles, **_kw: profiles,
            ):
                m.main.__wrapped__(make_gfs_cfg(tmp_path, staged))
    assert "batchC" in caplog.text
    # batchC still contributes to the aggregate (it just isn't
    # blocklist-filtered by its own batch's report).
    result = pl.read_parquet(tmp_path / "out" / "aggregate.parquet")
    assert len(result) == 5


# ---------------------------------------------------------------------------
# aggregate_{feature_type}.parquet — main() integration
# ---------------------------------------------------------------------------


def test_main_writes_one_file_per_present_feature_type(tmp_path, monkeypatch) -> None:
    staged = _stage_batches_multi_type(tmp_path)
    _run_gfs_main(tmp_path, monkeypatch, staged)
    out = tmp_path / "out"

    for type_name in ("mean", "KS", "KSnegLogP", "AUROC", "AUROCnegLogP"):
        assert (out / f"aggregate_{type_name}.parquet").exists()

    mean_df = pl.read_parquet(out / "aggregate_mean.parquet")
    assert set(mean_df.columns) == {"meta_aa_changes", "f1_mean", "f2_mean"}
    ks_df = pl.read_parquet(out / "aggregate_KS.parquet")
    assert set(ks_df.columns) == {"meta_aa_changes", "f1_KS"}
    ksneglogp_df = pl.read_parquet(out / "aggregate_KSnegLogP.parquet")
    assert set(ksneglogp_df.columns) == {"meta_aa_changes", "f1_KSnegLogP"}
    auroc_df = pl.read_parquet(out / "aggregate_AUROC.parquet")
    assert set(auroc_df.columns) == {"meta_aa_changes", "f1_AUROC"}
    aurocneglogp_df = pl.read_parquet(out / "aggregate_AUROCnegLogP.parquet")
    assert set(aurocneglogp_df.columns) == {"meta_aa_changes", "f1_AUROCnegLogP"}


def test_main_per_type_columns_union_equals_median_aggregate_columns(
    tmp_path, monkeypatch
) -> None:
    staged = _stage_batches_multi_type(tmp_path)
    _run_gfs_main(tmp_path, monkeypatch, staged)
    out = tmp_path / "out"

    union_cols = set()
    for p in sorted(out.glob("aggregate_*.parquet")):
        union_cols |= set(pl.read_parquet(p).columns) - {"meta_aa_changes"}
    assert union_cols == {
        "f1_mean",
        "f2_mean",
        "f1_KS",
        "f1_KSnegLogP",
        "f1_AUROC",
        "f1_AUROCnegLogP",
    }


def _stage_batches_correlated_features(work_dir):
    """Stage two batches whose f1_KS column is an exact positive multiple of
    f1_mean (2x). Per-column z-score normalization is affine, so this
    correlation survives normalize_batch_aggregate/median_across_batches
    intact -- real (unmocked) pycytominer correlation_threshold filtering
    (inside select_global_aggregate's pyc_feature_select call) is then
    guaranteed to drop one of the two columns, giving an observable
    pyc_feature_select-driven difference between agg_df (captured into the
    aggregate_{type}.parquet files) and the final aggregate.parquet -- unlike
    the global blocklist, which is already applied per batch by
    normalize_batch_aggregate, well before the per-type split point (so a
    blocklist-only difference is never observable there)."""
    batch_stems = ["batchA", "batchB"]
    for i, _batch_stem in enumerate(batch_stems, start=1):
        pl.DataFrame(
            {
                "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
                "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
                "f1_KS": [0.0, 2.0, 8.0, 10.0, 20.0],
            }
        ).write_parquet(work_dir / f"agg_input_{i}.parquet")
    for i, _batch_stem in enumerate(batch_stems, start=1):
        pl.DataFrame(
            {"feature": ["f1_mean", "f1_KS"], "feature_ok": [True, True]}
        ).write_parquet(work_dir / f"bl_input_{i}.parquet")
    return dict(
        agg_batch_stems=list(batch_stems),
        n_agg_files=len(batch_stems),
        bl_batch_stems=list(batch_stems),
        n_blocklist_files=len(batch_stems),
    )


def test_main_per_type_files_are_superset_of_final_aggregate_columns(
    tmp_path, monkeypatch
) -> None:
    # f1_mean and f1_KS are perfectly correlated -> pycytominer's real (not
    # mocked) correlation_threshold filter drops one of them inside
    # select_global_aggregate, so the final aggregate.parquet ends up with
    # strictly fewer feature columns than agg_df had. The dropped column
    # must still be present in its aggregate_{type}.parquet file, since that
    # file is captured before select_global_aggregate runs.
    staged = _stage_batches_correlated_features(tmp_path)
    monkeypatch.chdir(tmp_path)
    with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
        m.main.__wrapped__(make_gfs_cfg(tmp_path, staged))
    out = tmp_path / "out"
    result = pl.read_parquet(out / "aggregate.parquet")

    # Sanity check the correlation actually triggered a real drop -- one of
    # the two columns is missing from the final aggregate.
    result_features = {"f1_mean", "f1_KS"} & set(result.columns)
    assert len(result_features) == 1

    final_by_type = m.classify_features_by_type(
        [c for c in result.columns if c != "meta_aa_changes"],
        [cls._stat_suffix for cls in _AGGREGATORS.values()],
    )
    for type_name, final_cols in final_by_type.items():
        per_type_cols = set(
            pl.read_parquet(out / f"aggregate_{type_name}.parquet").columns
        )
        assert set(final_cols) <= per_type_cols

    # The specific column pycytominer dropped is still present in its own
    # per-type file, even though it's gone from the final aggregate.
    dropped = ({"f1_mean", "f1_KS"} - result_features).pop()
    dropped_type = "mean" if dropped == "f1_mean" else "KS"
    assert dropped in pl.read_parquet(out / f"aggregate_{dropped_type}.parquet").columns


# ---------------------------------------------------------------------------
# compute_impact_score — main() integration
# ---------------------------------------------------------------------------


def test_main_impact_score_column_present_by_default(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    result = _run_gfs_main(tmp_path, monkeypatch, staged)
    assert IMPACT_SCORE_COL in result.columns
    assert "meta_is_control" in result.columns


def test_main_impact_score_column_absent_when_disabled(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    result = _run_gfs_main(tmp_path, monkeypatch, staged, compute_impact_score=False)
    assert IMPACT_SCORE_COL not in result.columns


def test_main_impact_score_values_are_finite(tmp_path, monkeypatch) -> None:
    # A1A/A2A/A3A are the synonymous control group; their normalized feature
    # vectors are non-zero, so compute_impact_score produces finite scores
    # for all rows.
    staged = _stage_batches(tmp_path)
    result = _run_gfs_main(tmp_path, monkeypatch, staged)
    assert result[IMPACT_SCORE_COL].is_finite().all()


def test_main_synonymous_median_has_zero_impact_score(tmp_path, monkeypatch) -> None:
    # _write_batch_aggregate's f2_mean is exactly 2x f1_mean for the
    # synonymous control rows (A1A/A2A/A3A), so each control row's
    # normalized vector lies along the same direction and A2A (the middle
    # raw value) lands exactly on the control median vector -> impact
    # score 0. batchA and batchB carry identical data, so the cross-batch
    # median leaves each batch's per-variant normalized values unchanged.
    staged = _stage_batches(tmp_path)
    result = _run_gfs_main(tmp_path, monkeypatch, staged)
    median_row = result.filter(pl.col("meta_aa_changes") == "A2A")
    assert median_row[IMPACT_SCORE_COL][0] == pytest.approx(0.0, abs=1e-9)


def _stage_mismatched_batches(work_dir):
    """batchA and batchB deliberately diverge in all three ways this fix
    guards against: batchA carries a stray metadata column (``meta_extra``)
    batchB lacks, batchA has a feature column (``f4_mean``) batchB lacks,
    and ``f3_mean`` is blocked globally even though only batchA's blocklist
    reports on it (unanimity-among-reporters)."""
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
            "meta_extra": ["x", "x", "x", "x", "x"],
            "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
            "f2_mean": [0.0, 2.0, 8.0, 6.0, 12.0],
            "f3_mean": [0.0, 1.0, 2.0, 3.0, 4.0],
            "f4_mean": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    ).write_parquet(work_dir / "agg_input_1.parquet")
    pl.DataFrame(
        {
            "feature": ["f1_mean", "f2_mean", "f3_mean"],
            "feature_ok": [True, True, False],
        }
    ).write_parquet(work_dir / "bl_input_1.parquet")

    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
            "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
            "f2_mean": [0.0, 2.0, 8.0, 6.0, 12.0],
            "f3_mean": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    ).write_parquet(work_dir / "agg_input_2.parquet")
    pl.DataFrame(
        {"feature": ["f1_mean", "f2_mean"], "feature_ok": [True, True]}
    ).write_parquet(work_dir / "bl_input_2.parquet")

    return dict(
        agg_batch_stems=["batchA", "batchB"],
        n_agg_files=2,
        bl_batch_stems=["batchA", "batchB"],
        n_blocklist_files=2,
    )


def test_main_handles_mismatched_batch_schemas(tmp_path, monkeypatch) -> None:
    # compute_impact_score=False: this test is scoped to schema-mismatch
    # handling upstream of feature selection, not the impact-score step,
    # which re-adds its own meta_is_control/meta_impact_score afterward
    # (see the "compute_impact_score" test section).
    staged = _stage_mismatched_batches(tmp_path)
    result = _run_gfs_main(tmp_path, monkeypatch, staged, compute_impact_score=False)
    assert "f1_mean" in result.columns
    assert "f2_mean" in result.columns
    assert "f3_mean" not in result.columns  # globally blocked
    assert "f4_mean" not in result.columns  # not common to every batch
    assert "meta_extra" not in result.columns  # dropped: not label_column
    assert [c for c in result.columns if c.startswith("meta_")] == ["meta_aa_changes"]


# ---------------------------------------------------------------------------
# PCA / UMAP — main() integration
# ---------------------------------------------------------------------------


def test_main_pca_off_by_default_no_pc_columns(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    result = _run_gfs_main(tmp_path, monkeypatch, staged)
    assert not any(c.startswith("meta_pc_") for c in result.columns)
    assert not (tmp_path / "out" / "pca_components.parquet").exists()


def test_main_umap_off_by_default_no_umap_columns(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    result = _run_gfs_main(tmp_path, monkeypatch, staged)
    assert not any(c.startswith("meta_umap_") for c in result.columns)


def test_main_run_pca_adds_pc_columns(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    result = _run_gfs_main(
        tmp_path, monkeypatch, staged, run_pca=True, pca_n_components=2
    )
    assert "meta_pc_1" in result.columns
    assert "meta_pc_2" in result.columns


def test_main_run_pca_writes_components_file_with_expected_schema(
    tmp_path, monkeypatch
) -> None:
    staged = _stage_batches(tmp_path)
    _run_gfs_main(tmp_path, monkeypatch, staged, run_pca=True, pca_n_components=2)
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


def test_main_run_pca_components_file_absent_when_pca_off(
    tmp_path, monkeypatch
) -> None:
    staged = _stage_batches(tmp_path)
    _run_gfs_main(tmp_path, monkeypatch, staged, run_pca=False)
    assert not (tmp_path / "out" / "pca_components.parquet").exists()


def test_main_run_umap_adds_umap_columns(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    result = _run_gfs_main(
        tmp_path,
        monkeypatch,
        staged,
        run_umap=True,
        umap_n_components=2,
        umap_n_neighbors=2,
    )
    assert "meta_umap_1" in result.columns
    assert "meta_umap_2" in result.columns


def test_main_umap_metric_passed_to_compute_umap(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    monkeypatch.chdir(tmp_path)
    with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
        with patch(
            "fisseq_data_pipeline.featureselect.pycytominer.feature_select",
            side_effect=lambda profiles, **_kw: profiles,
        ):
            with patch(
                "fisseq_data_pipeline.globalfeatureselect.compute_umap"
            ) as mock_umap:
                mock_umap.return_value = pl.DataFrame(
                    {
                        "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
                        "meta_umap_1": [0.0] * 5,
                        "meta_umap_2": [0.0] * 5,
                    }
                )
                m.main.__wrapped__(
                    make_gfs_cfg(
                        tmp_path,
                        staged,
                        run_umap=True,
                        umap_metric="euclidean",
                        umap_n_neighbors=2,
                    )
                )
    # compute_umap(df, label_column, n_components, n_neighbors, metric, ...)
    assert mock_umap.call_args.args[4] == "euclidean"


def test_main_pca_umap_join_by_label_not_position(tmp_path, monkeypatch) -> None:
    # compute_pca/compute_umap return scores in a different row order than
    # the aggregate's own order; main() must join by label_column, not by
    # position.
    staged = _stage_batches(tmp_path)
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

    monkeypatch.chdir(tmp_path)
    with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
        with patch(
            "fisseq_data_pipeline.featureselect.pycytominer.feature_select",
            side_effect=lambda profiles, **_kw: profiles,
        ):
            with patch(
                "fisseq_data_pipeline.globalfeatureselect.compute_pca",
                return_value=(fake_scores_df, fake_components_df),
            ):
                with patch(
                    "fisseq_data_pipeline.globalfeatureselect.compute_umap",
                    return_value=fake_umap_df,
                ):
                    m.main.__wrapped__(
                        make_gfs_cfg(
                            tmp_path,
                            staged,
                            run_pca=True,
                            pca_n_components=1,
                            run_umap=True,
                            umap_n_components=1,
                        )
                    )
    result = pl.read_parquet(tmp_path / "out" / "aggregate.parquet")
    for label, expected in pc_values.items():
        row = result.filter(pl.col("meta_aa_changes") == label)
        assert row["meta_pc_1"][0] == pytest.approx(expected)
    for label, expected in umap_values.items():
        row = result.filter(pl.col("meta_aa_changes") == label)
        assert row["meta_umap_1"][0] == pytest.approx(expected)


def _stage_batches_with_null_feature(work_dir):
    # f3_mean is constant across the synonymous control group (A1A/A2A/A3A)
    # in both batches, so per-batch normalization stores std=None for it and
    # it normalizes to entirely null in the cross-batch median too --
    # compute_pca must drop it (with a warning) rather than fail.
    batch_stems = ["batchA", "batchB"]
    for i, _batch_stem in enumerate(batch_stems, start=1):
        pl.DataFrame(
            {
                "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
                "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
                "f2_mean": [0.0, 2.0, 8.0, 6.0, 12.0],
                "f3_mean": [5.0, 5.0, 5.0, 9.0, 3.0],
            }
        ).write_parquet(work_dir / f"agg_input_{i}.parquet")
        pl.DataFrame(
            {
                "feature": ["f1_mean", "f2_mean", "f3_mean"],
                "feature_ok": [True, True, True],
            }
        ).write_parquet(work_dir / f"bl_input_{i}.parquet")
    return dict(
        agg_batch_stems=list(batch_stems),
        n_agg_files=len(batch_stems),
        bl_batch_stems=list(batch_stems),
        n_blocklist_files=len(batch_stems),
    )


def test_main_pca_all_null_feature_column_dropped_with_warning(
    tmp_path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    staged = _stage_batches_with_null_feature(tmp_path)
    with caplog.at_level(logging.WARNING):
        _run_gfs_main(tmp_path, monkeypatch, staged, run_pca=True, pca_n_components=2)
    assert "f3_mean" in caplog.text
    components = pl.read_parquet(tmp_path / "out" / "pca_components.parquet")
    assert "f3_mean" not in components.columns
    assert {"f1_mean", "f2_mean"}.issubset(set(components.columns))


def test_main_impact_score_unaffected_by_pca_or_umap(tmp_path, monkeypatch) -> None:
    staged = _stage_batches(tmp_path)
    baseline = _run_gfs_main(tmp_path, monkeypatch, staged)
    # main() only ever writes into tmp_path / "out", so restaging the same
    # inputs and rerunning with PCA/UMAP enabled is safe -- the second call
    # simply overwrites that same output directory.
    with_embeddings = _run_gfs_main(
        tmp_path,
        monkeypatch,
        staged,
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
