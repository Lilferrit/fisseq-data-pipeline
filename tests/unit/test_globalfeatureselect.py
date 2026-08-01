from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.globalfeatureselect as m

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
# normalize_batch_aggregate
# ---------------------------------------------------------------------------


def _write_batch_aggregates(pipeline_dir, batch_stem: str) -> None:
    """A1A/A2A/A3A are Synonymous (classify_variant) and form the
    normalization reference; A1B/A1C are not."""
    agg_dir = pipeline_dir / "feature_select_batchwise" / batch_stem / "aggregates"
    agg_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
            "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
            "f2_mean": [0.0, 2.0, 8.0, 6.0, 12.0],
        }
    ).write_parquet(agg_dir / "mean.parquet")


def test_normalize_batch_aggregate_normalizes_to_synonymous_baseline(tmp_path) -> None:
    _write_batch_aggregates(tmp_path, "batchA")
    result = m.normalize_batch_aggregate(
        str(tmp_path), "batchA", "meta_aa_changes"
    ).collect()

    control_f1 = np.array([0.0, 1.0, 4.0])
    f1_mean, f1_std = control_f1.mean(), control_f1.std(ddof=1)
    row = result.filter(pl.col("meta_aa_changes") == "A1B")
    assert row["f1_mean"][0] == pytest.approx((5.0 - f1_mean) / f1_std, abs=1e-9)


def test_normalize_batch_aggregate_raises_on_empty_glob(tmp_path) -> None:
    with pytest.raises(ValueError):
        m.normalize_batch_aggregate(
            str(tmp_path), "nonexistent_batch", "meta_aa_changes"
        )


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
# main() — end to end
# ---------------------------------------------------------------------------


def _write_pipeline_dir(tmp_path, *, block_f2: bool = False):
    pipeline_dir = tmp_path / "pipeline"
    for batch_stem in ["batchA", "batchB"]:
        _write_batch_aggregates(pipeline_dir, batch_stem)
        bl_dir = pipeline_dir / "feature_select_batchwise" / batch_stem
        bl_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {"feature": ["f1_mean", "f2_mean"], "feature_ok": [True, not block_f2]}
        ).write_parquet(bl_dir / "blocklist.parquet")
    return pipeline_dir


def make_gfs_cfg(tmp_path, pipeline_dir, *, min_batches_ok=None) -> OmegaConf:
    return OmegaConf.structured(
        m.GlobalFeatureSelectConfig(
            output_dir=str(tmp_path / "out"),
            pipeline_dir=str(pipeline_dir),
            batch_stems=["batchA", "batchB"],
            min_batches_ok=min_batches_ok,
        )
    )


def test_main_writes_both_outputs(tmp_path) -> None:
    pipeline_dir = _write_pipeline_dir(tmp_path)
    with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
        with patch(
            "fisseq_data_pipeline.featureselect.pycytominer.feature_select",
            side_effect=lambda profiles, **_kw: profiles,
        ):
            m.main.__wrapped__(make_gfs_cfg(tmp_path, pipeline_dir))
    assert (tmp_path / "out" / "aggregate.parquet").exists()
    assert (tmp_path / "out" / "blocklist.parquet").exists()


def test_main_blocked_feature_absent_from_aggregate(tmp_path) -> None:
    pipeline_dir = _write_pipeline_dir(tmp_path, block_f2=True)
    with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
        with patch(
            "fisseq_data_pipeline.featureselect.pycytominer.feature_select",
            side_effect=lambda profiles, **_kw: profiles,
        ):
            m.main.__wrapped__(make_gfs_cfg(tmp_path, pipeline_dir))
    result = pl.read_parquet(tmp_path / "out" / "aggregate.parquet")
    assert "f2_mean" not in result.columns
    assert "f1_mean" in result.columns


def test_main_aggregate_has_one_row_per_variant(tmp_path) -> None:
    # Both batches contribute the same 5 variants (A1A/A2A/A3A/A1B/A1C) ->
    # median_across_batches collapses each to a single row.
    pipeline_dir = _write_pipeline_dir(tmp_path)
    with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
        with patch(
            "fisseq_data_pipeline.featureselect.pycytominer.feature_select",
            side_effect=lambda profiles, **_kw: profiles,
        ):
            m.main.__wrapped__(make_gfs_cfg(tmp_path, pipeline_dir))
    result = pl.read_parquet(tmp_path / "out" / "aggregate.parquet")
    assert len(result) == 5


def test_main_raises_on_empty_batch_stems(tmp_path) -> None:
    pipeline_dir = _write_pipeline_dir(tmp_path)
    cfg = OmegaConf.structured(
        m.GlobalFeatureSelectConfig(
            output_dir=str(tmp_path / "out"),
            pipeline_dir=str(pipeline_dir),
            batch_stems=[],
        )
    )
    with patch("fisseq_data_pipeline.globalfeatureselect.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(cfg)
