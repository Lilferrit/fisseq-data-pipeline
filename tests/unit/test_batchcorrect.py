import logging
from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

from fisseq_data_pipeline.batchcorrect import (
    BatchCorrectFitConfig,
    BatchCorrector,
    BatchCorrectTransformConfig,
    fit_main,
    transform_main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_lf(
    variant: list[str],
    batch: list[str],
    f1: list[float],
    f2: list[float],
) -> pl.LazyFrame:
    """Build a Float64 LazyFrame with meta_aa_changes and meta_batch columns."""
    return pl.DataFrame(
        {
            "meta_aa_changes": variant,
            "meta_batch": batch,
            "f1": pl.Series("f1", f1, dtype=pl.Float64),
            "f2": pl.Series("f2", f2, dtype=pl.Float64),
        }
    ).lazy()


# Fixture data: WT and M1A each span batches b1/b2 (kept); SINGLE only spans b1 (dropped).
#
#   WT   b1: f1=[1,2,3]   mean=2,  std=1  | f2=[10,20,30]    mean=20,  std=10
#   WT   b2: f1=[4,5,6]   mean=5,  std=1  | f2=[40,50,60]    mean=50,  std=10
#   M1A  b1: f1=[10,11,12] mean=11, std=1 | f2=[100,110,120] mean=110, std=10
#   M1A  b2: f1=[20,21,22] mean=21, std=1 | f2=[200,210,220] mean=210, std=10
#   SINGLE b1: f1=[100]                    | f2=[1000]
#
# centroids (mean of per-batch means/stds, not raw cells):
#   WT:  centroid_mean f1=3.5, centroid_std f1=1 | centroid_mean f2=35, centroid_std f2=10
#   M1A: centroid_mean f1=16,  centroid_std f1=1 | centroid_mean f2=160, centroid_std f2=10


def make_fixture_lf() -> pl.LazyFrame:
    variant = ["WT"] * 3 + ["WT"] * 3 + ["M1A"] * 3 + ["M1A"] * 3 + ["SINGLE"]
    batch = ["b1"] * 3 + ["b2"] * 3 + ["b1"] * 3 + ["b2"] * 3 + ["b1"]
    f1 = [1, 2, 3, 4, 5, 6, 10, 11, 12, 20, 21, 22, 100]
    f2 = [10, 20, 30, 40, 50, 60, 100, 110, 120, 200, 210, 220, 1000]
    return make_lf(variant, batch, [float(v) for v in f1], [float(v) for v in f2])


# ---------------------------------------------------------------------------
# BatchCorrector.from_lazyframe — happy path (variant spans >1 batch)
# ---------------------------------------------------------------------------


def test_from_lazyframe_computes_stats_vb_mean_and_std():
    corrector = BatchCorrector.from_lazyframe(
        make_fixture_lf(), label_col="meta_aa_changes"
    )
    stats = corrector.stats_vb

    def _get(variant, batch, stat, col):
        row = stats.filter(
            (pl.col("meta_aa_changes") == variant)
            & (pl.col("meta_batch") == batch)
            & (pl.col("_stat") == stat)
        )
        return row[col][0]

    assert _get("WT", "b1", "mean", "f1") == pytest.approx(2.0)
    assert _get("WT", "b1", "std", "f1") == pytest.approx(1.0)
    assert _get("WT", "b2", "mean", "f1") == pytest.approx(5.0)
    assert _get("M1A", "b1", "mean", "f2") == pytest.approx(110.0)
    assert _get("M1A", "b2", "std", "f2") == pytest.approx(10.0)


def test_from_lazyframe_computes_centroid_as_mean_of_batch_stats():
    corrector = BatchCorrector.from_lazyframe(
        make_fixture_lf(), label_col="meta_aa_changes"
    )
    centroids = corrector.centroids

    def _get(variant, stat, col):
        row = centroids.filter(
            (pl.col("meta_aa_changes") == variant) & (pl.col("_stat") == stat)
        )
        return row[col][0]

    # mean-of-means/mean-of-stds over batches, NOT mean over raw cells
    assert _get("WT", "mean", "f1") == pytest.approx((2.0 + 5.0) / 2)
    assert _get("WT", "std", "f1") == pytest.approx((1.0 + 1.0) / 2)
    assert _get("M1A", "mean", "f2") == pytest.approx((110.0 + 210.0) / 2)
    assert _get("M1A", "std", "f2") == pytest.approx((10.0 + 10.0) / 2)


# ---------------------------------------------------------------------------
# BatchCorrector.from_lazyframe — single-batch variant dropped, not error
# ---------------------------------------------------------------------------


def test_from_lazyframe_drops_single_batch_variant_without_error():
    corrector = BatchCorrector.from_lazyframe(
        make_fixture_lf(), label_col="meta_aa_changes"
    )
    assert "SINGLE" not in corrector.stats_vb.get_column("meta_aa_changes").to_list()
    assert "SINGLE" not in corrector.centroids.get_column("meta_aa_changes").to_list()
    # kept variants are unaffected
    assert set(corrector.centroids.get_column("meta_aa_changes").to_list()) == {
        "WT",
        "M1A",
    }


def test_from_lazyframe_logs_dropped_variant(caplog):
    with caplog.at_level(logging.INFO):
        BatchCorrector.from_lazyframe(make_fixture_lf(), label_col="meta_aa_changes")
    assert any("SINGLE" in record.message for record in caplog.records)
    assert any("Dropping 1 variant" in record.message for record in caplog.records)


def test_from_lazyframe_raises_if_wt_not_in_multiple_batches():
    # WT only present in one batch -> no WT centroid can be computed
    lf = make_lf(
        variant=["WT", "WT", "M1A", "M1A"],
        batch=["b1", "b1", "b1", "b2"],
        f1=[1.0, 2.0, 10.0, 20.0],
        f2=[10.0, 20.0, 100.0, 200.0],
    )
    with pytest.raises(ValueError, match="WT"):
        BatchCorrector.from_lazyframe(lf, label_col="meta_aa_changes")


# ---------------------------------------------------------------------------
# BatchCorrector.transform — WT rescale in isolation
# ---------------------------------------------------------------------------


def _make_corrector_by_hand(
    v_mean_vb,
    v_std_vb,
    v_centroid_mean,
    v_centroid_std,
    wt_centroid_mean,
    wt_centroid_std,
) -> BatchCorrector:
    """Hand-construct a BatchCorrector with one non-WT variant "V" in batch "b1"."""
    stats_vb = pl.DataFrame(
        {
            "meta_aa_changes": ["V", "WT"],
            "meta_batch": ["b1", "b1"],
            "_stat": ["mean", "mean"],
            "f1": [v_mean_vb, 0.0],
        }
    )
    stats_vb = pl.concat(
        [
            stats_vb,
            pl.DataFrame(
                {
                    "meta_aa_changes": ["V", "WT"],
                    "meta_batch": ["b1", "b1"],
                    "_stat": ["std", "std"],
                    "f1": [v_std_vb, 1.0],
                }
            ),
        ]
    )
    centroids = pl.DataFrame(
        {
            "meta_aa_changes": ["V", "WT", "V", "WT"],
            "_stat": ["mean", "mean", "std", "std"],
            "f1": [v_centroid_mean, wt_centroid_mean, v_centroid_std, wt_centroid_std],
        }
    )
    return BatchCorrector(
        stats_vb=stats_vb, centroids=centroids, label_col="meta_aa_changes"
    )


def test_transform_rescale_when_centroid_stds_equal():
    # centroid_std_v == centroid_std_WT isolates the rescale shift from the z-score
    corrector = _make_corrector_by_hand(
        v_mean_vb=10.0,
        v_std_vb=2.0,
        v_centroid_mean=100.0,
        v_centroid_std=1.0,
        wt_centroid_mean=5.0,
        wt_centroid_std=1.0,
    )
    lf = pl.DataFrame({"meta_aa_changes": ["V"], "f1": [14.0]}).lazy()
    out = corrector.transform(lf, batch="b1").collect()
    z = (14.0 - 10.0) / 2.0
    expected = z + 5.0  # z_vb + centroid_mean_WT, since both centroid stds are 1
    assert out["f1"][0] == pytest.approx(expected)


def test_transform_rescale_general_case_matches_hand_computation():
    v_mean_vb, v_std_vb = 10.0, 2.0
    v_centroid_mean, v_centroid_std = 100.0, 5.0
    wt_centroid_mean, wt_centroid_std = 3.0, 2.0
    x = 16.0

    corrector = _make_corrector_by_hand(
        v_mean_vb,
        v_std_vb,
        v_centroid_mean,
        v_centroid_std,
        wt_centroid_mean,
        wt_centroid_std,
    )
    lf = pl.DataFrame({"meta_aa_changes": ["V"], "f1": [x]}).lazy()
    out = corrector.transform(lf, batch="b1").collect()

    x_prime = (x - v_mean_vb) / v_std_vb * v_centroid_std + v_centroid_mean
    x_double_prime = (
        x_prime - v_centroid_mean
    ) / v_centroid_std * wt_centroid_std + wt_centroid_mean
    assert out["f1"][0] == pytest.approx(x_double_prime)


def test_transform_drops_rows_for_unfitted_variant():
    corrector = _make_corrector_by_hand(10.0, 2.0, 100.0, 5.0, 3.0, 2.0)
    lf = pl.DataFrame({"meta_aa_changes": ["V", "UNFITTED"], "f1": [14.0, 1.0]}).lazy()
    out = corrector.transform(lf, batch="b1").collect()
    assert out.height == 1
    assert out["meta_aa_changes"].to_list() == ["V"]


# ---------------------------------------------------------------------------
# fit_main / transform_main — output file count == input file count
# ---------------------------------------------------------------------------


def make_fit_cfg(tmp_path, output_dir):
    return OmegaConf.structured(
        BatchCorrectFitConfig(
            output_dir=str(output_dir),
            input_file=str(tmp_path / "input" / "*.parquet"),
        )
    )


def make_transform_cfg(input_file, output_dir, batch, stats_file, centroids_file):
    return OmegaConf.structured(
        BatchCorrectTransformConfig(
            output_dir=str(output_dir),
            input_file=str(input_file),
            batch=batch,
            stats_file=str(stats_file),
            centroids_file=str(centroids_file),
        )
    )


def write_batch_file(path, variant, meta_cell_id, f1, f2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "meta_aa_changes": variant,
            "meta_cell_id": meta_cell_id,
            "f1": pl.Series("f1", f1, dtype=pl.Float64),
            "f2": pl.Series("f2", f2, dtype=pl.Float64),
        }
    ).write_parquet(path)


def test_fit_and_transform_output_file_count_matches_input_and_traces_batches(tmp_path):
    input_dir = tmp_path / "input"
    write_batch_file(
        input_dir / "batch1.parquet",
        variant=["WT", "WT", "WT", "M1A", "M1A"],
        meta_cell_id=["b1_0", "b1_1", "b1_2", "b1_3", "b1_4"],
        f1=[1.0, 2.0, 3.0, 10.0, 11.0],
        f2=[10.0, 20.0, 30.0, 100.0, 110.0],
    )
    write_batch_file(
        input_dir / "batch2.parquet",
        variant=["WT", "WT", "WT", "M1A", "M1A"],
        meta_cell_id=["b2_0", "b2_1", "b2_2", "b2_3", "b2_4"],
        f1=[4.0, 5.0, 6.0, 20.0, 21.0],
        f2=[40.0, 50.0, 60.0, 200.0, 210.0],
    )

    fit_dir = tmp_path / "fit"
    with patch("fisseq_data_pipeline.batchcorrect.setup_logging"):
        fit_main.__wrapped__(make_fit_cfg(tmp_path, fit_dir))

    stats_file = fit_dir / "stats_vb.parquet"
    centroids_file = fit_dir / "centroids.parquet"
    assert stats_file.exists()
    assert centroids_file.exists()

    corrected_dir = tmp_path / "corrected"
    for stem in ("batch1", "batch2"):
        batch_out_dir = corrected_dir / stem
        with patch("fisseq_data_pipeline.batchcorrect.setup_logging"):
            transform_main.__wrapped__(
                make_transform_cfg(
                    input_dir / f"{stem}.parquet",
                    batch_out_dir,
                    stem,
                    stats_file,
                    centroids_file,
                )
            )

    output_files = sorted(corrected_dir.glob("*/*.parquet"))
    assert len(output_files) == 2

    out1 = pl.read_parquet(corrected_dir / "batch1" / "batch1.parquet")
    out2 = pl.read_parquet(corrected_dir / "batch2" / "batch2.parquet")

    # Correspondence: batch1's output meta_cell_ids all came from batch1's input, and vice versa.
    assert set(out1["meta_cell_id"].to_list()) == {
        "b1_0",
        "b1_1",
        "b1_2",
        "b1_3",
        "b1_4",
    }
    assert set(out2["meta_cell_id"].to_list()) == {
        "b2_0",
        "b2_1",
        "b2_2",
        "b2_3",
        "b2_4",
    }
