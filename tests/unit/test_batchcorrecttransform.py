from unittest.mock import patch

import polars as pl
from omegaconf import OmegaConf

from fisseq_data_pipeline.batchcorrect import BatchCorrectFitConfig
from fisseq_data_pipeline.batchcorrect import main as fit_main
from fisseq_data_pipeline.batchcorrecttransform import BatchCorrectTransformConfig, main


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


def test_main_output_file_count_matches_input_and_traces_batches(tmp_path):
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

    corrected_dir = tmp_path / "corrected"
    for stem in ("batch1", "batch2"):
        batch_out_dir = corrected_dir / stem
        with patch("fisseq_data_pipeline.batchcorrecttransform.setup_logging"):
            main.__wrapped__(
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
