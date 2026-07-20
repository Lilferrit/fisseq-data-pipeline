"""Integration tests for the FISSEQ Nextflow pipeline."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

# 10 WT barcodes × 6 cells = 60 WT cells
# 5 A1A barcodes × 6 cells = 30 Synonymous cells  (A→A at position 1)
# 5 M1K barcodes × 6 cells = 30 Single Missense cells
# 5 M1K:downsampled-half barcodes x 6 cells = 30 tagged Single Missense cells,
# which must pool with the untagged M1K rows under meta_aa_changes == "M1K"
# once qcfilter.py's filter_columns strips the ":downsampled-half" tag.
_VARIANTS = {
    "WT": ("bc_wt_{i:02d}", 10, 6),
    "A1A": ("bc_syn_{i:02d}", 5, 6),
    "M1K": ("bc_mis_{i:02d}", 5, 6),
    "M1K:downsampled-half": ("bc_mis_tag_{i:02d}", 5, 6),
}

_FEATURE_COLS = [
    "Cells_AreaShape_Area",
    "Cells_AreaShape_Perimeter",
    "Cells_Intensity_Mean",
    "Nuclei_AreaShape_Area",
    "Nuclei_Intensity_Max",
]

# Low thresholds so the small synthetic dataset passes every pipeline step.
_NF_PARAMS = [
    "--bc_threshold",
    "3",
    "--variant_bc_threshold",
    "3",
    "--ovwt_min_cells",
    "25",
    "--downsample_wt",
    "50",
    "--bvb_min_cells",
    "50",
    "--bvb_min_batches",
    "2",
    "--bootstrap",
    "3",
]

_PROJECT_ROOT = Path(__file__).parents[2]


def _write_batch(path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for variant, (bc_fmt, n_barcodes, cells_per_bc) in _VARIANTS.items():
        for i in range(n_barcodes):
            bc = bc_fmt.format(i=i)
            for _ in range(cells_per_bc):
                row: dict = {
                    "upBarcode": bc,
                    "aaChanges": variant,
                    "editDistance": 0,
                }
                for col in _FEATURE_COLS:
                    row[col] = float(rng.normal())
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


_OVWT_NF_PARAMS = [
    "--bc_threshold",
    "3",
    "--variant_bc_threshold",
    "3",
    "--ovwt_min_cells",
    "25",
    "--downsample_wt",
    "50",
]


def _run_pipeline(exp_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--input_dir",
            str(exp_dir),
            *_NF_PARAMS,
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _run_ovwt_pipeline(exp_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--workflow",
            "ovwt",
            "--input_dir",
            str(exp_dir),
            *_OVWT_NF_PARAMS,
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )


# ---------------------------------------------------------------------------
# Session fixture — pipeline runs once, all pipeline tests share the outputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipeline_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_experiment")
    _write_batch(exp_dir / "input" / "batch1.parquet", seed=42)
    _write_batch(exp_dir / "input" / "batch2.parquet", seed=99)

    result = _run_pipeline(exp_dir)
    return exp_dir, result


# ---------------------------------------------------------------------------
# Pipeline exit / structure tests
# ---------------------------------------------------------------------------


def test_pipeline_exits_cleanly(pipeline_outputs):
    _, result = pipeline_outputs
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_pipeline_qc_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    qc = exp_dir / "qc_filter" / batch_stem
    assert (qc / "filtered_cells.parquet").exists()
    assert (qc / "barcode_counts.parquet").exists()
    assert (qc / "variants_per_barcode.parquet").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_pipeline_tagged_variant_pools_with_base(pipeline_outputs, batch_stem):
    """A raw aaChanges value of 'M1K:downsampled-half' is split by
    qcfilter.py's filter_columns into meta_aa_changes == 'M1K' (pooled with
    the untagged M1K rows) and meta_variant_tag == 'downsampled-half'."""
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "qc_filter" / batch_stem / "filtered_cells.parquet")
    m1k = df.filter(pl.col("meta_aa_changes") == "M1K")
    # Both the untagged (bc_mis_*) and tagged (bc_mis_tag_*) M1K groups are
    # 5 barcodes x 6 cells = 30 cells each in _write_batch, so pooling both
    # under one meta_aa_changes group yields 60 cells.
    assert m1k.shape[0] == 60

    tags = set(m1k["meta_variant_tag"].to_list())
    assert tags == {None, "downsampled-half"}
    assert "M1K:downsampled-half" not in df["meta_aa_changes"].to_list()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_pipeline_normalization_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "normalization" / "cells" / f"{batch_stem}.parquet").exists()
    assert (
        exp_dir / "normalization" / "normalizers" / f"{batch_stem}.normalizer.parquet"
    ).exists()


def test_pipeline_batchvsbatch_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "batchvsbatch" / "pre" / "results.parquet").exists()
    assert (exp_dir / "batchvsbatch" / "post" / "results.parquet").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_pipeline_ovwt_batchwise_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    batch_dir = exp_dir / "ovwt_batchwise" / batch_stem
    assert (batch_dir / "results.parquet").exists()
    assert (batch_dir / "models.pkl").exists()
    assert (batch_dir / "test_index.parquet").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_pipeline_ovwt_batchwise_test_index_columns(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "ovwt_batchwise" / batch_stem / "test_index.parquet")
    assert set(df.columns) == {"row_idx", "origin_file"}


def test_pipeline_ovwt_global_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "ovwt_global" / "results.parquet").exists()
    assert (exp_dir / "ovwt_global" / "models.pkl").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_pipeline_feature_select_batchwise_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    batch_dir = exp_dir / "feature_select_batchwise" / batch_stem
    assert (batch_dir / "output.parquet").exists()
    assert (batch_dir / "blocklist.parquet").exists()


def test_pipeline_feature_select_global_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "feature_select_global" / "output.parquet").exists()
    assert (exp_dir / "feature_select_global" / "blocklist.parquet").exists()


def test_pipeline_permanova_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "permanova" / "permanova.parquet").exists()


# ---------------------------------------------------------------------------
# Pipeline output content tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_normalized_cells_wt_mean_near_zero(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "normalization" / "cells" / f"{batch_stem}.parquet")
    wt = df.filter(pl.col("meta_aa_changes") == "WT")
    feature_cols = [c for c in df.columns if not c.startswith("meta_")]
    for col in feature_cols:
        assert abs(wt[col].drop_nulls().mean()) < 0.5, (
            f"WT mean for {col} not near zero after normalization"
        )


@pytest.mark.parametrize("stage", ["pre", "post"])
def test_batchvsbatch_has_expected_columns(pipeline_outputs, stage):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "batchvsbatch" / stage / "results.parquet")
    expected = {"variant", "batch", "auroc", "mw_pvalue", "n_batch_cells", "n_cells"}
    assert expected.issubset(set(df.columns))
    assert len(df) > 0


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_results_have_auroc_columns(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "ovwt_batchwise" / batch_stem / "results.parquet")
    for col in ("train_auroc", "val_auroc", "test_auroc"):
        assert col in df.columns


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_feature_correlations_have_feature_ok_column(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "feature_select_batchwise" / batch_stem / "blocklist.parquet"
    )
    assert "feature_ok" in df.columns


def test_ovwt_global_uses_both_batches(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "ovwt_global" / "results.parquet")
    assert (df["meta_batch_num_unique"] == 2).all()


def test_feature_select_global_uses_both_batches(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "feature_select_global" / "output.parquet")
    assert (df["meta_batch_num_unique"] == 2).all()


def test_permanova_has_expected_columns(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "permanova" / "permanova.parquet")
    expected = {"meta_aa_changes", "f_statistic", "p_value", "meta_num_cells"}
    assert expected.issubset(set(df.columns))
    assert len(df) > 0


def test_permanova_uses_both_batches(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "permanova" / "permanova.parquet")
    assert (df["meta_batch_num_unique"] == 2).all()


def test_permanova_f_statistic_is_finite(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "permanova" / "permanova.parquet")
    assert df["f_statistic"].is_finite().all()
    assert df["p_value"].is_between(0.0, 1.0, closed="both").all()


# ---------------------------------------------------------------------------
# Batch correction branch (qc_filtering -> batch_correction -> permanova),
# independent of the normalize branch above.
# ---------------------------------------------------------------------------


def test_pipeline_batch_correction_fit_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    fit_dir = exp_dir / "batch_correction" / "fit"
    assert (fit_dir / "stats_vb.parquet").exists()
    assert (fit_dir / "centroids.parquet").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_pipeline_batch_correction_cells_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "batch_correction" / "cells" / f"{batch_stem}.parquet").exists()


def test_pipeline_batch_correction_permanova_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "batch_correction" / "permanova" / "permanova.parquet").exists()


def test_batch_correction_permanova_has_expected_columns(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "batch_correction" / "permanova" / "permanova.parquet"
    )
    expected = {"meta_aa_changes", "f_statistic", "p_value", "meta_num_cells"}
    assert expected.issubset(set(df.columns))
    assert len(df) > 0


def test_batch_correction_permanova_uses_both_batches(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "batch_correction" / "permanova" / "permanova.parquet"
    )
    assert (df["meta_batch_num_unique"] == 2).all()


def test_batch_correction_permanova_f_statistic_is_finite(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "batch_correction" / "permanova" / "permanova.parquet"
    )
    assert df["f_statistic"].is_finite().all()
    assert df["p_value"].is_between(0.0, 1.0, closed="both").all()


def test_batch_correction_wt_means_converge_across_batches(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df1 = pl.read_parquet(exp_dir / "batch_correction" / "cells" / "batch1.parquet")
    df2 = pl.read_parquet(exp_dir / "batch_correction" / "cells" / "batch2.parquet")
    wt1 = df1.filter(pl.col("meta_aa_changes") == "WT")
    wt2 = df2.filter(pl.col("meta_aa_changes") == "WT")
    feature_cols = [c for c in df1.columns if not c.startswith("meta_")]
    for col in feature_cols:
        mean1 = wt1[col].drop_nulls().mean()
        mean2 = wt2[col].drop_nulls().mean()
        assert abs(mean1 - mean2) < 0.5, (
            f"WT mean for {col} did not converge across batches after batch correction"
        )


# ---------------------------------------------------------------------------
# OvwtPipeline (ovwt.nf) — session fixture and tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ovwt_pipeline_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_ovwt_experiment")
    _write_batch(exp_dir / "input" / "batch1.parquet", seed=42)
    _write_batch(exp_dir / "input" / "batch2.parquet", seed=99)

    result = _run_ovwt_pipeline(exp_dir)
    return exp_dir, result


def test_ovwt_pipeline_exits_cleanly(ovwt_pipeline_outputs):
    _, result = ovwt_pipeline_outputs
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_pipeline_test_index_exists(ovwt_pipeline_outputs, batch_stem):
    exp_dir, _ = ovwt_pipeline_outputs
    assert (exp_dir / "ovwt_batchwise" / batch_stem / "test_index.parquet").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_pipeline_test_index_columns(ovwt_pipeline_outputs, batch_stem):
    exp_dir, _ = ovwt_pipeline_outputs
    df = pl.read_parquet(exp_dir / "ovwt_batchwise" / batch_stem / "test_index.parquet")
    assert set(df.columns) == {"row_idx", "origin_file"}


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_pipeline_cell_scores_exist(ovwt_pipeline_outputs, batch_stem):
    exp_dir, _ = ovwt_pipeline_outputs
    assert (
        exp_dir / "ovwt_cellscores_batchwise" / batch_stem / "cell_scores.parquet"
    ).exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_pipeline_cell_scores_row_count_matches_test_index(
    ovwt_pipeline_outputs, batch_stem
):
    exp_dir, _ = ovwt_pipeline_outputs
    index_df = pl.read_parquet(
        exp_dir / "ovwt_batchwise" / batch_stem / "test_index.parquet"
    )
    scores_df = pl.read_parquet(
        exp_dir / "ovwt_cellscores_batchwise" / batch_stem / "cell_scores.parquet"
    )
    assert len(scores_df) == len(index_df)


# ---------------------------------------------------------------------------
# params.config_dir — optional INPUT stage (workflows/fisseq.nf,
# workflows/ovwt.nf). Uses the lighter OvwtPipeline so this test isn't
# bottlenecked by the full feature-selection DAG; the thing under test is the
# INPUT -> QC_FILTER channel wiring, not downstream analysis.
# ---------------------------------------------------------------------------


def _write_input_config(path: Path, source_path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump({"input_paths": [str(source_path)]}, f)


def _run_config_dir_pipeline(
    exp_dir: Path, config_dir: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--workflow",
            "ovwt",
            "--input_dir",
            str(exp_dir),
            "--config_dir",
            str(config_dir),
            *_OVWT_NF_PARAMS,
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture(scope="session")
def config_dir_pipeline_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_config_dir_experiment")
    # One pre-staged batch, directly under input/, to exercise both code
    # paths (pre-staged glob + config-driven INPUT) feeding the same run.
    _write_batch(exp_dir / "input" / "batch1.parquet", seed=42)

    # One config-driven batch: raw source data lives outside input/, and a
    # YAML config in config_dir points INPUT at it.
    raw_dir = tmp_path_factory.mktemp("nf_config_dir_raw")
    _write_batch(raw_dir / "batch2_source.parquet", seed=99)
    config_dir = exp_dir / "configs"
    config_dir.mkdir()
    _write_input_config(config_dir / "batch2.yaml", raw_dir / "batch2_source.parquet")

    result = _run_config_dir_pipeline(exp_dir, config_dir)
    return exp_dir, result


def test_config_dir_pipeline_exits_cleanly(config_dir_pipeline_outputs):
    _, result = config_dir_pipeline_outputs
    assert result.returncode == 0, result.stderr


def test_config_dir_generates_input_parquet(config_dir_pipeline_outputs):
    exp_dir, _ = config_dir_pipeline_outputs
    assert (exp_dir / "input" / "batch2.parquet").exists()


def test_config_dir_pre_staged_batch_still_processed(config_dir_pipeline_outputs):
    exp_dir, _ = config_dir_pipeline_outputs
    assert (exp_dir / "qc_filter" / "batch1").exists()


def test_config_dir_batch_not_double_processed(config_dir_pipeline_outputs):
    """Regression test for the glob/channel dedup in workflows/ovwt.nf: the
    config-derived batch2 must be QC-filtered exactly once, not twice (once
    from INPUT's live channel output, once from a stale glob re-match of the
    file INPUT published to disk)."""
    exp_dir, _ = config_dir_pipeline_outputs
    qc_dir = exp_dir / "qc_filter" / "batch2"
    assert qc_dir.exists()

    df = pl.read_parquet(qc_dir / "filtered_cells.parquet")
    # Same variant/barcode/cell-count shape as _write_batch produces for any
    # single batch (60 WT + 30 Synonymous + 30 Missense + 30 tagged-Missense
    # cells, all passing the loose _OVWT_NF_PARAMS QC thresholds) — a doubled
    # channel would produce exactly 2x this row count.
    expected_cells = sum(n * c for _, n, c in _VARIANTS.values())
    assert df.shape[0] == expected_cells
