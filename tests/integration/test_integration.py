"""Integration tests for fisseq-env-init and the FISSEQ Nextflow pipeline."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

# 10 WT barcodes × 6 cells = 60 WT cells
# 5 A1A barcodes × 6 cells = 30 Synonymous cells  (A→A at position 1)
# 5 M1K barcodes × 6 cells = 30 Single Missense cells
_VARIANTS = {
    "WT":  ("bc_wt_{i:02d}",  10, 6),
    "A1A": ("bc_syn_{i:02d}",  5, 6),
    "M1K": ("bc_mis_{i:02d}", 5, 6),
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
    "--bc_threshold",            "3",
    "--variant_bc_threshold",    "3",
    "--ovwt_min_cells",          "25",
    "--permanova_n_bootstraps",  "3",
    "--permanova_sample_size",   "20",
]


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


def _run_pipeline(exp_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["nextflow", "run", "fisseq_pipeline.nf", "--input_dir", ".", *_NF_PARAMS],
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
    subprocess.run(["fisseq-env-init", str(exp_dir)], check=True)
    _write_batch(exp_dir / "input" / "batch1.parquet")

    result = _run_pipeline(exp_dir)
    return exp_dir, result


# ---------------------------------------------------------------------------
# fisseq-env-init tests
# ---------------------------------------------------------------------------


def test_env_init_creates_input_dir(tmp_path):
    subprocess.run(["fisseq-env-init", str(tmp_path)], check=True)
    assert (tmp_path / "input").is_dir()


def test_env_init_copies_pipeline(tmp_path):
    subprocess.run(["fisseq-env-init", str(tmp_path)], check=True)
    assert (tmp_path / "fisseq_pipeline.nf").is_file()


def test_env_init_copies_nextflow_config(tmp_path):
    subprocess.run(["fisseq-env-init", str(tmp_path)], check=True)
    assert (tmp_path / "nextflow.config").is_file()


def test_env_init_copies_readme(tmp_path):
    subprocess.run(["fisseq-env-init", str(tmp_path)], check=True)
    assert (tmp_path / "PIPELINE_README.md").is_file()


def test_env_init_creates_nested_target(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    subprocess.run(["fisseq-env-init", str(target)], check=True)
    assert target.is_dir()


# ---------------------------------------------------------------------------
# Pipeline exit / structure tests
# ---------------------------------------------------------------------------


def test_pipeline_exits_cleanly(pipeline_outputs):
    _, result = pipeline_outputs
    assert result.returncode == 0, result.stderr


def test_pipeline_qc_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    qc = exp_dir / "qc_filter" / "batch1"
    assert (qc / "filtered_cells.parquet").exists()
    assert (qc / "barcode_counts.parquet").exists()
    assert (qc / "variants_per_barcode.parquet").exists()


def test_pipeline_normalization_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "normalization" / "cells" / "batch1.parquet").exists()
    assert (exp_dir / "normalization" / "normalizers" / "batch1.normalizer.parquet").exists()


def test_pipeline_permanova_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "permanova" / "wildtype" / "permanova.parquet").exists()
    assert (exp_dir / "permanova" / "synonymous" / "permanova.parquet").exists()


def test_pipeline_ovwt_batchwise_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    batch_dir = exp_dir / "ovwt_batchwise" / "batch1"
    assert (batch_dir / "results.parquet").exists()
    assert (batch_dir / "models.pkl").exists()


def test_pipeline_ovwt_global_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "ovwt_global" / "results.parquet").exists()
    assert (exp_dir / "ovwt_global" / "models.pkl").exists()


def test_pipeline_feature_select_batchwise_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    batch_dir = exp_dir / "feature_select_batchwise" / "batch1"
    assert (batch_dir / "batch1.parquet").exists()
    assert (batch_dir / "feature_correlations.parquet").exists()


def test_pipeline_feature_select_global_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "feature_select_global" / "global.parquet").exists()
    assert (exp_dir / "feature_select_global" / "feature_correlations.parquet").exists()


# ---------------------------------------------------------------------------
# Pipeline output content tests
# ---------------------------------------------------------------------------


def test_normalized_cells_wt_mean_near_zero(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "normalization" / "cells" / "batch1.parquet")
    wt = df.filter(pl.col("meta_aa_changes") == "WT")
    feature_cols = [c for c in df.columns if not c.startswith("meta_")]
    for col in feature_cols:
        assert abs(wt[col].drop_nulls().mean()) < 0.5, (
            f"WT mean for {col} not near zero after normalization"
        )


def test_permanova_has_expected_columns(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "permanova" / "wildtype" / "permanova.parquet")
    assert "f_value" in df.columns
    assert "f_value_shuffled" in df.columns
    assert len(df) == 3  # permanova_n_bootstraps=3


def test_ovwt_results_have_auroc_columns(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "ovwt_batchwise" / "batch1" / "results.parquet")
    for col in ("train_auroc", "val_auroc", "test_auroc"):
        assert col in df.columns


def test_feature_correlations_have_feature_ok_column(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "feature_select_batchwise" / "batch1" / "feature_correlations.parquet"
    )
    assert "feature_ok" in df.columns
