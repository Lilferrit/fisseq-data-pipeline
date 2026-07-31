"""Integration tests for the FISSEQ Nextflow pipeline."""

import re
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

# 10 WT barcodes × 20 cells = 200 WT cells
# 5 A1A barcodes × 6 cells = 30 Synonymous cells  (A→A at position 1)
# 5 M1K barcodes × 6 cells = 30 Single Missense cells
# 5 M1K:downsampled-half barcodes x 6 cells = 30 tagged Single Missense cells,
# which must pool with the untagged M1K rows under meta_aa_changes == "M1K"
# once qcfilter.py's filter_columns strips the ":downsampled-half" tag.
# WT gets more cells/barcode than the other variants: wtvwt.py stratifies
# its 80/10/10 split on individual barcode (finer-grained than ovwt.py's
# per-variant stratification), and sklearn's stratified split requires at
# least 2 members per class in every split -- 6 cells/barcode reliably
# triggers "least populated class has only 1 member" once carved into a 10%
# slice, while 20 does not.
_VARIANTS = {
    "WT": ("bc_wt_{i:02d}", 10, 20),
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
# anova_blocklist_pvalue_threshold is raised well above the default 0.05: with this
# synthetic dataset's fixed seeds, the 5 feature columns have no real batch
# effect (ANOVA p-values ~0.16-0.98), so the default threshold blocks none of
# them -- making the filtered and unfiltered OvWT/batch-vs-batch runs
# indistinguishable for test purposes. 0.5 blocks 4 of the 5 features,
# exercising a non-trivial (some blocked, some not) split.
_NF_PARAMS = [
    "--barcode_count_threshold",
    "3",
    "--variant_barcode_count_threshold",
    "3",
    "--ovwt_min_cells",
    "25",
    "--ovwt_downsample_wt",
    "50",
    "--batchvsbatch_min_cells",
    "50",
    "--batchvsbatch_min_batches",
    "2",
    "--feature_select_bootstrap_reps",
    "3",
    "--anova_blocklist_pvalue_threshold",
    "0.5",
    "--wtvwt_min_cells_per_barcode",
    "10",
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
    "--barcode_count_threshold",
    "3",
    "--variant_barcode_count_threshold",
    "3",
    "--ovwt_min_cells",
    "25",
    "--ovwt_downsample_wt",
    "50",
]

# run_single_cell_scores / run_check_barcodes: barcode_check_min_cells is dropped
# to 2 (default 10) since this synthetic dataset only has 6 cells per
# barcode, and single_cell_scores_split=train (rather than the default
# test) is used to keep more cells per barcode after the 80/10/10 split so
# CHECK_BARCODES has enough per-barcode samples to compare.
_CHECK_BARCODES_NF_PARAMS = _NF_PARAMS + [
    "--run_check_barcodes",
    "true",
    "--single_cell_scores_split",
    "train",
    "--barcode_check_min_cells",
    "2",
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


def _run_check_barcodes_pipeline(exp_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--input_dir",
            str(exp_dir),
            *_CHECK_BARCODES_NF_PARAMS,
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
            "--pipeline_mode",
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
def test_pipeline_ovwt_batchwise_feature_filtered_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    batch_dir = exp_dir / "ovwt_batchwise_feature_filtered" / batch_stem
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
def test_pipeline_wtvwt_batchwise_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    batch_dir = exp_dir / "wtvwt_batchwise" / batch_stem
    assert (batch_dir / "results.parquet").exists()
    assert (batch_dir / "models.pkl").exists()


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


def test_pipeline_anova_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "anova" / "anova.parquet").exists()


def test_pipeline_anova_blocklist_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "anova_blocklist" / "anova_blocklist.parquet").exists()


def test_anova_blocklist_has_expected_columns(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "anova_blocklist" / "anova_blocklist.parquet")
    expected = {"feature", "p_value", "feature_ok"}
    assert expected.issubset(set(df.columns))
    assert len(df) > 0


def test_single_cell_scores_and_check_barcodes_disabled_by_default(pipeline_outputs):
    """params.run_single_cell_scores and params.run_check_barcodes both default to
    false -- FisseqPipeline's output set must be unchanged from before these
    flags existed."""
    exp_dir, _ = pipeline_outputs
    assert not (exp_dir / "ovwt_cellscores_batchwise").exists()
    assert not (exp_dir / "check_barcodes").exists()


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
def test_wtvwt_results_have_expected_columns(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "wtvwt_batchwise" / batch_stem / "results.parquet")
    expected = {
        "barcode_a",
        "barcode_b",
        "train_auroc",
        "val_auroc",
        "test_auroc",
        "n_cells_a",
        "n_cells_b",
    }
    assert expected.issubset(set(df.columns))
    # 10 WT barcodes -> C(10, 2) = 45 pairs, all above wtvwt_min_cells_per_barcode
    assert len(df) == 45


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


def test_anova_has_expected_columns(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "anova" / "anova.parquet")
    expected = {"feature", "f_value", "p_value"}
    assert expected.issubset(set(df.columns))
    assert len(df) > 0


def test_anova_f_statistic_is_finite(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "anova" / "anova.parquet")
    assert df["f_value"].is_finite().all()
    assert df["p_value"].is_between(0.0, 1.0, closed="both").all()


# ---------------------------------------------------------------------------
# Batch correction branch (qc_filtering -> batch_correction -> anova),
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


def test_pipeline_batch_correction_anova_outputs(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "batch_correction" / "anova" / "anova.parquet").exists()


def test_batch_correction_anova_has_expected_columns(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "batch_correction" / "anova" / "anova.parquet")
    expected = {"feature", "f_value", "p_value"}
    assert expected.issubset(set(df.columns))
    assert len(df) > 0


def test_batch_correction_anova_f_statistic_is_finite(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "batch_correction" / "anova" / "anova.parquet")
    assert df["f_value"].is_finite().all()
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
# run_single_cell_scores / run_check_barcodes — session fixture and tests
# (FisseqPipeline, params.run_check_barcodes=true implying run_single_cell_scores)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def check_barcodes_pipeline_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_check_barcodes_experiment")
    _write_batch(exp_dir / "input" / "batch1.parquet", seed=42)
    _write_batch(exp_dir / "input" / "batch2.parquet", seed=99)

    result = _run_check_barcodes_pipeline(exp_dir)
    return exp_dir, result


def test_check_barcodes_pipeline_exits_cleanly(check_barcodes_pipeline_outputs):
    _, result = check_barcodes_pipeline_outputs
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_check_barcodes_pipeline_cell_scores_outputs(
    check_barcodes_pipeline_outputs, batch_stem
):
    exp_dir, _ = check_barcodes_pipeline_outputs
    assert (
        exp_dir / "ovwt_cellscores_batchwise" / batch_stem / "cell_scores.parquet"
    ).exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_check_barcodes_pipeline_results_outputs(
    check_barcodes_pipeline_outputs, batch_stem
):
    exp_dir, _ = check_barcodes_pipeline_outputs
    assert (exp_dir / "check_barcodes" / batch_stem / "results.parquet").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_check_barcodes_results_have_expected_columns(
    check_barcodes_pipeline_outputs, batch_stem
):
    exp_dir, _ = check_barcodes_pipeline_outputs
    df = pl.read_parquet(exp_dir / "check_barcodes" / batch_stem / "results.parquet")
    expected = {
        "variant",
        "barcode",
        "group_mean",
        "comparison_barcode",
        "comparison_group_mean",
        "mean_diff",
        "p_adj",
        "reject",
    }
    assert expected.issubset(set(df.columns))
    # M1K pools 10 barcodes (5 untagged + 5 ":downsampled-half" tagged) and
    # A1A has 5 -- both should yield at least one comparison at
    # barcode_check_min_cells=2.
    assert len(df) > 0


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_check_barcodes_p_adj_in_unit_interval(
    check_barcodes_pipeline_outputs, batch_stem
):
    exp_dir, _ = check_barcodes_pipeline_outputs
    df = pl.read_parquet(exp_dir / "check_barcodes" / batch_stem / "results.parquet")
    assert df["p_adj"].is_between(0.0, 1.0, closed="both").all()


# ---------------------------------------------------------------------------
# run_barcode_filtered_ovwt (default true; _CHECK_BARCODES_NF_PARAMS already
# sets run_check_barcodes=true, so check_barcodes_pipeline_outputs exercises
# the default-enabled BARCODE_BLOCKLIST -> OVWT_BATCHWISE_BARCODE_FILTERED path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_barcode_blocklist_outputs(check_barcodes_pipeline_outputs, batch_stem):
    exp_dir, _ = check_barcodes_pipeline_outputs
    assert (
        exp_dir / "barcode_blocklist" / batch_stem / "barcode_blocklist.parquet"
    ).exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_barcode_blocklist_has_expected_columns(
    check_barcodes_pipeline_outputs, batch_stem
):
    exp_dir, _ = check_barcodes_pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "barcode_blocklist" / batch_stem / "barcode_blocklist.parquet"
    )
    assert set(df.columns) == {"barcode", "p_adj", "barcode_ok"}
    assert len(df) > 0


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_pipeline_ovwt_batchwise_barcode_filtered_outputs(
    check_barcodes_pipeline_outputs, batch_stem
):
    exp_dir, _ = check_barcodes_pipeline_outputs
    batch_dir = exp_dir / "ovwt_batchwise_barcode_filtered" / batch_stem
    assert (batch_dir / "results.parquet").exists()
    assert (batch_dir / "models.pkl").exists()
    assert (batch_dir / "test_index.parquet").exists()


@pytest.fixture(scope="session")
def run_barcode_filtered_ovwt_disabled_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp(
        "nf_run_barcode_filtered_ovwt_disabled_experiment"
    )
    _write_batch(exp_dir / "input" / "batch1.parquet", seed=42)
    _write_batch(exp_dir / "input" / "batch2.parquet", seed=99)

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--input_dir",
            str(exp_dir),
            "--run_barcode_filtered_ovwt",
            "false",
            *_CHECK_BARCODES_NF_PARAMS,
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return exp_dir, result


def test_run_barcode_filtered_ovwt_disabled_pipeline_exits_cleanly(
    run_barcode_filtered_ovwt_disabled_outputs,
):
    _, result = run_barcode_filtered_ovwt_disabled_outputs
    assert result.returncode == 0, result.stderr


def test_run_barcode_filtered_ovwt_disabled_skips_barcode_blocklist_and_filtered_output(
    run_barcode_filtered_ovwt_disabled_outputs,
):
    """params.run_barcode_filtered_ovwt defaults to true (see
    test_barcode_blocklist_outputs / test_pipeline_ovwt_batchwise_barcode_filtered_outputs
    for the default-enabled case); false must skip both BARCODE_BLOCKLIST and
    OVWT_BATCHWISE_BARCODE_FILTERED entirely, while CHECK_BARCODES (set
    explicitly via _CHECK_BARCODES_NF_PARAMS) is unaffected."""
    exp_dir, _ = run_barcode_filtered_ovwt_disabled_outputs
    assert not (exp_dir / "barcode_blocklist").exists()
    assert not (exp_dir / "ovwt_batchwise_barcode_filtered").exists()
    assert (exp_dir / "check_barcodes" / "batch1" / "results.parquet").exists()


def test_run_barcode_filtered_ovwt_default_true_does_not_force_check_barcodes(
    pipeline_outputs,
):
    """run_barcode_filtered_ovwt defaults to true, but must NOT force
    run_check_barcodes on -- otherwise the default FisseqPipeline output
    (run_check_barcodes=false) would silently start producing CHECK_BARCODES/
    BARCODE_BLOCKLIST/OVWT_BATCHWISE_BARCODE_FILTERED output. pipeline_outputs
    uses only _NF_PARAMS (no --run_check_barcodes, no --run_barcode_filtered_ovwt
    override), so this exercises both defaults simultaneously."""
    exp_dir, _ = pipeline_outputs
    assert not (exp_dir / "barcode_blocklist").exists()
    assert not (exp_dir / "ovwt_batchwise_barcode_filtered").exists()


def test_invalid_single_cell_scores_source_fails(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_invalid_source_experiment")
    _write_batch(exp_dir / "input" / "batch1.parquet", seed=42)

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--input_dir",
            str(exp_dir),
            "--run_single_cell_scores",
            "true",
            "--single_cell_scores_split",
            "bogus",
            *_NF_PARAMS,
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode != 0
    assert "single_cell_scores_split" in (result.stdout + result.stderr)


@pytest.fixture(scope="session")
def run_feature_filtered_ovwt_disabled_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp(
        "nf_run_feature_filtered_ovwt_disabled_experiment"
    )
    _write_batch(exp_dir / "input" / "batch1.parquet", seed=42)
    _write_batch(exp_dir / "input" / "batch2.parquet", seed=99)

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--input_dir",
            str(exp_dir),
            "--run_feature_filtered_ovwt",
            "false",
            *_NF_PARAMS,
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return exp_dir, result


def test_run_feature_filtered_ovwt_disabled_pipeline_exits_cleanly(
    run_feature_filtered_ovwt_disabled_outputs,
):
    _, result = run_feature_filtered_ovwt_disabled_outputs
    assert result.returncode == 0, result.stderr


def test_run_feature_filtered_ovwt_disabled_skips_filtered_output(
    run_feature_filtered_ovwt_disabled_outputs,
):
    """params.run_feature_filtered_ovwt defaults to true (see
    test_pipeline_ovwt_batchwise_feature_filtered_outputs for the
    default-enabled case); false must skip OVWT_BATCHWISE_FEATURE_FILTERED
    entirely."""
    exp_dir, _ = run_feature_filtered_ovwt_disabled_outputs
    assert not (exp_dir / "ovwt_batchwise_feature_filtered").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_run_feature_filtered_ovwt_disabled_unfiltered_output_unaffected(
    run_feature_filtered_ovwt_disabled_outputs, batch_stem
):
    exp_dir, _ = run_feature_filtered_ovwt_disabled_outputs
    batch_dir = exp_dir / "ovwt_batchwise" / batch_stem
    assert (batch_dir / "results.parquet").exists()
    assert (batch_dir / "models.pkl").exists()


@pytest.fixture(scope="session")
def run_wtvwt_disabled_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_run_wtvwt_disabled_experiment")
    _write_batch(exp_dir / "input" / "batch1.parquet", seed=42)
    _write_batch(exp_dir / "input" / "batch2.parquet", seed=99)

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--input_dir",
            str(exp_dir),
            "--run_wtvwt",
            "false",
            *_NF_PARAMS,
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return exp_dir, result


def test_run_wtvwt_disabled_pipeline_exits_cleanly(run_wtvwt_disabled_outputs):
    _, result = run_wtvwt_disabled_outputs
    assert result.returncode == 0, result.stderr


def test_run_wtvwt_disabled_skips_wtvwt_batchwise_output(run_wtvwt_disabled_outputs):
    """params.run_wtvwt defaults to true (see test_pipeline_wtvwt_batchwise_outputs
    for the default-enabled case); false must skip WTVWT_BATCHWISE entirely."""
    exp_dir, _ = run_wtvwt_disabled_outputs
    assert not (exp_dir / "wtvwt_batchwise").exists()


def test_run_wtvwt_disabled_ovwt_output_unaffected(run_wtvwt_disabled_outputs):
    """run_wtvwt is independent of run_ovwt -- disabling it must not affect
    OVWT_BATCHWISE's output."""
    exp_dir, _ = run_wtvwt_disabled_outputs
    assert (exp_dir / "ovwt_batchwise" / "batch1" / "results.parquet").exists()


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


def test_ovwt_pipeline_check_barcodes_disabled_by_default(ovwt_pipeline_outputs):
    exp_dir, _ = ovwt_pipeline_outputs
    assert not (exp_dir / "check_barcodes").exists()


@pytest.fixture(scope="session")
def ovwt_check_barcodes_pipeline_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_ovwt_check_barcodes_experiment")
    _write_batch(exp_dir / "input" / "batch1.parquet", seed=42)
    _write_batch(exp_dir / "input" / "batch2.parquet", seed=99)

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--pipeline_mode",
            "ovwt",
            "--input_dir",
            str(exp_dir),
            "--run_check_barcodes",
            "true",
            "--single_cell_scores_split",
            "train",
            "--barcode_check_min_cells",
            "2",
            *_OVWT_NF_PARAMS,
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return exp_dir, result


def test_ovwt_check_barcodes_pipeline_exits_cleanly(
    ovwt_check_barcodes_pipeline_outputs,
):
    _, result = ovwt_check_barcodes_pipeline_outputs
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_check_barcodes_results_outputs(
    ovwt_check_barcodes_pipeline_outputs, batch_stem
):
    exp_dir, _ = ovwt_check_barcodes_pipeline_outputs
    assert (exp_dir / "check_barcodes" / batch_stem / "results.parquet").exists()


# ---------------------------------------------------------------------------
# params.yaml_config_dir — optional INPUT stage (workflows/fisseq.nf,
# workflows/ovwt.nf). Uses the lighter OvwtPipeline so this test isn't
# bottlenecked by the full feature-selection DAG; the thing under test is the
# INPUT -> QC_FILTER channel wiring, not downstream analysis.
# ---------------------------------------------------------------------------


def _write_input_config(path: Path, source_path: Path, **overrides) -> None:
    cfg = {"input_paths": [str(source_path)]}
    cfg.update(overrides)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def _run_config_dir_pipeline(
    exp_dir: Path, config_dir: Path, resume: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            # -ansi-log false forces Nextflow's plain-text per-process summary
            # (".../CHECK_BARCODES ... cached: N" style lines) instead of the
            # redrawing ANSI dashboard renderer. Nextflow's choice between the
            # two isn't a reliable function of stdout being a pipe (non-tty)
            # under subprocess capture -- it can still pick the ANSI renderer
            # depending on inherited TERM/environment -- and
            # test_batch_yaml_unrelated_key_change_does_not_bust_resume_cache
            # greps result.stdout for the literal "completed=0"/"cached=N"
            # text, which only exists in plain-text mode.
            "-ansi-log",
            "false",
            "--pipeline_mode",
            "ovwt",
            "--input_dir",
            str(exp_dir),
            "--yaml_config_dir",
            str(config_dir),
            *_OVWT_NF_PARAMS,
            *(["-resume"] if resume else []),
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


# ---------------------------------------------------------------------------
# Per-batch YAML parameter overrides (lib/BatchParams.groovy). Two
# config-driven batches per test, so both are YAML-controllable, using
# OvwtPipeline for the same "not bottlenecked by the full DAG" reason as the
# yaml_config_dir tests above.
# ---------------------------------------------------------------------------


def _two_batch_config_dir(tmp_path_factory, **batch1_overrides):
    """Set up an experiment dir with two config-driven batches; batch1 gets
    `**batch1_overrides` merged into its YAML, batch2 has no overrides."""
    exp_dir = tmp_path_factory.mktemp("nf_batch_override_experiment")
    raw_dir = tmp_path_factory.mktemp("nf_batch_override_raw")
    _write_batch(raw_dir / "batch1_source.parquet", seed=1)
    _write_batch(raw_dir / "batch2_source.parquet", seed=2)
    config_dir = exp_dir / "configs"
    config_dir.mkdir()
    _write_input_config(
        config_dir / "batch1.yaml",
        raw_dir / "batch1_source.parquet",
        **batch1_overrides,
    )
    _write_input_config(config_dir / "batch2.yaml", raw_dir / "batch2_source.parquet")
    return exp_dir, config_dir


def test_batch_yaml_numeric_override_takes_effect(tmp_path_factory):
    # batch1 overrides barcode_count_threshold well above every barcode's
    # cell count (6), so its filtered_cells.parquet ends up empty; batch2
    # keeps the pipeline-wide default (3, from _OVWT_NF_PARAMS), which every
    # barcode clears, keeping the full synthetic dataset.
    exp_dir, config_dir = _two_batch_config_dir(
        tmp_path_factory, barcode_count_threshold=1000
    )
    result = _run_config_dir_pipeline(exp_dir, config_dir)
    assert result.returncode == 0, result.stderr

    batch1_df = pl.read_parquet(
        exp_dir / "qc_filter" / "batch1" / "filtered_cells.parquet"
    )
    batch2_df = pl.read_parquet(
        exp_dir / "qc_filter" / "batch2" / "filtered_cells.parquet"
    )
    assert batch1_df.shape[0] == 0
    expected_cells = sum(n * c for _, n, c in _VARIANTS.values())
    assert batch2_df.shape[0] == expected_cells

    # The override must be logged clearly (batch name, key, default, override).
    assert "batch1" in result.stdout
    assert "barcode_count_threshold" in result.stdout


def test_batch_yaml_gating_override_takes_effect(tmp_path_factory):
    # Pipeline-wide default leaves run_check_barcodes off; batch1's YAML
    # turns it on for itself only. barcode_check_min_cells/
    # single_cell_scores_split come from the CLI defaults below (shared by
    # both batches) so CHECK_BARCODES has enough per-barcode samples to
    # compare, matching _CHECK_BARCODES_NF_PARAMS's rationale above.
    exp_dir, config_dir = _two_batch_config_dir(
        tmp_path_factory, run_check_barcodes=True
    )
    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--pipeline_mode",
            "ovwt",
            "--input_dir",
            str(exp_dir),
            "--yaml_config_dir",
            str(config_dir),
            *_OVWT_NF_PARAMS,
            "--barcode_check_min_cells",
            "2",
            "--single_cell_scores_split",
            "train",
        ],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert (exp_dir / "check_barcodes" / "batch1" / "results.parquet").exists()
    assert not (exp_dir / "check_barcodes" / "batch2").exists()


def test_batch_yaml_unknown_key_rejected(tmp_path_factory):
    exp_dir, config_dir = _two_batch_config_dir(tmp_path_factory, totally_bogus_param=1)
    result = _run_config_dir_pipeline(exp_dir, config_dir)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "unrecognized" in output
    assert "batch1" in output


def test_batch_yaml_missing_input_paths_rejected(tmp_path_factory):
    exp_dir = tmp_path_factory.mktemp("nf_missing_input_paths_experiment")
    config_dir = exp_dir / "configs"
    config_dir.mkdir()
    with open(config_dir / "batch1.yaml", "w") as f:
        yaml.safe_dump({"qc_n_variants": 5}, f)

    result = _run_config_dir_pipeline(exp_dir, config_dir)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "input_paths" in output
    assert "required" in output


def test_batch_yaml_pipeline_wide_key_rejected(tmp_path_factory):
    exp_dir, config_dir = _two_batch_config_dir(tmp_path_factory, run_global=False)
    result = _run_config_dir_pipeline(exp_dir, config_dir)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "pipeline-wide-only" in output
    # Distinct error path from the unknown-key case above.
    assert "unrecognized" not in output


def test_batch_yaml_unrelated_key_change_does_not_bust_resume_cache(tmp_path_factory):
    # This is the actual point of the per-batch override mechanism: a batch
    # process only receives the resolved val() scalars it consumes, never
    # the whole batch YAML/config map, so an unrelated key change in that
    # YAML must not invalidate -resume caching for tasks that don't consume
    # it. feature_select_min_correlation has zero process consumers in
    # OvwtPipeline (it's only wired into BLOCKLIST_BATCHWISE/_GLOBAL, part
    # of FisseqPipeline's feature-selection chain, which OvwtPipeline
    # doesn't have) -- so overriding it for batch1 should leave every task
    # for both batches cached on the next -resume run.
    exp_dir, config_dir = _two_batch_config_dir(tmp_path_factory)
    result1 = _run_config_dir_pipeline(exp_dir, config_dir)
    assert result1.returncode == 0, result1.stderr

    batch1_yaml_path = config_dir / "batch1.yaml"
    with open(batch1_yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg["feature_select_min_correlation"] = 0.9
    with open(batch1_yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)

    result2 = _run_config_dir_pipeline(exp_dir, config_dir, resume=True)
    assert result2.returncode == 0, result2.stderr
    # The override was still resolved and logged...
    assert "batch1" in result2.stdout
    assert "feature_select_min_correlation" in result2.stdout
    # ...but every task from both batches stayed cached -- nothing to
    # re-run, since no process consumes this key.
    assert "completed=0" in result2.stdout
    cached_match = re.search(r"cached=(\d+)", result2.stdout)
    assert cached_match is not None
    assert int(cached_match.group(1)) > 0
