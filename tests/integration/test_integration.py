"""Integration tests for the FISSEQ Nextflow pipeline."""

import json
import os
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

_PROJECT_ROOT = Path(__file__).parents[2]

# Avoids a network round-trip (version check) on every `nextflow run` call.
_NF_ENV = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true"}


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


def _write_input_config(path: Path, source_path: Path, **overrides) -> None:
    cfg = {"input_paths": [str(source_path)]}
    cfg.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def _stage_batch(
    exp_dir: Path, raw_dir: Path, name: str, seed: int, **overrides
) -> None:
    """Write a raw batch parquet under raw_dir and its mandatory YAML config
    under exp_dir/configs/ -- YAML configs are the only way to declare a
    batch now that direct input-directory scanning has been removed."""
    source = raw_dir / f"{name}_source.parquet"
    _write_batch(source, seed=seed)
    _write_input_config(exp_dir / "configs" / f"{name}.yaml", source, **overrides)


def _params_file_args(exp_dir: Path, global_groups) -> list:
    """List-valued params (e.g. global_groups) can't be expressed as a bare
    CLI flag -- `--global_groups foo,bar` arrives as the single String
    "foo,bar", and Groovy's `as List<String>` on a String splits it into
    individual characters, not comma-separated elements (see
    docs/configuration.md). A `-params-file` JSON document is parsed
    properly instead."""
    if global_groups is None:
        return []
    params_path = exp_dir / "_test_params.json"
    with open(params_path, "w") as f:
        json.dump({"global_groups": list(global_groups)}, f)
    return ["-params-file", str(params_path)]


def _run_pipeline(exp_dir: Path, global_groups=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "-ansi-log",
            "false",
            "--pipeline_dir",
            str(exp_dir),
            *_params_file_args(exp_dir, global_groups),
            *_NF_PARAMS,
        ],
        cwd=exp_dir,
        env=_NF_ENV,
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
            "-ansi-log",
            "false",
            "--pipeline_dir",
            str(exp_dir),
            *_CHECK_BARCODES_NF_PARAMS,
        ],
        cwd=exp_dir,
        env=_NF_ENV,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _run_ovwt_pipeline(
    exp_dir: Path, resume: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            # -ansi-log false forces Nextflow's plain-text per-process summary
            # instead of the redrawing ANSI dashboard renderer, which the
            # per-batch-override tests below rely on when grepping stdout for
            # override log lines / "cached=" resume summaries.
            "-ansi-log",
            "false",
            "--pipeline_mode",
            "ovwt",
            "--pipeline_dir",
            str(exp_dir),
            *_OVWT_NF_PARAMS,
            *(["-resume"] if resume else []),
        ],
        cwd=exp_dir,
        env=_NF_ENV,
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
    raw_dir = tmp_path_factory.mktemp("nf_experiment_raw")
    _stage_batch(exp_dir, raw_dir, "batch1", seed=42)
    _stage_batch(exp_dir, raw_dir, "batch2", seed=99)

    result = _run_pipeline(exp_dir)
    return exp_dir, result


# ---------------------------------------------------------------------------
# Pipeline exit / structure tests
# ---------------------------------------------------------------------------


def test_pipeline_exits_cleanly(pipeline_outputs):
    _, result = pipeline_outputs
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_pipeline_input_stage_outputs(pipeline_outputs, batch_stem):
    """Every batch is declared via a mandatory YAML config now -- INPUT
    always runs, unconditionally, for every batch."""
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "input" / f"{batch_stem}.parquet").exists()


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


def test_pipeline_no_global_dir_by_default(pipeline_outputs):
    """params.global_groups defaults to null -- no global processes run and
    no global/ directory is produced at all unless a group is active."""
    exp_dir, _ = pipeline_outputs
    assert not (exp_dir / "global").exists()


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
# Global groups (params.global_groups / each batch's YAML global_group key).
# Four batches: batch1 in group "siteA" only, batch2 in both "siteA" and
# "siteB", batch3 in "siteB" only, batch4 in no group at all -- exercises
# per-group scoping (overlapping and non-overlapping membership) and
# exclusion of ungrouped batches, all in one run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def grouped_global_pipeline_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_grouped_experiment")
    raw_dir = tmp_path_factory.mktemp("nf_grouped_raw")
    _stage_batch(exp_dir, raw_dir, "batch1", seed=1, global_group="siteA")
    _stage_batch(exp_dir, raw_dir, "batch2", seed=2, global_group=["siteA", "siteB"])
    _stage_batch(exp_dir, raw_dir, "batch3", seed=3, global_group="siteB")
    _stage_batch(exp_dir, raw_dir, "batch4", seed=4)  # no global_group at all

    result = _run_pipeline(exp_dir, global_groups=["siteA", "siteB"])
    return exp_dir, result


def test_grouped_pipeline_exits_cleanly(grouped_global_pipeline_outputs):
    _, result = grouped_global_pipeline_outputs
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("group", ["siteA", "siteB"])
def test_grouped_batchvsbatch_outputs(grouped_global_pipeline_outputs, group):
    exp_dir, _ = grouped_global_pipeline_outputs
    assert (
        exp_dir / "global" / group / "batchvsbatch" / "pre" / "results.parquet"
    ).exists()
    assert (
        exp_dir / "global" / group / "batchvsbatch" / "post" / "results.parquet"
    ).exists()


@pytest.mark.parametrize("stage", ["pre", "post"])
def test_batchvsbatch_has_expected_columns(grouped_global_pipeline_outputs, stage):
    exp_dir, _ = grouped_global_pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "global" / "siteA" / "batchvsbatch" / stage / "results.parquet"
    )
    expected = {"variant", "batch", "auroc", "mw_pvalue", "n_batch_cells", "n_cells"}
    assert expected.issubset(set(df.columns))
    assert len(df) > 0


@pytest.mark.parametrize("group", ["siteA", "siteB"])
def test_grouped_ovwt_global_outputs(grouped_global_pipeline_outputs, group):
    exp_dir, _ = grouped_global_pipeline_outputs
    assert (exp_dir / "global" / group / "ovwt_global" / "results.parquet").exists()
    assert (exp_dir / "global" / group / "ovwt_global" / "models.pkl").exists()


@pytest.mark.parametrize("group", ["siteA", "siteB"])
def test_grouped_feature_select_global_outputs(grouped_global_pipeline_outputs, group):
    exp_dir, _ = grouped_global_pipeline_outputs
    assert (
        exp_dir / "global" / group / "feature_select" / "aggregate.parquet"
    ).exists()
    assert (
        exp_dir / "global" / group / "feature_select" / "blocklist.parquet"
    ).exists()


@pytest.mark.parametrize(
    "group,expected_batches",
    [("siteA", {"batch1", "batch2"}), ("siteB", {"batch2", "batch3"})],
)
def test_grouped_batchvsbatch_scoped_to_group_membership(
    grouped_global_pipeline_outputs, group, expected_batches
):
    exp_dir, _ = grouped_global_pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "global" / group / "batchvsbatch" / "post" / "results.parquet"
    )
    assert set(df["batch"].unique().to_list()) == expected_batches


@pytest.mark.parametrize("group", ["siteA", "siteB"])
def test_grouped_ovwt_global_scoped_to_group_membership(
    grouped_global_pipeline_outputs, group
):
    """Each group has exactly 2 member batches -- meta_batch_num_unique must
    reflect only that group's batches, not all 4 batches in the run."""
    exp_dir, _ = grouped_global_pipeline_outputs
    df = pl.read_parquet(exp_dir / "global" / group / "ovwt_global" / "results.parquet")
    assert (df["meta_batch_num_unique"] == 2).all()


@pytest.mark.parametrize("group", ["siteA", "siteB"])
def test_grouped_feature_select_global_scoped_to_group_membership(
    grouped_global_pipeline_outputs, group
):
    """GLOBAL_FEATURE_SELECT's aggregate has no per-variant batch metadata
    (unlike the BATCHWISE finalize output) -- verify scoping instead via the
    blocklist's n_batches column, which counts how many member batches'
    blocklists contributed to each feature's global decision. Each group has
    exactly 2 member batches."""
    exp_dir, _ = grouped_global_pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "global" / group / "feature_select" / "blocklist.parquet"
    )
    assert (df["n_batches"] == 2).all()


def test_grouped_stage_group_normalization_cells_scoped(
    grouped_global_pipeline_outputs,
):
    """Regression test for the STAGE_GROUP_CELLS staging mechanism itself:
    each group's normalization_cells/ directory must contain exactly that
    group's member batches, nothing more."""
    exp_dir, _ = grouped_global_pipeline_outputs
    site_a = {
        p.stem
        for p in (exp_dir / "global" / "siteA" / "normalization_cells").glob(
            "*.parquet"
        )
    }
    site_b = {
        p.stem
        for p in (exp_dir / "global" / "siteB" / "normalization_cells").glob(
            "*.parquet"
        )
    }
    assert site_a == {"batch1", "batch2"}
    assert site_b == {"batch2", "batch3"}


def test_grouped_batch_with_no_group_excluded_from_global_only(
    grouped_global_pipeline_outputs,
):
    """batch4 has no global_group key -- it must never appear in any group's
    global output, even though params.global_groups is non-empty, but it is
    still fully processed batchwise like every other batch."""
    exp_dir, _ = grouped_global_pipeline_outputs
    assert (exp_dir / "qc_filter" / "batch4" / "filtered_cells.parquet").exists()
    for group in ("siteA", "siteB"):
        df = pl.read_parquet(
            exp_dir / "global" / group / "batchvsbatch" / "post" / "results.parquet"
        )
        assert "batch4" not in df["batch"].unique().to_list()


# ---------------------------------------------------------------------------
# run_single_cell_scores / run_check_barcodes — session fixture and tests
# (FisseqPipeline, params.run_check_barcodes=true implying run_single_cell_scores)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def check_barcodes_pipeline_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_check_barcodes_experiment")
    raw_dir = tmp_path_factory.mktemp("nf_check_barcodes_raw")
    _stage_batch(exp_dir, raw_dir, "batch1", seed=42)
    _stage_batch(exp_dir, raw_dir, "batch2", seed=99)

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
def disabled_toggles_pipeline_outputs(tmp_path_factory):
    """run_barcode_filtered_ovwt, run_feature_filtered_ovwt, and run_wtvwt each
    gate their own unrelated branch of the DAG (BARCODE_BLOCKLIST/
    OVWT_BATCHWISE_BARCODE_FILTERED, OVWT_BATCHWISE_FEATURE_FILTERED, and
    WTVWT_BATCHWISE respectively) with no interaction between them, so one run
    with all three disabled can verify each toggle's effect independently --
    no need for three separate `nextflow run` invocations. Uses
    _CHECK_BARCODES_NF_PARAMS (run_check_barcodes=true) so
    run_barcode_filtered_ovwt=false is actually exercised (it only takes
    effect when run_check_barcodes is also true). Also leaves
    params.global_groups unset (the default), so this doubles as the
    "no groups active" case for the no-global-dir assertion below."""
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_disabled_toggles_experiment")
    raw_dir = tmp_path_factory.mktemp("nf_disabled_toggles_raw")
    _stage_batch(exp_dir, raw_dir, "batch1", seed=42)
    _stage_batch(exp_dir, raw_dir, "batch2", seed=99)

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "-ansi-log",
            "false",
            "--pipeline_dir",
            str(exp_dir),
            "--run_barcode_filtered_ovwt",
            "false",
            "--run_feature_filtered_ovwt",
            "false",
            "--run_wtvwt",
            "false",
            *_CHECK_BARCODES_NF_PARAMS,
        ],
        cwd=exp_dir,
        env=_NF_ENV,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return exp_dir, result


def test_disabled_toggles_pipeline_exits_cleanly(disabled_toggles_pipeline_outputs):
    _, result = disabled_toggles_pipeline_outputs
    assert result.returncode == 0, result.stderr


def test_run_barcode_filtered_ovwt_disabled_skips_barcode_blocklist_and_filtered_output(
    disabled_toggles_pipeline_outputs,
):
    """params.run_barcode_filtered_ovwt defaults to true (see
    test_barcode_blocklist_outputs / test_pipeline_ovwt_batchwise_barcode_filtered_outputs
    for the default-enabled case); false must skip both BARCODE_BLOCKLIST and
    OVWT_BATCHWISE_BARCODE_FILTERED entirely, while CHECK_BARCODES (set
    explicitly via _CHECK_BARCODES_NF_PARAMS) is unaffected."""
    exp_dir, _ = disabled_toggles_pipeline_outputs
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
    raw_dir = tmp_path_factory.mktemp("nf_invalid_source_raw")
    _stage_batch(exp_dir, raw_dir, "batch1", seed=42)

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--pipeline_dir",
            str(exp_dir),
            "--run_single_cell_scores",
            "true",
            "--single_cell_scores_split",
            "bogus",
            *_NF_PARAMS,
        ],
        cwd=exp_dir,
        env=_NF_ENV,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode != 0
    assert "single_cell_scores_split" in (result.stdout + result.stderr)


def test_run_feature_filtered_ovwt_disabled_skips_filtered_output(
    disabled_toggles_pipeline_outputs,
):
    """params.run_feature_filtered_ovwt defaults to true (see
    test_pipeline_ovwt_batchwise_feature_filtered_outputs for the
    default-enabled case); false must skip OVWT_BATCHWISE_FEATURE_FILTERED
    entirely."""
    exp_dir, _ = disabled_toggles_pipeline_outputs
    assert not (exp_dir / "ovwt_batchwise_feature_filtered").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_run_feature_filtered_ovwt_disabled_unfiltered_output_unaffected(
    disabled_toggles_pipeline_outputs, batch_stem
):
    exp_dir, _ = disabled_toggles_pipeline_outputs
    batch_dir = exp_dir / "ovwt_batchwise" / batch_stem
    assert (batch_dir / "results.parquet").exists()
    assert (batch_dir / "models.pkl").exists()


def test_run_wtvwt_disabled_skips_wtvwt_batchwise_output(
    disabled_toggles_pipeline_outputs,
):
    """params.run_wtvwt defaults to true (see test_pipeline_wtvwt_batchwise_outputs
    for the default-enabled case); false must skip WTVWT_BATCHWISE entirely."""
    exp_dir, _ = disabled_toggles_pipeline_outputs
    assert not (exp_dir / "wtvwt_batchwise").exists()


def test_run_wtvwt_disabled_ovwt_output_unaffected(disabled_toggles_pipeline_outputs):
    """run_wtvwt is independent of run_ovwt -- disabling it must not affect
    OVWT_BATCHWISE's output."""
    exp_dir, _ = disabled_toggles_pipeline_outputs
    assert (exp_dir / "ovwt_batchwise" / "batch1" / "results.parquet").exists()


def test_disabled_toggles_pipeline_no_global_dir(disabled_toggles_pipeline_outputs):
    """params.global_groups is left unset here (the default) -- no global/
    directory should exist even on a run that otherwise exercises many
    other toggles."""
    exp_dir, _ = disabled_toggles_pipeline_outputs
    assert not (exp_dir / "global").exists()


# ---------------------------------------------------------------------------
# OvwtPipeline (ovwt.nf) — session fixture and tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ovwt_pipeline_outputs(tmp_path_factory):
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")

    exp_dir = tmp_path_factory.mktemp("nf_ovwt_experiment")
    raw_dir = tmp_path_factory.mktemp("nf_ovwt_raw")
    _stage_batch(exp_dir, raw_dir, "batch1", seed=42)
    _stage_batch(exp_dir, raw_dir, "batch2", seed=99)

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
    raw_dir = tmp_path_factory.mktemp("nf_ovwt_check_barcodes_raw")
    _stage_batch(exp_dir, raw_dir, "batch1", seed=42)
    _stage_batch(exp_dir, raw_dir, "batch2", seed=99)

    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--pipeline_mode",
            "ovwt",
            "--pipeline_dir",
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
        env=_NF_ENV,
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
# Per-batch YAML parameter overrides (lib/BatchParams.groovy). Two
# config-driven batches per test, using OvwtPipeline since it isn't
# bottlenecked by the full feature-selection DAG.
# ---------------------------------------------------------------------------


def _two_batch_config_dir(tmp_path_factory, **batch1_overrides) -> Path:
    """Set up an experiment dir with two config-driven batches; batch1 gets
    `**batch1_overrides` merged into its YAML, batch2 has no overrides."""
    exp_dir = tmp_path_factory.mktemp("nf_batch_override_experiment")
    raw_dir = tmp_path_factory.mktemp("nf_batch_override_raw")
    _stage_batch(exp_dir, raw_dir, "batch1", seed=1, **batch1_overrides)
    _stage_batch(exp_dir, raw_dir, "batch2", seed=2)
    return exp_dir


def test_batch_yaml_numeric_override_takes_effect(tmp_path_factory):
    # batch1 overrides barcode_count_threshold well above every barcode's
    # cell count (6), so its filtered_cells.parquet ends up empty; batch2
    # keeps the pipeline-wide default (3, from _OVWT_NF_PARAMS), which every
    # barcode clears, keeping the full synthetic dataset.
    exp_dir = _two_batch_config_dir(tmp_path_factory, barcode_count_threshold=1000)
    result = _run_ovwt_pipeline(exp_dir)
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
    exp_dir = _two_batch_config_dir(tmp_path_factory, run_check_barcodes=True)
    result = subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            "--pipeline_mode",
            "ovwt",
            "--pipeline_dir",
            str(exp_dir),
            *_OVWT_NF_PARAMS,
            "--barcode_check_min_cells",
            "2",
            "--single_cell_scores_split",
            "train",
        ],
        cwd=exp_dir,
        env=_NF_ENV,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert (exp_dir / "check_barcodes" / "batch1" / "results.parquet").exists()
    assert not (exp_dir / "check_barcodes" / "batch2").exists()


def test_batch_yaml_unknown_key_rejected(tmp_path_factory):
    exp_dir = _two_batch_config_dir(tmp_path_factory, totally_bogus_param=1)
    result = _run_ovwt_pipeline(exp_dir)
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

    result = _run_ovwt_pipeline(exp_dir)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "input_paths" in output
    assert "required" in output


def test_batch_yaml_pipeline_wide_key_rejected(tmp_path_factory):
    """global_groups (like the old run_global before it) is pipeline-wide-only
    -- a batch YAML that tries to set it gets a clear rejection, distinct
    from the unrecognized-key case below."""
    exp_dir = _two_batch_config_dir(tmp_path_factory, global_groups=["siteA"])
    result = _run_ovwt_pipeline(exp_dir)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "pipeline-wide-only" in output
    # Distinct error path from the unknown-key case above.
    assert "unrecognized" not in output


def test_batch_yaml_global_group_accepted(tmp_path_factory):
    """Unlike global_groups (pipeline-wide-only, plural), global_group
    (singular, batch-YAML-only) must be accepted -- not rejected as
    unrecognized or pipeline-wide-only."""
    exp_dir = _two_batch_config_dir(tmp_path_factory, global_group="siteA")
    result = _run_ovwt_pipeline(exp_dir)
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "unrecognized" not in output
    assert "pipeline-wide-only" not in output


def test_batch_yaml_global_group_list_accepted(tmp_path_factory):
    """global_group may also be a list of strings."""
    exp_dir = _two_batch_config_dir(tmp_path_factory, global_group=["siteA", "siteB"])
    result = _run_ovwt_pipeline(exp_dir)
    assert result.returncode == 0, result.stderr
