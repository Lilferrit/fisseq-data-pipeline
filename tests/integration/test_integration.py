"""Integration tests for the FISSEQ Nextflow pipeline.

Runs the real pipeline on synthetic data via `nextflow run`, under
`-profile local` (no container -- the tests exercise the workflow wiring and
the Python stages, not the image).
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

_PROJECT_ROOT = Path(__file__).parents[2]

# Avoids a network round-trip (version check) on every `nextflow run` call.
_NF_ENV = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true"}

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

# 10 WT barcodes x 20 cells = 200 WT cells
# 5 A1A barcodes x 6 cells = 30 Synonymous cells  (A->A at position 1)
# 5 M1K barcodes x 6 cells = 30 Single Missense cells
# 5 M1K:downsampled-half barcodes x 6 cells = 30 tagged Single Missense cells,
# which must pool with the untagged M1K rows under meta_aa_changes == "M1K"
# once qcfilter.py's filter_columns strips the ":downsampled-half" tag.
#
# WT gets more cells per barcode than the other variants because OvWT's
# stratification key is (meta_barcode, is_wt): at 20 cells a WT barcode clears
# _MIN_STRATUM_SIZE and keeps its own stratum, while the 6-cell variant
# barcodes fall below it and collapse into the shared "rare|variant" bucket --
# which is exactly the graceful degradation that mechanism exists for.
_VARIANTS = {
    "WT": ("bc_wt_{i:02d}", 10, 20),
    "A1A": ("bc_syn_{i:02d}", 5, 6),
    "M1K": ("bc_mis_{i:02d}", 5, 6),
    "M1K:downsampled-half": ("bc_mis_tag_{i:02d}", 5, 6),
}

# GLOBAL_OVWT and GLOBAL_FEATURE_SELECT both fit a per-experiment Normalizer on
# that experiment's synonymous ("control") rows. A single control row makes std
# (ddof=1) undefined, nulling every feature -- so the channeled fixture needs
# more synonymous labels than the shared _VARIANTS' lone "A1A". Counts/prefix
# mirror "A1A" so the barcode/variant count thresholds are satisfied the same
# way.
_EXTRA_SYNONYMOUS_VARIANTS = {
    "A2A": ("bc_syn2_{i:02d}", 5, 6),
    "A3A": ("bc_syn3_{i:02d}", 5, 6),
}

_FEATURE_COLS = [
    "Cells_AreaShape_Area",
    "Cells_AreaShape_Perimeter",
    "Cells_Intensity_Mean",
    "Nuclei_AreaShape_Area",
    "Nuclei_Intensity_Max",
]

# Low thresholds so the small synthetic dataset passes every pipeline step.
_TEST_PARAM_OVERRIDES = {
    "barcode_count_threshold": 3,
    "variant_barcode_count_threshold": 3,
    "ovwt_min_cells": 25,
    "ovwt_n_folds": 3,
    "ovwt_downsample_wt": True,
    "feature_select_bootstrap_reps": 3,
}


def _write_batch(path: Path, seed: int = 42, variants: dict = None) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for variant, (bc_fmt, n_barcodes, cells_per_bc) in (variants or _VARIANTS).items():
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


def _write_params_file(exp_dir: Path, experiments: list[dict], **overrides) -> Path:
    """
    Build a test params file from the repo's real params.yaml.

    Starting from the shipped defaults rather than a hand-written dict means
    these tests also assert that params.yaml itself parses and carries every
    key the workflow reads.
    """
    with open(_PROJECT_ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)
    params.update(_TEST_PARAM_OVERRIDES)
    params.update(overrides)
    params["experiments"] = experiments
    path = exp_dir / "_test_params.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(params, f)
    return path


def _stage_experiment(
    raw_dir: Path, name: str, seed: int, variants: dict = None, **entry
) -> dict:
    """Write a raw batch parquet and return its `experiments:` entry."""
    source = raw_dir / f"{name}_source.parquet"
    _write_batch(source, seed=seed, variants=variants)
    return {"batch_stem": name, "input_paths": [str(source)], **entry}


def _run_pipeline(
    exp_dir: Path, experiments: list[dict], **overrides
) -> subprocess.CompletedProcess:
    params_path = _write_params_file(exp_dir, experiments, **overrides)
    return subprocess.run(
        [
            "nextflow",
            "run",
            str(_PROJECT_ROOT),
            # Plain-text per-process summary instead of the redrawing ANSI
            # dashboard, so stdout can be grepped.
            "-ansi-log",
            "false",
            "-profile",
            "local",
            "-params-file",
            str(params_path),
            "--pipeline_dir",
            str(exp_dir),
        ],
        cwd=exp_dir,
        env=_NF_ENV,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.fixture(scope="session", autouse=True)
def _require_nextflow():
    if shutil.which("nextflow") is None:
        pytest.skip("nextflow not on PATH")


# ---------------------------------------------------------------------------
# Session fixture -- the batchwise pipeline runs once, tests share the outputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipeline_outputs(tmp_path_factory):
    exp_dir = tmp_path_factory.mktemp("nf_experiment")
    raw_dir = tmp_path_factory.mktemp("nf_experiment_raw")
    experiments = [
        _stage_experiment(raw_dir, "batch1", seed=42),
        _stage_experiment(raw_dir, "batch2", seed=99),
    ]
    result = _run_pipeline(exp_dir, experiments)
    return exp_dir, result


# ---------------------------------------------------------------------------
# Pipeline exit / structure
# ---------------------------------------------------------------------------


def test_pipeline_exits_cleanly(pipeline_outputs):
    _, result = pipeline_outputs
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_input_stage_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "input" / f"{batch_stem}.parquet").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_qc_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    qc = exp_dir / "qc_filter" / batch_stem
    assert (qc / "filtered_cells.parquet").exists()
    assert (qc / "barcode_counts.parquet").exists()
    assert (qc / "variants_per_barcode.parquet").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_tagged_variant_pools_with_base(pipeline_outputs, batch_stem):
    """
    A raw aaChanges of 'M1K:downsampled-half' splits into
    meta_aa_changes == 'M1K' (pooled with the untagged M1K rows) and
    meta_variant_tag == 'downsampled-half'.
    """
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "qc_filter" / batch_stem / "filtered_cells.parquet")
    m1k = df.filter(pl.col("meta_aa_changes") == "M1K")
    # 5 barcodes x 6 cells for each of the tagged and untagged groups.
    assert m1k.shape[0] == 60
    assert set(m1k["meta_variant_tag"].to_list()) == {None, "downsampled-half"}
    assert "M1K:downsampled-half" not in df["meta_aa_changes"].to_list()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_normalization_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    assert (exp_dir / "normalization" / "cells" / f"{batch_stem}.parquet").exists()
    assert (
        exp_dir / "normalization" / "normalizers" / f"{batch_stem}.normalizer.parquet"
    ).exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_normalized_cells_wt_mean_near_zero(pipeline_outputs, batch_stem):
    """NORMALIZE fits on wildtype cells, so their post-fit mean sits at ~0."""
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "normalization" / "cells" / f"{batch_stem}.parquet")
    wt = df.filter(pl.col("meta_aa_changes") == "WT")
    for col in _FEATURE_COLS:
        assert abs(wt[col].mean()) < 1e-6


def test_no_global_dir_by_default(pipeline_outputs):
    """global_channels defaults to null, so no global stage runs at all."""
    exp_dir, _ = pipeline_outputs
    assert not (exp_dir / "global").exists()


# ---------------------------------------------------------------------------
# OVWT_BATCHWISE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_batchwise_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    d = exp_dir / "ovwt_batchwise" / batch_stem
    assert (d / "results.parquet").exists()
    assert (d / "cell_scores.parquet").exists()
    assert (d / "models.pkl").exists()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_results_schema(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "ovwt_batchwise" / batch_stem / "results.parquet")
    assert df.columns == [
        "meta_aa_changes",
        "auroc_pooled",
        "auroc_median_barcode",
        "meta_n_barcodes",
        "meta_n_cells",
    ]
    assert df.height > 0


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_aurocs_in_unit_interval(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(exp_dir / "ovwt_batchwise" / batch_stem / "results.parquet")
    for col in ("auroc_pooled", "auroc_median_barcode"):
        vals = df[col].drop_nulls()
        assert vals.min() >= 0.0
        assert vals.max() <= 1.0


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_cell_scores_every_cell_scored_once(pipeline_outputs, batch_stem):
    """
    K-fold CV partitions, so every cell carries exactly one out-of-fold score
    per variant it was scored against -- no nulls, no NaNs.
    """
    exp_dir, _ = pipeline_outputs
    df = pl.read_parquet(
        exp_dir / "ovwt_batchwise" / batch_stem / "cell_scores.parquet"
    )
    assert "score" in df.columns
    assert "meta_variant_scored_against" in df.columns
    assert df["score"].null_count() == 0
    assert df["score"].is_nan().sum() == 0
    # Metadata only -- no feature columns leak into cell_scores.
    assert all(c.startswith("meta_") or c == "score" for c in df.columns)


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_ovwt_models_are_per_fold(pipeline_outputs, batch_stem):
    import pickle

    exp_dir, _ = pipeline_outputs
    with open(exp_dir / "ovwt_batchwise" / batch_stem / "models.pkl", "rb") as f:
        models = pickle.load(f)
    assert models
    for fold_models in models.values():
        assert len(fold_models) == _TEST_PARAM_OVERRIDES["ovwt_n_folds"]


def test_no_removed_ovwt_artifacts(pipeline_outputs):
    """The old split-index and feature-importance outputs are gone."""
    exp_dir, _ = pipeline_outputs
    d = exp_dir / "ovwt_batchwise" / "batch1"
    for gone in (
        "test_index.parquet",
        "train_index.parquet",
        "feature_importance.parquet",
    ):
        assert not (d / gone).exists()


def test_removed_stages_produce_no_output(pipeline_outputs):
    exp_dir, _ = pipeline_outputs
    for gone in (
        "ovwt_batchwise_barcode_filtered",
        "ovwt_cellscores_batchwise",
        "check_barcodes",
        "barcode_blocklist",
        "wtvwt_batchwise",
        "wtvvariantpool_batchwise",
    ):
        assert not (exp_dir / gone).exists(), f"{gone} should no longer be produced"


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_feature_select_batchwise_outputs(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    fs = exp_dir / "feature_select_batchwise" / batch_stem
    assert (fs / "blocklist.parquet").exists()
    assert (fs / "output.parquet").exists()
    assert (fs / "aggregates").is_dir()


@pytest.mark.parametrize("batch_stem", ["batch1", "batch2"])
def test_feature_correlations_have_feature_ok_column(pipeline_outputs, batch_stem):
    exp_dir, _ = pipeline_outputs
    bl = pl.read_parquet(
        exp_dir / "feature_select_batchwise" / batch_stem / "blocklist.parquet"
    )
    assert "feature_ok" in bl.columns


def test_bootstrap_replicates_are_not_identical(pipeline_outputs):
    """
    Each bootstrap derives its seed as random_seed + bootstrap_idx. Collapsing
    those to one constant would make every replicate identical and silently
    destroy feature selection, so assert they actually differ.
    """
    exp_dir, _ = pipeline_outputs
    splits = exp_dir / "feature_select_batchwise" / "batch1" / "splits"
    halves = sorted(splits.glob("bootstrap_*/half1.parquet"))
    assert len(halves) == _TEST_PARAM_OVERRIDES["feature_select_bootstrap_reps"]
    seen = {pl.read_parquet(h).to_pandas().to_csv(index=False) for h in halves}
    assert len(seen) > 1, "every bootstrap split is identical -- seed offset lost"


# ---------------------------------------------------------------------------
# Global channels
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def global_channel_outputs(tmp_path_factory):
    exp_dir = tmp_path_factory.mktemp("nf_channels")
    raw_dir = tmp_path_factory.mktemp("nf_channels_raw")
    variants = {**_VARIANTS, **_EXTRA_SYNONYMOUS_VARIANTS}
    experiments = [
        # chan_a only
        _stage_experiment(
            raw_dir, "batch1", seed=42, variants=variants, global_channel="chan_a"
        ),
        # both channels -- membership is a list
        _stage_experiment(
            raw_dir,
            "batch2",
            seed=99,
            variants=variants,
            global_channel=["chan_a", "chan_b"],
        ),
        # chan_b only
        _stage_experiment(
            raw_dir, "batch3", seed=7, variants=variants, global_channel="chan_b"
        ),
        # deliberately in no channel -- must be excluded from both global stages
        _stage_experiment(raw_dir, "batch4", seed=13, variants=variants),
    ]
    result = _run_pipeline(exp_dir, experiments, global_channels=["chan_a", "chan_b"])
    return exp_dir, result


def test_channeled_pipeline_exits_cleanly(global_channel_outputs):
    _, result = global_channel_outputs
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("chan", ["chan_a", "chan_b"])
def test_channeled_global_ovwt_outputs(global_channel_outputs, chan):
    exp_dir, _ = global_channel_outputs
    out = (
        exp_dir / "global" / chan / "ovwt_distinguishability" / "global_scores.parquet"
    )
    assert out.exists()


@pytest.mark.parametrize("chan", ["chan_a", "chan_b"])
def test_channeled_global_ovwt_schema(global_channel_outputs, chan):
    exp_dir, _ = global_channel_outputs
    df = pl.read_parquet(
        exp_dir / "global" / chan / "ovwt_distinguishability" / "global_scores.parquet"
    )
    assert set(df.columns) == {
        "meta_aa_changes",
        "meta_median_auroc_pooled",
        "meta_median_auroc_median_barcode",
        "meta_num_experiments",
    }
    assert df.height > 0


@pytest.mark.parametrize("chan", ["chan_a", "chan_b"])
def test_channeled_global_ovwt_scoped_to_membership(global_channel_outputs, chan):
    """
    Each channel has exactly 2 member experiments, so no variant can report
    more than 2 contributing experiments -- batch4 (no channel) must not leak in.
    """
    exp_dir, _ = global_channel_outputs
    df = pl.read_parquet(
        exp_dir / "global" / chan / "ovwt_distinguishability" / "global_scores.parquet"
    )
    assert df["meta_num_experiments"].max() <= 2


@pytest.mark.parametrize("chan", ["chan_a", "chan_b"])
def test_channeled_global_feature_select_outputs(global_channel_outputs, chan):
    exp_dir, _ = global_channel_outputs
    fs = exp_dir / "global" / chan / "feature_select"
    assert (fs / "aggregate.parquet").exists()
    assert (fs / "blocklist.parquet").exists()


def test_unchanneled_experiment_still_runs_batchwise(global_channel_outputs):
    """batch4 belongs to no channel, but is processed batchwise as normal."""
    exp_dir, _ = global_channel_outputs
    assert (exp_dir / "ovwt_batchwise" / "batch4" / "results.parquet").exists()
    assert (exp_dir / "feature_select_batchwise" / "batch4" / "output.parquet").exists()


def test_no_channel_subdir_for_undeclared_channel(global_channel_outputs):
    exp_dir, _ = global_channel_outputs
    assert sorted(p.name for p in (exp_dir / "global").iterdir()) == [
        "chan_a",
        "chan_b",
    ]


# ---------------------------------------------------------------------------
# Run gates
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gates_off_outputs(tmp_path_factory):
    exp_dir = tmp_path_factory.mktemp("nf_gates_off")
    raw_dir = tmp_path_factory.mktemp("nf_gates_off_raw")
    experiments = [_stage_experiment(raw_dir, "batch1", seed=42)]
    result = _run_pipeline(
        exp_dir, experiments, run_ovwt=False, run_feature_selection=False
    )
    return exp_dir, result


def test_gates_off_pipeline_exits_cleanly(gates_off_outputs):
    _, result = gates_off_outputs
    assert result.returncode == 0, result.stderr


def test_run_ovwt_false_skips_ovwt(gates_off_outputs):
    exp_dir, _ = gates_off_outputs
    assert not (exp_dir / "ovwt_batchwise").exists()


def test_run_feature_selection_false_skips_feature_select(gates_off_outputs):
    exp_dir, _ = gates_off_outputs
    assert not (exp_dir / "feature_select_batchwise").exists()


def test_gates_off_upstream_stages_still_run(gates_off_outputs):
    exp_dir, _ = gates_off_outputs
    assert (exp_dir / "input" / "batch1.parquet").exists()
    assert (exp_dir / "normalization" / "cells" / "batch1.parquet").exists()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_random_seed_reproduces_ovwt_results(tmp_path_factory):
    """The release's core promise: one seed, and a rerun matches."""
    raw_dir = tmp_path_factory.mktemp("nf_seed_raw")
    experiments = [_stage_experiment(raw_dir, "batch1", seed=42)]

    outs = []
    for i in range(2):
        exp_dir = tmp_path_factory.mktemp(f"nf_seed_{i}")
        result = _run_pipeline(
            exp_dir, experiments, random_seed=1234, run_feature_selection=False
        )
        assert result.returncode == 0, result.stderr
        outs.append(
            pl.read_parquet(
                exp_dir / "ovwt_batchwise" / "batch1" / "results.parquet"
            ).sort("meta_aa_changes")
        )
    assert outs[0].equals(outs[1])


# ---------------------------------------------------------------------------
# experiments: schema validation
# ---------------------------------------------------------------------------


def _expect_failure(tmp_path_factory, name, experiments, **overrides):
    exp_dir = tmp_path_factory.mktemp(name)
    result = _run_pipeline(exp_dir, experiments, **overrides)
    assert result.returncode != 0
    return result.stdout + result.stderr


def test_empty_experiments_rejected(tmp_path_factory):
    out = _expect_failure(tmp_path_factory, "nf_empty", [])
    assert "must be a non-empty list" in out


def test_missing_batch_stem_rejected(tmp_path_factory):
    out = _expect_failure(
        tmp_path_factory, "nf_no_stem", [{"input_paths": ["/nonexistent.parquet"]}]
    )
    assert "batch_stem" in out


def test_missing_input_paths_rejected(tmp_path_factory):
    out = _expect_failure(tmp_path_factory, "nf_no_inputs", [{"batch_stem": "b1"}])
    assert "input_paths" in out


def test_duplicate_batch_stem_rejected(tmp_path_factory, tmp_path):
    src = tmp_path / "s.parquet"
    _write_batch(src)
    out = _expect_failure(
        tmp_path_factory,
        "nf_dupe",
        [
            {"batch_stem": "b1", "input_paths": [str(src)]},
            {"batch_stem": "b1", "input_paths": [str(src)]},
        ],
    )
    assert "duplicate batch_stem" in out


def test_unknown_experiment_key_rejected(tmp_path_factory, tmp_path):
    """A pipeline-wide param set per experiment must fail loudly, not silently."""
    src = tmp_path / "s.parquet"
    _write_batch(src)
    out = _expect_failure(
        tmp_path_factory,
        "nf_unknown_key",
        [{"batch_stem": "b1", "input_paths": [str(src)], "ovwt_min_cells": 10}],
    )
    assert "unrecognized key" in out
    assert "ovwt_min_cells" in out
