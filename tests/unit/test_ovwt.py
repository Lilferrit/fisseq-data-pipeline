"""Unit tests for the k-fold cross-validated OVWT_BATCHWISE stage.

Fixture sizing note: the inner 80/20 train/calibration split
(``split_indices_stratified``) runs *inside* each outer fold, so a
``(barcode, is_wt)`` stratum wants comfortably more than ``n_folds`` members
for the folds to carry a real signal. Undersized fixtures used to send every
variant into ``ovwt_batchwise``'s per-variant ``except`` branch, and the test
would then pass against empty output while asserting nothing. These fixtures
use ``n_folds=3`` with ~15 cells per barcode for that reason;
``test_normal_variant_survives`` guards the failure mode directly.
"""

import pathlib
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.ovwt import (
    _MIN_STRATUM_SIZE,
    OvwtConfig,
    _stratification_key,
    downsample_wildtype,
    filter_min_cells,
    ovwt_batchwise,
    predict_binary,
)
from fisseq_data_pipeline.utils.xgbparams import XGBoostConfig

LABEL = "meta_aa_changes"
WT = "WT"


def _cells(
    variants: dict[str, int] | None = None,
    wt_barcodes: int = 3,
    cells_per_wt_barcode: int = 15,
    barcodes_per_variant: int = 2,
    seed: int = 0,
) -> pl.DataFrame:
    """
    Cell-level frame with a real signal: variants sit away from WT on
    Intensity_Mean, so the classifier has something to find.

    ``variants`` maps a variant label to its per-barcode cell count.
    """
    variants = variants or {"M1K": 15, "A1A": 15}
    rng = np.random.default_rng(seed)
    rows = {LABEL: [], "meta_barcode": [], "Intensity_Mean": [], "Texture_Var": []}

    for b in range(wt_barcodes):
        n = cells_per_wt_barcode
        rows[LABEL] += [WT] * n
        rows["meta_barcode"] += [f"wt_bc{b}"] * n
        rows["Intensity_Mean"] += rng.normal(0.0, 0.3, n).tolist()
        rows["Texture_Var"] += rng.random(n).tolist()

    for offset, (variant, per_barcode) in enumerate(variants.items(), start=1):
        for b in range(barcodes_per_variant):
            rows[LABEL] += [variant] * per_barcode
            rows["meta_barcode"] += [f"{variant}_bc{b}"] * per_barcode
            rows["Intensity_Mean"] += rng.normal(
                3.0 * offset, 0.3, per_barcode
            ).tolist()
            rows["Texture_Var"] += rng.random(per_barcode).tolist()

    return pl.DataFrame(rows)


def _cfg(**overrides) -> OvwtConfig:
    base = dict(
        output_dir="unused",
        input_file="unused",
        label_column=LABEL,
        wt_label=WT,
        n_folds=3,
        calibrate=True,
        min_cells=None,
        downsample_wt=False,
        random_seed=0,
        xgboost=XGBoostConfig(),
    )
    base.update(overrides)
    return OvwtConfig(**base)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = OvwtConfig(output_dir="o", input_file="i")
    assert cfg.wt_label == "WT"
    assert cfg.n_folds == 5
    assert cfg.calibrate is True
    assert cfg.min_cells == 250
    assert cfg.downsample_wt is True
    assert cfg.label_column == "meta_aa_changes"
    assert isinstance(cfg.xgboost, XGBoostConfig)


def test_config_has_no_stage_local_seed():
    """AppConfig.random_seed is the only seed -- see config/app.py."""
    fields = {f for f in OvwtConfig.__dataclass_fields__}
    assert "random_state" not in fields
    assert "seed" not in fields
    assert OvwtConfig(output_dir="o", input_file="i").random_seed == 0


# ---------------------------------------------------------------------------
# filter_min_cells
# ---------------------------------------------------------------------------


def test_filter_min_cells_none_is_no_op():
    df = _cells()
    assert filter_min_cells(df, LABEL, WT, None).equals(df)


def test_filter_min_cells_drops_small_variants():
    df = _cells(variants={"BIG": 15, "SMALL": 2})
    out = filter_min_cells(df, LABEL, WT, min_cells=20)
    labels = set(out.get_column(LABEL).unique().to_list())
    assert "BIG" in labels
    assert "SMALL" not in labels


def test_filter_min_cells_always_keeps_wildtype():
    # WT has 3 barcodes x 15 cells = 45, far below this threshold
    df = _cells(variants={"BIG": 15})
    out = filter_min_cells(df, LABEL, WT, min_cells=1000)
    assert WT in out.get_column(LABEL).unique().to_list()


# ---------------------------------------------------------------------------
# downsample_wildtype
# ---------------------------------------------------------------------------


def test_downsample_wildtype_no_op_when_already_small():
    df = _cells(wt_barcodes=1, cells_per_wt_barcode=5, variants={"M1K": 30})
    assert len(downsample_wildtype(df, LABEL, WT, seed=0)) == len(df)


def test_downsample_wildtype_reduces_to_largest_variant_group():
    # WT: 4 x 25 = 100 cells; M1K: 2 x 10 = 20 cells
    df = _cells(
        wt_barcodes=4,
        cells_per_wt_barcode=25,
        variants={"M1K": 10},
        barcodes_per_variant=2,
    )
    out = downsample_wildtype(df, LABEL, WT, seed=0)
    n_wt = len(out.filter(pl.col(LABEL) == WT))
    # Per-barcode rounding can miss the target by up to (n wt barcodes).
    assert abs(n_wt - 20) <= 4


def test_downsample_wildtype_preserves_barcode_proportions():
    """The point of the barcode-proportional draw -- a uniform one would not."""
    rng = np.random.default_rng(0)
    # Deliberately lopsided WT barcodes: 80 / 16 / 4
    rows = {LABEL: [], "meta_barcode": [], "Intensity_Mean": [], "Texture_Var": []}
    for bc, n in [("wt_a", 80), ("wt_b", 16), ("wt_c", 4)]:
        rows[LABEL] += [WT] * n
        rows["meta_barcode"] += [bc] * n
        rows["Intensity_Mean"] += rng.random(n).tolist()
        rows["Texture_Var"] += rng.random(n).tolist()
    rows[LABEL] += ["M1K"] * 25
    rows["meta_barcode"] += ["m1k_bc0"] * 25
    rows["Intensity_Mean"] += rng.random(25).tolist()
    rows["Texture_Var"] += rng.random(25).tolist()
    df = pl.DataFrame(rows)

    out = downsample_wildtype(df, LABEL, WT, seed=0)
    kept = (
        out.filter(pl.col(LABEL) == WT)
        .group_by("meta_barcode")
        .len()
        .sort("meta_barcode")
    )
    counts = dict(zip(kept.get_column("meta_barcode"), kept.get_column("len")))
    # 25/100 of each barcode: 20 / 4 / 1
    assert counts["wt_a"] == 20
    assert counts["wt_b"] == 4
    assert counts["wt_c"] == 1


def test_downsample_wildtype_reproducible_with_same_seed():
    df = _cells(wt_barcodes=4, cells_per_wt_barcode=25, variants={"M1K": 10})
    a = downsample_wildtype(df, LABEL, WT, seed=3)
    b = downsample_wildtype(df, LABEL, WT, seed=3)
    assert a.equals(b)


def test_downsample_wildtype_no_variants_is_no_op():
    df = _cells(variants={}).filter(pl.col(LABEL) == WT)
    assert len(downsample_wildtype(df, LABEL, WT, seed=0)) == len(df)


# ---------------------------------------------------------------------------
# _stratification_key
# ---------------------------------------------------------------------------


def test_stratification_key_keeps_common_strata_distinct():
    barcodes = np.array(["a"] * _MIN_STRATUM_SIZE + ["b"] * _MIN_STRATUM_SIZE)
    is_wt = np.array([True] * _MIN_STRATUM_SIZE + [False] * _MIN_STRATUM_SIZE)
    keys = _stratification_key(barcodes, is_wt)
    assert set(keys) == {"a|wt", "b|variant"}


def test_stratification_key_collapses_rare_strata():
    barcodes = np.array(["a"] * _MIN_STRATUM_SIZE + ["rareone"])
    is_wt = np.array([True] * _MIN_STRATUM_SIZE + [True])
    keys = _stratification_key(barcodes, is_wt)
    assert keys[-1] == "rare|wt"
    assert set(keys[:-1]) == {"a|wt"}


def test_stratification_key_never_merges_across_wt_boundary():
    """A rare WT barcode and a rare variant barcode must not share a bucket."""
    barcodes = np.array(["rare_wt", "rare_var"])
    is_wt = np.array([True, False])
    keys = _stratification_key(barcodes, is_wt)
    assert keys[0] == "rare|wt"
    assert keys[1] == "rare|variant"
    assert keys[0] != keys[1]


# ---------------------------------------------------------------------------
# ovwt_batchwise
# ---------------------------------------------------------------------------


def test_ovwt_batchwise_output_columns():
    results, cell_scores, models = ovwt_batchwise(_cells().lazy(), _cfg())
    assert results.columns == [
        LABEL,
        "auroc_pooled",
        "auroc_median_barcode",
        "meta_n_barcodes",
        "meta_n_cells",
    ]
    assert "score" in cell_scores.columns
    assert "meta_variant_scored_against" in cell_scores.columns
    # cell_scores carries metadata only -- no feature columns leak through
    assert all(c.startswith("meta_") or c == "score" for c in cell_scores.columns)
    assert set(models) == {"M1K", "A1A"}


def test_normal_variant_survives():
    """Guards the fixture-too-small failure mode described in the module docstring."""
    results, _, models = ovwt_batchwise(_cells().lazy(), _cfg())
    assert len(results) == 2
    assert models


def test_ovwt_batchwise_no_nans_in_oof_scores():
    """StratifiedKFold partitions, so every cell is scored exactly once."""
    _, cell_scores, _ = ovwt_batchwise(_cells().lazy(), _cfg())
    assert cell_scores.get_column("score").is_nan().sum() == 0
    assert cell_scores.get_column("score").null_count() == 0


def test_ovwt_batchwise_scores_every_cell_once_per_variant():
    df = _cells()
    _, cell_scores, _ = ovwt_batchwise(df.lazy(), _cfg())
    for variant in ("M1K", "A1A"):
        subset = cell_scores.filter(pl.col("meta_variant_scored_against") == variant)
        expected = len(df.filter(pl.col(LABEL).is_in([variant, WT])))
        assert len(subset) == expected


def test_ovwt_batchwise_one_model_per_fold():
    cfg = _cfg(n_folds=3)
    _, _, models = ovwt_batchwise(_cells().lazy(), cfg)
    for fold_models in models.values():
        assert len(fold_models) == cfg.n_folds


def test_ovwt_batchwise_calibrators_present_when_enabled():
    _, _, models = ovwt_batchwise(_cells().lazy(), _cfg(calibrate=True))
    for fold_models in models.values():
        assert all(calibrator is not None for _, calibrator in fold_models)


def test_ovwt_batchwise_no_calibrators_when_disabled():
    _, _, models = ovwt_batchwise(_cells().lazy(), _cfg(calibrate=False))
    for fold_models in models.values():
        assert all(calibrator is None for _, calibrator in fold_models)


def test_ovwt_batchwise_recovers_separable_signal():
    """A well-separated variant should score near 1."""
    results, _, _ = ovwt_batchwise(_cells().lazy(), _cfg())
    assert results.get_column("auroc_pooled").min() > 0.9


def test_ovwt_batchwise_barcode_counts_are_per_variant():
    results, _, _ = ovwt_batchwise(_cells(barcodes_per_variant=2).lazy(), _cfg())
    # meta_n_barcodes counts the variant's OWN barcodes, not the WT ones
    assert set(results.get_column("meta_n_barcodes").to_list()) == {2}


def test_ovwt_batchwise_median_barcode_auroc_is_not_null():
    results, _, _ = ovwt_batchwise(_cells().lazy(), _cfg())
    assert results.get_column("auroc_median_barcode").null_count() == 0


def test_ovwt_batchwise_is_reproducible():
    a, _, _ = ovwt_batchwise(_cells().lazy(), _cfg(random_seed=5))
    b, _, _ = ovwt_batchwise(_cells().lazy(), _cfg(random_seed=5))
    assert a.sort(LABEL).equals(b.sort(LABEL))


def test_rare_barcode_variant_is_scored_not_skipped():
    """
    A variant whose barcodes contribute a lone cell used to die in the inner
    split (``ValueError: The least populated class in y has only 1 member``)
    and get dropped. It now trains: singleton strata go to the train half.
    """
    df = _cells(variants={"GOOD": 15, "TINY": 1}, barcodes_per_variant=2)
    results, _, models = ovwt_batchwise(df.lazy(), _cfg())
    labels = results.get_column(LABEL).to_list()
    assert "GOOD" in labels
    assert "TINY" in labels
    assert "TINY" in models


def test_variant_failure_is_isolated_not_fatal(monkeypatch):
    """One doomed variant must not take the whole run down with it."""
    import fisseq_data_pipeline.ovwt as ovwt_mod

    real_train = ovwt_mod.train_binary_xgboost

    def _fail_on_small_variant(train, val, cfg):
        if (train.get_column(LABEL) == "TINY").any():
            raise ValueError("synthetic training failure")
        return real_train(train, val, cfg)

    monkeypatch.setattr(ovwt_mod, "train_binary_xgboost", _fail_on_small_variant)

    df = _cells(variants={"GOOD": 15, "TINY": 3}, barcodes_per_variant=2)
    results, _, models = ovwt_batchwise(df.lazy(), _cfg())
    labels = results.get_column(LABEL).to_list()
    assert "GOOD" in labels
    assert "TINY" not in labels
    assert "TINY" not in models


def test_all_variants_filtered_out_yields_empty_but_typed_frames():
    df = _cells(variants={"M1K": 15})
    results, cell_scores, models = ovwt_batchwise(df.lazy(), _cfg(min_cells=10_000))
    assert len(results) == 0
    assert len(cell_scores) == 0
    assert models == {}
    assert results.columns == [
        LABEL,
        "auroc_pooled",
        "auroc_median_barcode",
        "meta_n_barcodes",
        "meta_n_cells",
    ]
    assert results.schema["auroc_pooled"] == pl.Float64
    assert cell_scores.schema["score"] == pl.Float64
    assert cell_scores.schema["meta_variant_scored_against"] == pl.String


# ---------------------------------------------------------------------------
# predict_binary
# ---------------------------------------------------------------------------


def test_predict_binary_returns_one_score_per_row():
    df = _cells()
    _, _, models = ovwt_batchwise(df.lazy(), _cfg())
    model, _ = models["M1K"][0]
    subset = df.filter(pl.col(LABEL).is_in(["M1K", WT]))
    features = subset.select([LABEL, "Intensity_Mean", "Texture_Var"])
    scores = predict_binary(features, model, LABEL, WT)
    assert scores.shape == (len(subset),)


def test_predict_binary_scores_wildtype_higher():
    """Models predict P(wildtype), so WT rows must score above variant rows."""
    df = _cells()
    _, _, models = ovwt_batchwise(df.lazy(), _cfg())
    model, _ = models["M1K"][0]
    subset = df.filter(pl.col(LABEL).is_in(["M1K", WT]))
    features = subset.select([LABEL, "Intensity_Mean", "Texture_Var"])
    scores = predict_binary(features, model, LABEL, WT)
    is_wt = subset.get_column(LABEL).to_numpy() == WT
    assert scores[is_wt].mean() > scores[~is_wt].mean()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_end_to_end(tmp_path: pathlib.Path):
    cells_path = tmp_path / "normalized.parquet"
    _cells().write_parquet(cells_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_data_pipeline.ovwt",
            "output_dir=.",
            f"input_file={cells_path}",
            f"label_column={LABEL}",
            "n_folds=3",
            "min_cells=null",
            "downsample_wt=false",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    for name in ("results.parquet", "cell_scores.parquet", "models.pkl"):
        assert (tmp_path / name).exists(), f"{name} missing"

    results = pl.read_parquet(tmp_path / "results.parquet")
    assert set(results.get_column(LABEL).to_list()) == {"M1K", "A1A"}


@pytest.mark.parametrize("n_folds", [2, 3])
def test_cli_respects_n_folds(tmp_path: pathlib.Path, n_folds: int):
    import pickle

    cells_path = tmp_path / "normalized.parquet"
    _cells().write_parquet(cells_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_data_pipeline.ovwt",
            "output_dir=.",
            f"input_file={cells_path}",
            f"label_column={LABEL}",
            f"n_folds={n_folds}",
            "min_cells=null",
            "downsample_wt=false",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "models.pkl", "rb") as f:
        models = pickle.load(f)
    assert all(len(folds) == n_folds for folds in models.values())
