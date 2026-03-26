from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import pytest
import scipy.stats

import fisseq_data_pipeline.aggregate as m


@pytest.fixture
def toy_norm_df() -> pl.DataFrame:
    """
    Small dataset with 2 batches, 2 labels per batch, and controls present.
    Features: f1, f2.
    """
    return pl.DataFrame(
        {
            "_meta_batch": ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"],
            "_meta_label": ["ctrl", "ctrl", "var", "var", "ctrl", "ctrl", "var", "var"],
            "_meta_is_control": [True, True, False, False, True, True, False, False],
            "f1": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 13.0, 14.0],
            "f2": [5.0, 7.0, 6.0, 6.0, 1.0, 3.0, 0.0, 2.0],
        }
    )


# ---------------------------------------------------------------------------
# ReferenceBaseAggregator / EMDAggregator
# ---------------------------------------------------------------------------


def test_emd_aggregator_returns_expected_columns(
    monkeypatch, toy_norm_df: pl.DataFrame
) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    control_df = toy_norm_df.filter(pl.col("_meta_is_control"))
    agg = m.EMDAggregator(control_df)
    result = agg.aggregate(toy_norm_df)

    assert {"_meta_label", "_meta_batch", "f1_EMD", "f2_EMD"}.issubset(
        set(result.columns)
    )


def test_emd_aggregator_matches_scipy(monkeypatch, toy_norm_df: pl.DataFrame) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    control_df = toy_norm_df.filter(pl.col("_meta_is_control"))
    agg = m.EMDAggregator(control_df)
    result = agg.aggregate(toy_norm_df)

    row = (
        result.filter(
            (pl.col("_meta_batch") == "b1") & (pl.col("_meta_label") == "var")
        )
        .to_dicts()
        .pop()
    )

    expected_f1 = scipy.stats.wasserstein_distance([2.0, 3.0], [0.0, 1.0])
    expected_f2 = scipy.stats.wasserstein_distance([6.0, 6.0], [5.0, 7.0])
    assert row["f1_EMD"] == pytest.approx(expected_f1, rel=1e-12, abs=1e-12)
    assert row["f2_EMD"] == pytest.approx(expected_f2, rel=1e-12, abs=1e-12)


def test_emd_aggregator_excludes_wt(monkeypatch, toy_norm_df: pl.DataFrame) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    df_with_wt = toy_norm_df.with_columns(
        pl.when(pl.col("_meta_label") == "ctrl")
        .then(pl.lit("WT"))
        .otherwise(pl.col("_meta_label"))
        .alias("_meta_label")
    )
    control_df = toy_norm_df.filter(pl.col("_meta_is_control"))
    agg = m.EMDAggregator(control_df)
    result = agg.aggregate(df_with_wt)

    assert "WT" not in result["_meta_label"].to_list()


# ---------------------------------------------------------------------------
# NativeAggregator subclasses
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df() -> pl.DataFrame:
    """Two groups, two features, no WT rows."""
    return pl.DataFrame(
        {
            "_meta_batch": ["b1", "b1", "b1", "b1", "b1", "b1"],
            "_meta_label": ["A", "A", "A", "B", "B", "B"],
            "f1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "f2": [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
        }
    )


def _get_row(df: pl.DataFrame, label: str) -> dict:
    return df.filter(pl.col("_meta_label") == label).to_dicts().pop()


def test_mean_aggregator(monkeypatch, simple_df: pl.DataFrame) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])
    result = m.MeanAggregator().aggregate(simple_df)

    assert {"_meta_label", "_meta_batch", "f1_mean", "f2_mean"}.issubset(
        set(result.columns)
    )
    row_a = _get_row(result, "A")
    assert row_a["f1_mean"] == pytest.approx(np.mean([1.0, 2.0, 3.0]))
    assert row_a["f2_mean"] == pytest.approx(np.mean([4.0, 5.0, 6.0]))

    row_b = _get_row(result, "B")
    assert row_b["f1_mean"] == pytest.approx(np.mean([10.0, 20.0, 30.0]))
    assert row_b["f2_mean"] == pytest.approx(np.mean([40.0, 50.0, 60.0]))


def test_median_aggregator(monkeypatch, simple_df: pl.DataFrame) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])
    result = m.MedianAggregator().aggregate(simple_df)

    assert {"_meta_label", "_meta_batch", "f1_median", "f2_median"}.issubset(
        set(result.columns)
    )
    row_a = _get_row(result, "A")
    assert row_a["f1_median"] == pytest.approx(np.median([1.0, 2.0, 3.0]))
    assert row_a["f2_median"] == pytest.approx(np.median([4.0, 5.0, 6.0]))


def test_mad_aggregator(monkeypatch, simple_df: pl.DataFrame) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])
    result = m.MADAggregator().aggregate(simple_df)

    assert {"_meta_label", "_meta_batch", "f1_MAD", "f2_MAD"}.issubset(
        set(result.columns)
    )
    row_a = _get_row(result, "A")
    vals = np.array([1.0, 2.0, 3.0])
    expected_mad = np.median(np.abs(vals - np.median(vals)))
    assert row_a["f1_MAD"] == pytest.approx(expected_mad)


def test_std_aggregator(monkeypatch, simple_df: pl.DataFrame) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])
    result = m.StdAggregator().aggregate(simple_df)

    assert {"_meta_label", "_meta_batch", "f1_std", "f2_std"}.issubset(
        set(result.columns)
    )
    row_a = _get_row(result, "A")
    # polars std uses ddof=1 by default
    assert row_a["f1_std"] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))
    assert row_a["f2_std"] == pytest.approx(np.std([4.0, 5.0, 6.0], ddof=1))


def test_native_aggregators_exclude_wt(monkeypatch) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1"])
    df = pl.DataFrame(
        {
            "_meta_batch": ["b1", "b1", "b1"],
            "_meta_label": ["WT", "A", "A"],
            "f1": [0.0, 1.0, 2.0],
        }
    )
    for agg_cls in (
        m.MeanAggregator,
        m.MedianAggregator,
        m.MADAggregator,
        m.StdAggregator,
    ):
        result = agg_cls().aggregate(df)
        assert "WT" not in result["_meta_label"].to_list()


# ---------------------------------------------------------------------------
# module-level aggregate()
# ---------------------------------------------------------------------------


def test_aggregate_no_normalize_returns_medians_and_emds(
    monkeypatch, toy_norm_df: pl.DataFrame
) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    agg_df, normalizer = m.aggregate(toy_norm_df, normalize_emds=False)
    assert normalizer is None

    assert set(["_meta_batch", "_meta_label", "f1", "f2", "f1_EMD", "f2_EMD"]).issubset(
        set(agg_df.columns)
    )

    row = (
        agg_df.filter(
            (pl.col("_meta_batch") == "b1") & (pl.col("_meta_label") == "var")
        )
        .to_dicts()
        .pop()
    )
    assert row["f1"] == pytest.approx(np.median([2.0, 3.0]))
    assert row["f2"] == pytest.approx(np.median([6.0, 6.0]))

    exp_f1 = scipy.stats.wasserstein_distance([2.0, 3.0], [0.0, 1.0])
    exp_f2 = scipy.stats.wasserstein_distance([6.0, 6.0], [5.0, 7.0])
    assert row["f1_EMD"] == pytest.approx(exp_f1, rel=1e-12, abs=1e-12)
    assert row["f2_EMD"] == pytest.approx(exp_f2, rel=1e-12, abs=1e-12)


def test_aggregate_with_normalize_uses_fit_and_normalize(
    monkeypatch, toy_norm_df: pl.DataFrame
) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    @dataclass
    class DummyNormalizer:
        tag: str = "dummy"

        def save(self, path: Path) -> None:
            path.write_bytes(b"dummy")

    def fake_fit_normalizer(
        lf: pl.LazyFrame, fit_batch_wise: bool, fit_only_on_control: bool
    ):
        assert fit_batch_wise is True
        assert fit_only_on_control is False
        cols = set(lf.collect_schema().names())
        assert {"_meta_batch", "_meta_label", "f1_EMD", "f2_EMD"}.issubset(cols)
        return DummyNormalizer()

    def fake_normalize(lf: pl.LazyFrame, normalizer: DummyNormalizer) -> pl.LazyFrame:
        return lf.with_columns(
            (pl.col("f1_EMD") * 10).alias("f1_EMD"),
            (pl.col("f2_EMD") * 10).alias("f2_EMD"),
        )

    monkeypatch.setattr(m, "fit_normalizer", fake_fit_normalizer)
    monkeypatch.setattr(m, "normalize", fake_normalize)
    monkeypatch.setattr(m, "Normalizer", DummyNormalizer, raising=False)

    norm_agg_df, normalizer = m.aggregate(toy_norm_df, normalize_emds=True)
    assert isinstance(normalizer, DummyNormalizer)

    raw_df, _ = m.aggregate(toy_norm_df, normalize_emds=False)

    raw_row = (
        raw_df.filter(
            (pl.col("_meta_batch") == "b2") & (pl.col("_meta_label") == "var")
        )
        .to_dicts()
        .pop()
    )
    norm_row = (
        norm_agg_df.filter(
            (pl.col("_meta_batch") == "b2") & (pl.col("_meta_label") == "var")
        )
        .to_dicts()
        .pop()
    )

    assert norm_row["f1_EMD"] == pytest.approx(raw_row["f1_EMD"] * 10)
    assert norm_row["f2_EMD"] == pytest.approx(raw_row["f2_EMD"] * 10)
    assert norm_row["f1"] == pytest.approx(raw_row["f1"])
    assert norm_row["f2"] == pytest.approx(raw_row["f2"])


def test_aggregate_norm_only_to_synonymous_marks_controls_for_fit(
    monkeypatch, toy_norm_df: pl.DataFrame
) -> None:
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    def fake_variant_classification(label: str) -> str:
        return "Synonymous" if label == "ctrl" else "Nonsynonymous"

    monkeypatch.setattr(m, "variant_classification", fake_variant_classification)

    @dataclass
    class DummyNormalizer:
        tag: str = "dummy-syn"

        def save(self, path: Path) -> None:
            path.write_bytes(b"dummy")

    def fake_fit_normalizer(
        lf: pl.LazyFrame, fit_batch_wise: bool, fit_only_on_control: bool
    ):
        assert fit_batch_wise is True
        assert fit_only_on_control is True

        cols = set(lf.collect_schema().names())
        assert {
            "_meta_batch",
            "_meta_label",
            "f1_EMD",
            "f2_EMD",
            "_meta_is_control",
        }.issubset(cols)

        df = lf.collect()
        flags = (
            df.select(["_meta_label", "_meta_is_control"])
            .unique()
            .sort("_meta_label")
            .to_dicts()
        )
        by_label = {d["_meta_label"]: d["_meta_is_control"] for d in flags}
        assert by_label["ctrl"] is True
        assert by_label["var"] is False

        return DummyNormalizer()

    def fake_normalize(lf: pl.LazyFrame, normalizer: DummyNormalizer) -> pl.LazyFrame:
        return lf.with_columns(
            (pl.col("f1_EMD") + 1).alias("f1_EMD"),
            (pl.col("f2_EMD") + 1).alias("f2_EMD"),
        )

    monkeypatch.setattr(m, "fit_normalizer", fake_fit_normalizer)
    monkeypatch.setattr(m, "normalize", fake_normalize)
    monkeypatch.setattr(m, "Normalizer", DummyNormalizer, raising=False)

    norm_agg_df, normalizer = m.aggregate(
        toy_norm_df, normalize_emds=True, norm_only_to_synonymous=True
    )
    assert isinstance(normalizer, DummyNormalizer)

    assert set(["_meta_batch", "_meta_label", "f1", "f2", "f1_EMD", "f2_EMD"]).issubset(
        set(norm_agg_df.columns)
    )


def test_compute_cli_native_aggregator(
    monkeypatch, tmp_path: Path, toy_norm_df: pl.DataFrame
) -> None:
    """compute_cli with a native aggregator writes aggregated.parquet."""
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    input_path = tmp_path / "input.parquet"
    toy_norm_df.write_parquet(input_path)

    m.compute_cli(input_path, tmp_path / "out", aggregator="mean")

    out_path = tmp_path / "out" / "aggregated.parquet"
    assert out_path.exists()
    result = pl.read_parquet(out_path)
    assert {"_meta_label", "_meta_batch", "f1_mean", "f2_mean"}.issubset(
        set(result.columns)
    )
    # raw feature columns should not be present
    assert "f1" not in result.columns
    assert "f2" not in result.columns


def test_compute_cli_reference_aggregator(
    monkeypatch, tmp_path: Path, toy_norm_df: pl.DataFrame
) -> None:
    """compute_cli with a reference aggregator writes aggregated.parquet."""
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    input_path = tmp_path / "input.parquet"
    toy_norm_df.write_parquet(input_path)

    m.compute_cli(input_path, tmp_path / "out", aggregator="EMD")

    out_path = tmp_path / "out" / "aggregated.parquet"
    assert out_path.exists()
    result = pl.read_parquet(out_path)
    assert {"_meta_label", "_meta_batch", "f1_EMD", "f2_EMD"}.issubset(
        set(result.columns)
    )


def test_compute_cli_unknown_aggregator(
    tmp_path: Path, toy_norm_df: pl.DataFrame
) -> None:
    input_path = tmp_path / "input.parquet"
    toy_norm_df.write_parquet(input_path)

    with pytest.raises(ValueError, match="Unknown aggregator"):
        m.compute_cli(input_path, tmp_path / "out", aggregator="bogus")


def test_normalize_cli_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    """normalize_cli writes normalized.parquet and normalizer.pkl."""
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1_EMD"])

    # Build a minimal aggregate df with synonymous and missense labels
    agg_df = pl.DataFrame(
        {
            "_meta_batch": ["b1", "b1", "b1", "b1"],
            "_meta_label": ["A1A", "A1A", "A1B", "A1B"],  # A1A → Synonymous
            "f1_EMD": [0.1, 0.2, 1.0, 1.5],
        }
    )

    class DummyNormalizer:
        def save(self, path: Path) -> None:
            path.write_bytes(b"ok")

    def fake_fit_normalizer(lf, fit_batch_wise, fit_only_on_control):
        assert fit_batch_wise is True
        assert fit_only_on_control is True
        return DummyNormalizer()

    def fake_normalize(lf, normalizer):
        return lf.with_columns((pl.col("f1_EMD") * 2).alias("f1_EMD"))

    monkeypatch.setattr(m, "fit_normalizer", fake_fit_normalizer)
    monkeypatch.setattr(m, "normalize", fake_normalize)

    input_path = tmp_path / "agg.parquet"
    agg_df.write_parquet(input_path)

    m.normalize_cli(input_path, tmp_path / "out")

    assert (tmp_path / "out" / "normalized.parquet").exists()
    assert (tmp_path / "out" / "normalizer.pkl").exists()

    result = pl.read_parquet(tmp_path / "out" / "normalized.parquet")
    # synthetic _meta_is_control should be dropped from output
    assert "_meta_is_control" not in result.columns
    assert "f1_EMD" in result.columns
