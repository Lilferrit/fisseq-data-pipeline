# test_emd_aggregate.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import pytest
import scipy.stats

# IMPORTANT:
# Replace `your_module` with the actual module path where these live, e.g.
# from casanovoutils.foo import EMDAggregator, aggregate, cli_wrapper
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


def test_emdaggregator_builds_reference_cache(
    monkeypatch, toy_norm_df: pl.DataFrame
) -> None:
    """
    EMDAggregator should cache per-(batch, feature) reference arrays from the
    provided reference dataframe (controls-only in this pipeline).
    """
    # Make get_feature_cols deterministic for the test
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    control_df = toy_norm_df.filter(pl.col("_meta_is_control"))
    agg = m.EMDAggregator(control_df)

    # Ensure batches exist in cache
    assert "b1" in agg.col_dict
    assert "b2" in agg.col_dict

    # Ensure features exist per batch
    assert set(agg.col_dict["b1"].keys()) == {"f1", "f2"}
    assert set(agg.col_dict["b2"].keys()) == {"f1", "f2"}

    # Ensure cached arrays match control values (order preserved by filter)
    np.testing.assert_allclose(agg.col_dict["b1"]["f1"], np.array([0.0, 1.0]))
    np.testing.assert_allclose(agg.col_dict["b1"]["f2"], np.array([5.0, 7.0]))
    np.testing.assert_allclose(agg.col_dict["b2"]["f1"], np.array([10.0, 11.0]))
    np.testing.assert_allclose(agg.col_dict["b2"]["f2"], np.array([1.0, 3.0]))


def test_agg_emd_matches_scipy(monkeypatch, toy_norm_df: pl.DataFrame) -> None:
    """
    agg_emd should compute scipy.stats.wasserstein_distance between the group's
    values and the cached reference distribution for (batch, feature).
    """
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    control_df = toy_norm_df.filter(pl.col("_meta_is_control"))
    agg = m.EMDAggregator(control_df)

    # Group is the "var" rows in batch b1 for feature f1
    feature_series = (
        toy_norm_df.filter(
            (pl.col("_meta_batch") == "b1") & (pl.col("_meta_label") == "var")
        )
        .select(pl.col("f1"))
        .to_series()
    )
    batch_series = pl.Series("_meta_batch_dummy", ["b1", "b1"])

    out = agg.agg_emd([feature_series, batch_series])

    expected = scipy.stats.wasserstein_distance(
        np.array([2.0, 3.0]),  # var values
        np.array([0.0, 1.0]),  # control values for b1
    )

    assert out.dtype == pl.Float64
    assert out.len() == 1
    assert out.item() == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_aggregate_no_normalize_returns_medians_and_emds(
    monkeypatch, toy_norm_df: pl.DataFrame
) -> None:
    """
    aggregate(..., normalize_emds=False) should return (agg_df, None) and
    include medians for feature cols + EMD columns.
    """
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    agg_df, normalizer = m.aggregate(toy_norm_df, normalize_emds=False)
    assert normalizer is None

    # Expected columns: keys + features + feature_EMD columns
    assert set(["_meta_batch", "_meta_label", "f1", "f2", "f1_EMD", "f2_EMD"]).issubset(
        set(agg_df.columns)
    )

    # Check one group's medians: (b1, var)
    row = (
        agg_df.filter(
            (pl.col("_meta_batch") == "b1") & (pl.col("_meta_label") == "var")
        )
        .to_dicts()
        .pop()
    )
    assert row["f1"] == pytest.approx(np.median([2.0, 3.0]))
    assert row["f2"] == pytest.approx(np.median([6.0, 6.0]))

    # Check the same group's EMDs against b1 controls
    exp_f1 = scipy.stats.wasserstein_distance([2.0, 3.0], [0.0, 1.0])
    exp_f2 = scipy.stats.wasserstein_distance([6.0, 6.0], [5.0, 7.0])
    assert row["f1_EMD"] == pytest.approx(exp_f1, rel=1e-12, abs=1e-12)
    assert row["f2_EMD"] == pytest.approx(exp_f2, rel=1e-12, abs=1e-12)


def test_aggregate_with_normalize_uses_fit_and_normalize(
    monkeypatch, toy_norm_df: pl.DataFrame
) -> None:
    """
    aggregate(..., normalize_emds=True) should call fit_normalizer and normalize,
    and return a dataframe where the EMD columns are replaced by normalized values.
    We stub normalization to make behavior deterministic.
    """
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    # Stub normalizer + functions
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
        # Ensure expected columns exist
        cols = set(lf.collect_schema().names())
        assert {"_meta_batch", "_meta_label", "f1_EMD", "f2_EMD"}.issubset(cols)
        return DummyNormalizer()

    def fake_normalize(lf: pl.LazyFrame, normalizer: DummyNormalizer) -> pl.LazyFrame:
        # For testing, "normalize" by multiplying EMD columns by 10
        return lf.with_columns(
            (pl.col("f1_EMD") * 10).alias("f1_EMD"),
            (pl.col("f2_EMD") * 10).alias("f2_EMD"),
        )

    monkeypatch.setattr(m, "fit_normalizer", fake_fit_normalizer)
    monkeypatch.setattr(m, "normalize", fake_normalize)
    monkeypatch.setattr(m, "Normalizer", DummyNormalizer, raising=False)

    norm_agg_df, normalizer = m.aggregate(toy_norm_df, normalize_emds=True)
    assert isinstance(normalizer, DummyNormalizer)

    # Compare normalized EMDs to raw EMDs * 10 for one group
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

    # Medians should be unchanged
    assert norm_row["f1"] == pytest.approx(raw_row["f1"])
    assert norm_row["f2"] == pytest.approx(raw_row["f2"])


def test_aggregate_norm_only_to_synonymous_marks_controls_for_fit(
    monkeypatch, toy_norm_df: pl.DataFrame
) -> None:
    """
    When norm_only_to_synonymous=True, aggregate() should:
      - add a synthetic `_meta_is_control` column to the aggregated EMD dataframe
        where groups classified as "Synonymous" are True
      - call fit_normalizer(..., fit_only_on_control=True)
    """
    monkeypatch.setattr(m, "get_feature_cols", lambda df, as_string=True: ["f1", "f2"])

    # Classify only the "ctrl" label as Synonymous (others not)
    def fake_variant_classification(label: str) -> str:
        return "Synonymous" if label == "ctrl" else "Nonsynonymous"

    monkeypatch.setattr(m, "variant_classification", fake_variant_classification)

    # Stub normalizer + functions
    @dataclass
    class DummyNormalizer:
        tag: str = "dummy-syn"

        def save(self, path: Path) -> None:
            path.write_bytes(b"dummy")

    def fake_fit_normalizer(
        lf: pl.LazyFrame, fit_batch_wise: bool, fit_only_on_control: bool
    ):
        assert fit_batch_wise is True
        assert fit_only_on_control is True  # <-- key behavior for this test

        cols = set(lf.collect_schema().names())
        assert {
            "_meta_batch",
            "_meta_label",
            "f1_EMD",
            "f2_EMD",
            "_meta_is_control",
        }.issubset(cols)

        # Validate control marking semantics: "ctrl" groups are True, "var" groups False
        df = lf.collect()
        flags = (
            df.select(["_meta_label", "_meta_is_control"])
            .unique()
            .sort("_meta_label")
            .to_dicts()
        )
        # Expect both labels present
        by_label = {d["_meta_label"]: d["_meta_is_control"] for d in flags}
        assert by_label["ctrl"] is True
        assert by_label["var"] is False

        return DummyNormalizer()

    def fake_normalize(lf: pl.LazyFrame, normalizer: DummyNormalizer) -> pl.LazyFrame:
        # Deterministic transform to show we ran through normalize()
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

    # Sanity: output should still have medians + EMD columns
    assert set(["_meta_batch", "_meta_label", "f1", "f2", "f1_EMD", "f2_EMD"]).issubset(
        set(norm_agg_df.columns)
    )


def test_cli_wrapper_writes_outputs(
    monkeypatch, tmp_path: Path, toy_norm_df: pl.DataFrame
) -> None:
    """
    cli_wrapper should always write aggregated.parquet, and if aggregate returns
    a non-None normalizer it should save emd_normalizer.pkl too.
    """

    class DummyNormalizer2:
        def save(self, path: Path) -> None:
            path.write_bytes(b"ok")

    # Stub aggregate to avoid relying on the full pipeline
    out_df = pl.DataFrame(
        {
            "_meta_batch": ["b1"],
            "_meta_label": ["x"],
            "f1": [1.0],
            "f1_EMD": [0.1],
        }
    )

    def fake_aggregate(norm_df, normalize_emds: bool = True):
        return out_df, (DummyNormalizer2() if normalize_emds else None)

    monkeypatch.setattr(m, "aggregate", fake_aggregate)

    m.cli_wrapper(toy_norm_df, tmp_path, normalize_emds=True)

    assert (tmp_path / "aggregated.parquet").exists()
    assert (tmp_path / "emd_normalizer.pkl").exists()

    # Sanity check parquet readable
    reread = pl.read_parquet(tmp_path / "aggregated.parquet")
    assert reread.shape == out_df.shape
    assert reread.columns == out_df.columns
