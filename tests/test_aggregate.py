from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import scipy.stats
from omegaconf import OmegaConf

import fisseq_data_pipeline.aggregate as m
from fisseq_data_pipeline.aggregate import AggregateConfig
from fisseq_data_pipeline.constants import CONTROL_COLUMN_NAME

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_norm_df() -> pl.DataFrame:
    """Cell-level dataset: WT cells are controls, A1B cells are variants."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B", "WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False, True, True, False, False],
            "f1": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 13.0, 14.0],
            "f2": [5.0, 7.0, 6.0, 6.0, 1.0, 3.0, 0.0, 2.0],
        }
    )


@pytest.fixture
def simple_df() -> pl.DataFrame:
    """Two variant groups with no control rows."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B", "B"],
            "meta_is_control": [False, False, False, False, False, False],
            "f1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "f2": [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
        }
    )


def _get_row(df: pl.DataFrame, label: str) -> dict:
    return df.filter(pl.col("meta_aa_changes") == label).to_dicts().pop()


# ---------------------------------------------------------------------------
# variant_classification
# ---------------------------------------------------------------------------


def test_variant_classification_adds_column():
    lf = pl.DataFrame({"meta_aa_changes": ["A1A", "A1B"]}).lazy()
    result = m.variant_classification(lf, "meta_aa_changes").collect()
    assert CONTROL_COLUMN_NAME in result.columns


def test_variant_classification_synonymous_is_true():
    lf = pl.DataFrame({"meta_aa_changes": ["A1A"]}).lazy()
    result = m.variant_classification(lf, "meta_aa_changes").collect()
    assert result[CONTROL_COLUMN_NAME][0] is True


def test_variant_classification_missense_is_false():
    lf = pl.DataFrame({"meta_aa_changes": ["A1B"]}).lazy()
    result = m.variant_classification(lf, "meta_aa_changes").collect()
    assert result[CONTROL_COLUMN_NAME][0] is False


def test_variant_classification_wt_is_false():
    lf = pl.DataFrame({"meta_aa_changes": ["WT"]}).lazy()
    result = m.variant_classification(lf, "meta_aa_changes").collect()
    assert result[CONTROL_COLUMN_NAME][0] is False


def test_variant_classification_frameshift_is_false():
    lf = pl.DataFrame({"meta_aa_changes": ["A1fs"]}).lazy()
    result = m.variant_classification(lf, "meta_aa_changes").collect()
    assert result[CONTROL_COLUMN_NAME][0] is False


def test_variant_classification_nonsense_is_false():
    lf = pl.DataFrame({"meta_aa_changes": ["A1X"]}).lazy()
    result = m.variant_classification(lf, "meta_aa_changes").collect()
    assert result[CONTROL_COLUMN_NAME][0] is False


def test_variant_classification_column_is_boolean():
    lf = pl.DataFrame({"meta_aa_changes": ["A1A", "A1B"]}).lazy()
    result = m.variant_classification(lf, "meta_aa_changes").collect()
    assert result[CONTROL_COLUMN_NAME].dtype == pl.Boolean


def test_variant_classification_custom_label_col():
    lf = pl.DataFrame({"variant": ["A1A", "A1B"]}).lazy()
    result = m.variant_classification(lf, "variant").collect()
    assert result[CONTROL_COLUMN_NAME].to_list() == [True, False]


# ---------------------------------------------------------------------------
# ReferenceBaseAggregator / EMDAggregator
# ---------------------------------------------------------------------------


def test_emd_aggregator_returns_expected_columns(toy_norm_df: pl.DataFrame) -> None:
    control_df = toy_norm_df.filter(pl.col("meta_is_control"))
    agg = m.EMDAggregator(control_df)
    result = agg.aggregate(toy_norm_df)
    assert {"meta_aa_changes", "f1_EMD", "f2_EMD"}.issubset(set(result.columns))


def test_emd_aggregator_matches_scipy(toy_norm_df: pl.DataFrame) -> None:
    control_df = toy_norm_df.filter(pl.col("meta_is_control"))
    agg = m.EMDAggregator(control_df)
    result = agg.aggregate(toy_norm_df)

    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()

    ref_f1 = [0.0, 1.0, 10.0, 11.0]
    ref_f2 = [5.0, 7.0, 1.0, 3.0]
    var_f1 = [2.0, 3.0, 13.0, 14.0]
    var_f2 = [6.0, 6.0, 0.0, 2.0]

    assert row["f1_EMD"] == pytest.approx(
        scipy.stats.wasserstein_distance(var_f1, ref_f1), rel=1e-12, abs=1e-12
    )
    assert row["f2_EMD"] == pytest.approx(
        scipy.stats.wasserstein_distance(var_f2, ref_f2), rel=1e-12, abs=1e-12
    )


def test_emd_aggregator_excludes_control_rows(toy_norm_df: pl.DataFrame) -> None:
    control_df = toy_norm_df.filter(pl.col("meta_is_control"))
    agg = m.EMDAggregator(control_df)
    result = agg.aggregate(toy_norm_df)
    assert "WT" not in result["meta_aa_changes"].to_list()


# ---------------------------------------------------------------------------
# NativeAggregator subclasses
# ---------------------------------------------------------------------------


def test_mean_aggregator(simple_df: pl.DataFrame) -> None:
    result = m.MeanAggregator().aggregate(simple_df)
    assert {"meta_aa_changes", "f1_mean", "f2_mean"}.issubset(set(result.columns))
    row_a = _get_row(result, "A")
    assert row_a["f1_mean"] == pytest.approx(np.mean([1.0, 2.0, 3.0]))
    assert row_a["f2_mean"] == pytest.approx(np.mean([4.0, 5.0, 6.0]))
    row_b = _get_row(result, "B")
    assert row_b["f1_mean"] == pytest.approx(np.mean([10.0, 20.0, 30.0]))
    assert row_b["f2_mean"] == pytest.approx(np.mean([40.0, 50.0, 60.0]))


def test_median_aggregator(simple_df: pl.DataFrame) -> None:
    result = m.MedianAggregator().aggregate(simple_df)
    assert {"meta_aa_changes", "f1_median", "f2_median"}.issubset(set(result.columns))
    row_a = _get_row(result, "A")
    assert row_a["f1_median"] == pytest.approx(np.median([1.0, 2.0, 3.0]))
    assert row_a["f2_median"] == pytest.approx(np.median([4.0, 5.0, 6.0]))


def test_mad_aggregator(simple_df: pl.DataFrame) -> None:
    result = m.MADAggregator().aggregate(simple_df)
    assert {"meta_aa_changes", "f1_MAD", "f2_MAD"}.issubset(set(result.columns))
    row_a = _get_row(result, "A")
    vals = np.array([1.0, 2.0, 3.0])
    expected_mad = np.median(np.abs(vals - np.median(vals)))
    assert row_a["f1_MAD"] == pytest.approx(expected_mad)


def test_std_aggregator(simple_df: pl.DataFrame) -> None:
    result = m.StdAggregator().aggregate(simple_df)
    assert {"meta_aa_changes", "f1_std", "f2_std"}.issubset(set(result.columns))
    row_a = _get_row(result, "A")
    assert row_a["f1_std"] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))
    assert row_a["f2_std"] == pytest.approx(np.std([4.0, 5.0, 6.0], ddof=1))


def test_native_aggregators_exclude_control_rows() -> None:
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "A", "A"],
            "meta_is_control": [True, False, False],
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
        assert "WT" not in result["meta_aa_changes"].to_list()


# ---------------------------------------------------------------------------
# MultiAggregator
# ---------------------------------------------------------------------------


def test_multi_aggregator_columns_include_all_sub_aggregator_outputs(
    simple_df: pl.DataFrame,
) -> None:
    agg = m.MultiAggregator(
        [m.MeanAggregator(), m.MedianAggregator(), m.StdAggregator()]
    )
    result = agg.aggregate(simple_df)
    assert {"meta_aa_changes", "f1_mean", "f2_mean"}.issubset(set(result.columns))
    assert {"f1_median", "f2_median"}.issubset(set(result.columns))
    assert {"f1_std", "f2_std"}.issubset(set(result.columns))


def test_multi_aggregator_label_column_appears_once(simple_df: pl.DataFrame) -> None:
    agg = m.MultiAggregator([m.MeanAggregator(), m.MedianAggregator()])
    result = agg.aggregate(simple_df)
    assert result.columns.count("meta_aa_changes") == 1


def test_multi_aggregator_values_match_individual_aggregators(
    simple_df: pl.DataFrame,
) -> None:
    mean_result = m.MeanAggregator().aggregate(simple_df)
    std_result = m.StdAggregator().aggregate(simple_df)
    multi_result = m.MultiAggregator([m.MeanAggregator(), m.StdAggregator()]).aggregate(
        simple_df
    )

    row_a_mean = mean_result.filter(pl.col("meta_aa_changes") == "A").to_dicts().pop()
    row_a_multi = multi_result.filter(pl.col("meta_aa_changes") == "A").to_dicts().pop()
    assert row_a_multi["f1_mean"] == pytest.approx(row_a_mean["f1_mean"])

    row_a_std = std_result.filter(pl.col("meta_aa_changes") == "A").to_dicts().pop()
    assert row_a_multi["f1_std"] == pytest.approx(row_a_std["f1_std"])


def test_multi_aggregator_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        m.MultiAggregator([]).aggregate(
            pl.DataFrame(
                {"meta_aa_changes": ["A"], "meta_is_control": [False], "f1": [1.0]}
            )
        )


# ---------------------------------------------------------------------------
# aggregate() function
# ---------------------------------------------------------------------------


def test_aggregate_emd_returns_expected_columns(toy_norm_df: pl.DataFrame) -> None:
    result = m.aggregate(
        toy_norm_df.lazy(), label_col="meta_aa_changes", aggregator_name="EMD"
    )
    assert {"meta_aa_changes", "f1_EMD", "f2_EMD"}.issubset(set(result.columns))


def test_aggregate_mean_returns_expected_columns(simple_df: pl.DataFrame) -> None:
    result = m.aggregate(
        simple_df.lazy(), label_col="meta_aa_changes", aggregator_name="mean"
    )
    assert {"meta_aa_changes", "f1_mean", "f2_mean"}.issubset(set(result.columns))


def test_aggregate_emd_excludes_control_rows(toy_norm_df: pl.DataFrame) -> None:
    result = m.aggregate(
        toy_norm_df.lazy(), label_col="meta_aa_changes", aggregator_name="EMD"
    )
    assert "WT" not in result["meta_aa_changes"].to_list()


def test_aggregate_multi_returns_columns_from_all_non_emd_aggregators(
    toy_norm_df: pl.DataFrame,
) -> None:
    result = m.aggregate(
        toy_norm_df.lazy(), label_col="meta_aa_changes", aggregator_name="multi"
    )
    assert "meta_aa_changes" in result.columns
    assert result.columns.count("meta_aa_changes") == 1
    for suffix in ("_mean", "_median", "_MAD", "_std", "_KS", "_QQ", "_AUROC"):
        assert any(c.endswith(suffix) for c in result.columns), (
            f"missing {suffix} columns"
        )
    assert not any(c.endswith("_EMD") for c in result.columns)


def test_aggregate_unknown_raises() -> None:
    lf = pl.DataFrame(
        {"meta_aa_changes": ["A"], "meta_is_control": [False], "f1": [1.0]}
    ).lazy()
    with pytest.raises(ValueError, match="Unknown aggregator"):
        m.aggregate(lf, label_col="meta_aa_changes", aggregator_name="bogus")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def make_agg_cfg(
    tmp_path, *, output_root=None, save_normalizer=False, aggregator="mean"
) -> OmegaConf:
    """Return a DictConfig for AggregateConfig with sensible test defaults."""
    return OmegaConf.structured(
        AggregateConfig(
            output_dir=str(tmp_path),
            output_root=output_root,
            input_file=str(tmp_path / "input.parquet"),
            save_normalizer=save_normalizer,
            aggregator=aggregator,
        )
    )


def write_agg_input_parquet(tmp_path) -> None:
    """Write cell-level test parquet with WT controls, synonymous and missense variants."""
    pl.DataFrame(
        {
            "meta_aa_changes": [
                "WT",
                "WT",
                "WT",
                "A1A",
                "A1A",
                "A1A",
                "A2A",
                "A2A",
                "A2A",
                "A1B",
                "A1B",
                "A1B",
            ],
            "meta_is_control": [
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "f1": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 10.0, 10.0, 10.0],
            "f2": [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 30.0, 30.0, 30.0],
        }
    ).write_parquet(tmp_path / "input.parquet")


def test_main_creates_output_file(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    assert (tmp_path / "input.parquet").exists()


def test_main_output_contains_label_column(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert "meta_aa_changes" in result.columns


def test_main_synonymous_rows_normalized_to_zero_mean(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "input.parquet")
    # A1A and A2A are synonymous; after normalization their mean should be ~0
    syn_rows = result.filter(pl.col("meta_aa_changes").is_in(["A1A", "A2A"]))
    assert syn_rows["f1_mean"].mean() == pytest.approx(0.0, abs=1e-6)


def test_main_output_root_naming(tmp_path):
    write_agg_input_parquet(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path, output_root=root))
    assert (tmp_path / "run1.input.parquet").exists()


def test_main_saves_normalizer_when_configured(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path, save_normalizer=True))
    assert (tmp_path / "normalizer.parquet").exists()
