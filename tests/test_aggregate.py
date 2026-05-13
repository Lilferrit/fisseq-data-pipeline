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
# Null-value fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def null_df() -> pl.DataFrame:
    """Variant group A has one null per feature; group B is all-null for f1."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B"],
            "meta_is_control": [False, False, False, False, False],
            "f1": pl.Series([1.0, None, 3.0, None, None], dtype=pl.Float64),
            "f2": pl.Series([4.0, 5.0, None, 7.0, 8.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def null_ref_df() -> pl.DataFrame:
    """Control rows: f1 has one null; f2 is entirely null."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "WT"],
            "meta_is_control": [True, True, True],
            "f1": pl.Series([1.0, None, 3.0], dtype=pl.Float64),
            "f2": pl.Series([None, None, None], dtype=pl.Float64),
        }
    )


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
    tmp_path,
    *,
    output_root=None,
    save_normalizer=False,
    aggregator="mean",
    block_list_file=None,
) -> OmegaConf:
    """Return a DictConfig for AggregateConfig with sensible test defaults."""
    return OmegaConf.structured(
        AggregateConfig(
            output_dir=str(tmp_path),
            output_root=output_root,
            input_file=str(tmp_path / "input.parquet"),
            save_normalizer=save_normalizer,
            aggregator=aggregator,
            block_list_file=block_list_file,
        )
    )


def write_agg_input_parquet(tmp_path, *, with_barcode: bool = False) -> None:
    """Write cell-level test parquet with WT controls, synonymous and missense variants."""
    data = {
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
    if with_barcode:
        data["meta_barcode"] = [
            "bc1",
            "bc2",
            "bc3",
            "bc1",
            "bc2",
            "bc1",
            "bc1",
            "bc1",
            "bc2",
            "bc3",
            "bc3",
            "bc3",
        ]
    pl.DataFrame(data).write_parquet(tmp_path / "input.parquet")


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


# ---------------------------------------------------------------------------
# Null handling — native aggregators
# ---------------------------------------------------------------------------


def test_mean_aggregator_ignores_nulls(null_df: pl.DataFrame) -> None:
    result = m.MeanAggregator().aggregate(null_df)
    row_a = _get_row(result, "A")
    assert row_a["f1_mean"] == pytest.approx(2.0)  # mean of [1, 3]
    assert row_a["f2_mean"] == pytest.approx(4.5)  # mean of [4, 5]


def test_mean_aggregator_all_null_returns_null(null_df: pl.DataFrame) -> None:
    result = m.MeanAggregator().aggregate(null_df)
    row_b = _get_row(result, "B")
    assert row_b["f1_mean"] is None


def test_median_aggregator_ignores_nulls(null_df: pl.DataFrame) -> None:
    result = m.MedianAggregator().aggregate(null_df)
    row_a = _get_row(result, "A")
    assert row_a["f1_median"] == pytest.approx(2.0)  # median of [1, 3]


def test_median_aggregator_all_null_returns_null(null_df: pl.DataFrame) -> None:
    result = m.MedianAggregator().aggregate(null_df)
    row_b = _get_row(result, "B")
    assert row_b["f1_median"] is None


def test_mad_aggregator_ignores_nulls(null_df: pl.DataFrame) -> None:
    result = m.MADAggregator().aggregate(null_df)
    row_a = _get_row(result, "A")
    vals = np.array([1.0, 3.0])
    expected_mad = np.median(np.abs(vals - np.median(vals)))
    assert row_a["f1_MAD"] == pytest.approx(expected_mad)


def test_mad_aggregator_all_null_returns_null(null_df: pl.DataFrame) -> None:
    result = m.MADAggregator().aggregate(null_df)
    row_b = _get_row(result, "B")
    assert row_b["f1_MAD"] is None


def test_std_aggregator_ignores_nulls(null_df: pl.DataFrame) -> None:
    result = m.StdAggregator().aggregate(null_df)
    row_a = _get_row(result, "A")
    assert row_a["f1_std"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


def test_std_aggregator_all_null_returns_null(null_df: pl.DataFrame) -> None:
    result = m.StdAggregator().aggregate(null_df)
    row_b = _get_row(result, "B")
    assert row_b["f1_std"] is None


# ---------------------------------------------------------------------------
# Null handling — reference-based aggregators
# ---------------------------------------------------------------------------


def _ref_based_null_df() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Variant group A with one null; reference with one null. f2 all-null in variant."""
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "WT", "A1B", "A1B", "A1B"],
            "meta_is_control": [True, True, True, False, False, False],
            "f1": pl.Series([1.0, None, 3.0, 10.0, None, 30.0], dtype=pl.Float64),
            "f2": pl.Series([5.0, 6.0, 7.0, None, None, None], dtype=pl.Float64),
        }
    )
    ref = full.filter(pl.col("meta_is_control"))
    return full, ref


def test_emd_aggregator_ignores_nulls_in_variant() -> None:
    full, ref = _ref_based_null_df()
    agg = m.EMDAggregator(ref)
    result = agg.aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    expected = scipy.stats.wasserstein_distance([10.0, 30.0], [1.0, 3.0])
    assert row["f1_EMD"] == pytest.approx(expected)


def test_emd_aggregator_all_null_variant_returns_null() -> None:
    full, ref = _ref_based_null_df()
    agg = m.EMDAggregator(ref)
    result = agg.aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    assert row["f2_EMD"] is None


def test_emd_aggregator_all_null_reference_returns_null() -> None:
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False],
            "f1": pl.Series([None, None, 1.0, 2.0], dtype=pl.Float64),
        }
    )
    ref = full.filter(pl.col("meta_is_control"))
    result = m.EMDAggregator(ref).aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    assert row["f1_EMD"] is None


def test_ks_aggregator_ignores_nulls_in_variant() -> None:
    full, ref = _ref_based_null_df()
    agg = m.KSAggregator(ref)
    result = agg.aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    expected = scipy.stats.ks_2samp([10.0, 30.0], [1.0, 3.0]).statistic
    assert row["f1_KS"] == pytest.approx(expected)


def test_ks_aggregator_all_null_variant_returns_null() -> None:
    full, ref = _ref_based_null_df()
    result = m.KSAggregator(ref).aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    assert row["f2_KS"] is None


def test_ks_aggregator_all_null_reference_returns_null() -> None:
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False],
            "f1": pl.Series([None, None, 1.0, 2.0], dtype=pl.Float64),
        }
    )
    ref = full.filter(pl.col("meta_is_control"))
    result = m.KSAggregator(ref).aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    assert row["f1_KS"] is None


def test_qq_aggregator_ignores_nulls_in_variant() -> None:
    full, ref = _ref_based_null_df()
    agg = m.QQCorrelationAggregator(ref)
    result = agg.aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    assert row["f1_QQ"] is not None


def test_qq_aggregator_all_null_variant_returns_null() -> None:
    full, ref = _ref_based_null_df()
    result = m.QQCorrelationAggregator(ref).aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    assert row["f2_QQ"] is None


def test_qq_aggregator_all_null_reference_returns_null() -> None:
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False],
            "f1": pl.Series([None, None, 1.0, 2.0], dtype=pl.Float64),
        }
    )
    ref = full.filter(pl.col("meta_is_control"))
    result = m.QQCorrelationAggregator(ref).aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    assert row["f1_QQ"] is None


def test_auroc_aggregator_ignores_nulls_in_variant() -> None:
    full, ref = _ref_based_null_df()
    result = m.AUROCAggregator(ref).aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    assert row["f1_AUROC"] is not None


def test_auroc_aggregator_all_null_variant_returns_null() -> None:
    full, ref = _ref_based_null_df()
    result = m.AUROCAggregator(ref).aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    assert row["f2_AUROC"] is None


def test_auroc_aggregator_all_null_reference_returns_null() -> None:
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False],
            "f1": pl.Series([None, None, 1.0, 2.0], dtype=pl.Float64),
        }
    )
    ref = full.filter(pl.col("meta_is_control"))
    result = m.AUROCAggregator(ref).aggregate(full)
    row = result.filter(pl.col("meta_aa_changes") == "A1B").to_dicts().pop()
    assert row["f1_AUROC"] is None


# ---------------------------------------------------------------------------
# Block list
# ---------------------------------------------------------------------------


def test_aggregate_blocked_feature_excluded(simple_df: pl.DataFrame) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list={"f1_mean"},
    )
    assert "f1_mean" not in result.columns


def test_aggregate_unblocked_feature_included(simple_df: pl.DataFrame) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list={"f1_mean"},
    )
    assert "f2_mean" in result.columns


def test_aggregate_none_block_list_no_effect(simple_df: pl.DataFrame) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list=None,
    )
    assert {"f1_mean", "f2_mean"}.issubset(set(result.columns))


def test_aggregate_unknown_feature_in_block_list_ignored(
    simple_df: pl.DataFrame,
) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list={"f1_does_not_exist"},
    )
    assert {"f1_mean", "f2_mean"}.issubset(set(result.columns))


def test_aggregate_block_list_with_reference_based_aggregator(
    toy_norm_df: pl.DataFrame,
) -> None:
    result = m.aggregate(
        toy_norm_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="EMD",
        block_list={"f1_EMD"},
    )
    assert "f1_EMD" not in result.columns
    assert "f2_EMD" in result.columns


def test_main_block_list_file_excludes_features(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    bl_path = tmp_path / "block_list.parquet"
    pl.DataFrame(
        {"feature": ["f1_mean", "f2_mean"], "feature_ok": [False, True]}
    ).write_parquet(bl_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path, block_list_file=str(bl_path)))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert "f1_mean" not in result.columns
    assert "f2_mean" in result.columns


def test_main_output_contains_meta_num_cells(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert "meta_num_cells" in result.columns


def test_main_meta_num_cells_reflects_cell_level_counts(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "input.parquet")
    counts = dict(
        zip(result["meta_aa_changes"].to_list(), result["meta_num_cells"].to_list())
    )
    assert counts["A1B"] == 3


def test_main_barcode_metadata_serializes_to_parquet(tmp_path) -> None:
    write_agg_input_parquet(tmp_path, with_barcode=True)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "input.parquet")
    assert "meta_num_unique_barcodes" in result.columns
    assert "meta_barcode_counts" in result.columns
    assert result["meta_barcode_counts"].null_count() == 0


# ---------------------------------------------------------------------------
# get_aggregate_meta_data
# ---------------------------------------------------------------------------


@pytest.fixture
def meta_lf_no_barcode() -> pl.LazyFrame:
    """Three cells for label A, two for label B; no barcode column."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B"],
            "meta_is_control": [False, False, False, False, False],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).lazy()


@pytest.fixture
def meta_lf_with_barcode() -> pl.LazyFrame:
    """Three cells for label A (two unique barcodes), two for label B (one unique)."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B"],
            "meta_is_control": [False, False, False, False, False],
            "meta_barcode": ["bc1", "bc1", "bc2", "bc3", "bc3"],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).lazy()


def test_get_aggregate_meta_data_returns_lazyframe(
    meta_lf_no_barcode: pl.LazyFrame,
) -> None:
    result = m.get_aggregate_meta_data(meta_lf_no_barcode, "meta_aa_changes")
    assert isinstance(result, pl.LazyFrame)


def test_get_aggregate_meta_data_one_row_per_label(
    meta_lf_no_barcode: pl.LazyFrame,
) -> None:
    result = m.get_aggregate_meta_data(meta_lf_no_barcode, "meta_aa_changes").collect()
    assert len(result) == 2


def test_get_aggregate_meta_data_label_column_present(
    meta_lf_no_barcode: pl.LazyFrame,
) -> None:
    result = m.get_aggregate_meta_data(meta_lf_no_barcode, "meta_aa_changes").collect()
    assert "meta_aa_changes" in result.columns


def test_get_aggregate_meta_data_num_cells_correct(
    meta_lf_no_barcode: pl.LazyFrame,
) -> None:
    result = m.get_aggregate_meta_data(meta_lf_no_barcode, "meta_aa_changes").collect()
    counts = dict(
        zip(result["meta_aa_changes"].to_list(), result["meta_num_cells"].to_list())
    )
    assert counts["A"] == 3
    assert counts["B"] == 2


def test_get_aggregate_meta_data_no_barcode_columns_without_meta_barcode(
    meta_lf_no_barcode: pl.LazyFrame,
) -> None:
    result = m.get_aggregate_meta_data(meta_lf_no_barcode, "meta_aa_changes").collect()
    assert "meta_num_unique_barcodes" not in result.columns
    assert "meta_barcode_counts" not in result.columns


def test_get_aggregate_meta_data_barcode_columns_present_with_meta_barcode(
    meta_lf_with_barcode: pl.LazyFrame,
) -> None:
    result = m.get_aggregate_meta_data(
        meta_lf_with_barcode, "meta_aa_changes"
    ).collect()
    assert "meta_num_unique_barcodes" in result.columns
    assert "meta_barcode_counts" in result.columns


def test_get_aggregate_meta_data_unique_barcodes_correct(
    meta_lf_with_barcode: pl.LazyFrame,
) -> None:
    result = m.get_aggregate_meta_data(
        meta_lf_with_barcode, "meta_aa_changes"
    ).collect()
    counts = dict(
        zip(
            result["meta_aa_changes"].to_list(),
            result["meta_num_unique_barcodes"].to_list(),
        )
    )
    assert counts["A"] == 2
    assert counts["B"] == 1


def test_get_aggregate_meta_data_barcode_counts_not_null(
    meta_lf_with_barcode: pl.LazyFrame,
) -> None:
    result = m.get_aggregate_meta_data(
        meta_lf_with_barcode, "meta_aa_changes"
    ).collect()
    assert result["meta_barcode_counts"].null_count() == 0
