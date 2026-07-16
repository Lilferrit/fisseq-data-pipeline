from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import scipy.stats
import sklearn.metrics
from omegaconf import OmegaConf

import fisseq_data_pipeline.aggregate as m
from fisseq_data_pipeline.aggregate import AggregateConfig
from fisseq_data_pipeline.utils.constants import CONTROL_COLUMN_NAME, IMPACT_SCORE_COL

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
# KSAggregator / AUROCAggregator / QQCorrelationAggregator — native vs.
# scipy/sklearn ground truth
# ---------------------------------------------------------------------------


@pytest.fixture
def native_stats_df() -> pl.DataFrame:
    """
    Reference pool (WT, continuous) plus three variant groups exercising
    different value shapes: RANDOM (continuous, no ties), TIES (repeated
    integer values), SINGLE (one distinct value repeated).
    """
    rng = np.random.default_rng(0)
    ref_vals = rng.standard_normal(40).tolist()
    random_vals = rng.standard_normal(12).tolist()
    tie_vals = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0]
    single_vals = [5.0] * 4

    labels = (
        ["WT"] * len(ref_vals)
        + ["RANDOM"] * len(random_vals)
        + ["TIES"] * len(tie_vals)
        + ["SINGLE"] * len(single_vals)
    )
    values = ref_vals + random_vals + tie_vals + single_vals
    return pl.DataFrame(
        {
            "meta_aa_changes": labels,
            "meta_is_control": [lbl == "WT" for lbl in labels],
            "f1": values,
        }
    )


def _group_and_ref(df: pl.DataFrame, label: str) -> tuple[list[float], list[float]]:
    ref = df.filter(pl.col("meta_is_control"))["f1"].to_list()
    group = df.filter(pl.col("meta_aa_changes") == label)["f1"].to_list()
    return group, ref


def test_ks_aggregator_returns_expected_columns(toy_norm_df: pl.DataFrame) -> None:
    result = m.KSAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert {"meta_aa_changes", "f1_KS", "f2_KS"}.issubset(set(result.columns))


def test_ks_aggregator_excludes_control_rows(toy_norm_df: pl.DataFrame) -> None:
    result = m.KSAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert "WT" not in result["meta_aa_changes"].to_list()


@pytest.mark.parametrize("label", ["RANDOM", "TIES", "SINGLE"])
def test_ks_aggregator_matches_scipy(native_stats_df: pl.DataFrame, label: str) -> None:
    result = m.KSAggregator().aggregate(native_stats_df.lazy()).collect()
    row = _get_row(result, label)
    group, ref = _group_and_ref(native_stats_df, label)
    expected = scipy.stats.ks_2samp(group, ref).statistic
    assert row["f1_KS"] == pytest.approx(expected, abs=1e-9)


def test_auroc_aggregator_returns_expected_columns(toy_norm_df: pl.DataFrame) -> None:
    result = m.AUROCAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert {"meta_aa_changes", "f1_AUROC", "f2_AUROC"}.issubset(set(result.columns))


def test_auroc_aggregator_excludes_control_rows(toy_norm_df: pl.DataFrame) -> None:
    result = m.AUROCAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert "WT" not in result["meta_aa_changes"].to_list()


@pytest.mark.parametrize("label", ["RANDOM", "TIES", "SINGLE"])
def test_auroc_aggregator_matches_sklearn_unsymmetrized(
    native_stats_df: pl.DataFrame, label: str
) -> None:
    """Raw (un-symmetrized) sklearn.metrics.roc_auc_score — no `1 - auroc` folding."""
    result = m.AUROCAggregator().aggregate(native_stats_df.lazy()).collect()
    row = _get_row(result, label)
    group, ref = _group_and_ref(native_stats_df, label)
    labels = [0] * len(ref) + [1] * len(group)
    expected = sklearn.metrics.roc_auc_score(labels, ref + group)
    assert row["f1_AUROC"] == pytest.approx(expected, abs=1e-9)


def test_auroc_aggregator_directional_higher_approaches_one() -> None:
    """Variant consistently higher than reference -> AUROC near 1.0."""
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 5 + ["A"] * 5,
            "meta_is_control": [True] * 5 + [False] * 5,
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0] + [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
    row = _get_row(m.AUROCAggregator().aggregate(df.lazy()).collect(), "A")
    assert row["f1_AUROC"] == pytest.approx(1.0)


def test_auroc_aggregator_directional_lower_approaches_zero() -> None:
    """Variant consistently lower than reference -> AUROC near 0.0.

    Regression guard for symmetrization: the old ``if auroc < 0.5: auroc =
    1 - auroc`` behavior would have folded this to ~1.0 instead.
    """
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 5 + ["A"] * 5,
            "meta_is_control": [True] * 5 + [False] * 5,
            "f1": [10.0, 11.0, 12.0, 13.0, 14.0] + [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    row = _get_row(m.AUROCAggregator().aggregate(df.lazy()).collect(), "A")
    assert row["f1_AUROC"] == pytest.approx(0.0)


def test_auroc_aggregator_fully_overlapping_near_half() -> None:
    """Variant and reference drawn from the identical set of values -> AUROC near 0.5."""
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 6 + ["A"] * 6,
            "meta_is_control": [True] * 6 + [False] * 6,
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
        }
    )
    row = _get_row(m.AUROCAggregator().aggregate(df.lazy()).collect(), "A")
    assert row["f1_AUROC"] == pytest.approx(0.5)


def test_qq_aggregator_returns_expected_columns(toy_norm_df: pl.DataFrame) -> None:
    result = m.QQCorrelationAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert {"meta_aa_changes", "f1_QQ", "f2_QQ"}.issubset(set(result.columns))


def test_qq_aggregator_excludes_control_rows(toy_norm_df: pl.DataFrame) -> None:
    result = m.QQCorrelationAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert "WT" not in result["meta_aa_changes"].to_list()


@pytest.mark.parametrize("label", ["RANDOM", "TIES"])
def test_qq_aggregator_matches_scipy_default_n_quantiles(
    native_stats_df: pl.DataFrame, label: str
) -> None:
    result = m.QQCorrelationAggregator().aggregate(native_stats_df.lazy()).collect()
    row = _get_row(result, label)
    group, ref = _group_and_ref(native_stats_df, label)
    probs = np.linspace(0, 1, 100)
    expected = scipy.stats.pearsonr(
        np.quantile(group, probs), np.quantile(ref, probs)
    ).statistic
    assert row["f1_QQ"] == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("label", ["RANDOM", "TIES"])
def test_qq_aggregator_matches_scipy_custom_n_quantiles(
    native_stats_df: pl.DataFrame, label: str
) -> None:
    n_quantiles = 17
    result = (
        m.QQCorrelationAggregator(n_quantiles=n_quantiles)
        .aggregate(native_stats_df.lazy())
        .collect()
    )
    row = _get_row(result, label)
    group, ref = _group_and_ref(native_stats_df, label)
    probs = np.linspace(0, 1, n_quantiles)
    expected = scipy.stats.pearsonr(
        np.quantile(group, probs), np.quantile(ref, probs)
    ).statistic
    assert row["f1_QQ"] == pytest.approx(expected, abs=1e-8)


def test_qq_aggregator_single_value_group_returns_null(
    native_stats_df: pl.DataFrame,
) -> None:
    """A constant-valued group has an exactly constant quantile profile, so
    the correlation is mathematically undefined (matches scipy's
    ConstantInputWarning -> nan -> None convention)."""
    result = m.QQCorrelationAggregator().aggregate(native_stats_df.lazy()).collect()
    row = _get_row(result, "SINGLE")
    assert row["f1_QQ"] is None


def test_qq_aggregator_constant_reference_returns_null() -> None:
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 5 + ["A"] * 3,
            "meta_is_control": [True] * 5 + [False] * 3,
            "f1": [7.0] * 5 + [1.0, 2.0, 3.0],
        }
    )
    row = _get_row(m.QQCorrelationAggregator().aggregate(df.lazy()).collect(), "A")
    assert row["f1_QQ"] is None


# ---------------------------------------------------------------------------
# _collect_reference_pool
# ---------------------------------------------------------------------------


def test_reference_pool_collected_once(toy_norm_df: pl.DataFrame) -> None:
    with patch(
        "fisseq_data_pipeline.aggregate._collect_reference_pool",
        wraps=m._collect_reference_pool,
    ) as spy:
        m.KSAggregator().aggregate(toy_norm_df.lazy()).collect()
    assert spy.call_count == 1


def test_reference_pool_values_match_control_rows(toy_norm_df: pl.DataFrame) -> None:
    pool = m._collect_reference_pool(toy_norm_df.lazy(), ["f1", "f2"])
    expected_f1 = toy_norm_df.filter(pl.col("meta_is_control"))["f1"].to_list()
    expected_f2 = toy_norm_df.filter(pl.col("meta_is_control"))["f2"].to_list()
    assert pool["f1"].tolist() == expected_f1
    assert pool["f2"].tolist() == expected_f2


def test_reference_pool_none_becomes_nan_but_clean_still_drops_it() -> None:
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "WT"],
            "meta_is_control": [True, True, True],
            "f1": pl.Series([1.0, None, 3.0], dtype=pl.Float64),
        }
    )
    pool = m._collect_reference_pool(df.lazy(), ["f1"])
    assert np.isnan(pool["f1"][1])
    assert m._clean(pool["f1"]).tolist() == [1.0, 3.0]


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
    result = m.MeanAggregator().aggregate(simple_df.lazy()).collect()
    assert {"meta_aa_changes", "f1_mean", "f2_mean"}.issubset(set(result.columns))
    row_a = _get_row(result, "A")
    assert row_a["f1_mean"] == pytest.approx(np.mean([1.0, 2.0, 3.0]))
    assert row_a["f2_mean"] == pytest.approx(np.mean([4.0, 5.0, 6.0]))
    row_b = _get_row(result, "B")
    assert row_b["f1_mean"] == pytest.approx(np.mean([10.0, 20.0, 30.0]))
    assert row_b["f2_mean"] == pytest.approx(np.mean([40.0, 50.0, 60.0]))


def test_median_aggregator(simple_df: pl.DataFrame) -> None:
    result = m.MedianAggregator().aggregate(simple_df.lazy()).collect()
    assert {"meta_aa_changes", "f1_median", "f2_median"}.issubset(set(result.columns))
    row_a = _get_row(result, "A")
    assert row_a["f1_median"] == pytest.approx(np.median([1.0, 2.0, 3.0]))
    assert row_a["f2_median"] == pytest.approx(np.median([4.0, 5.0, 6.0]))


def test_mad_aggregator(simple_df: pl.DataFrame) -> None:
    result = m.MADAggregator().aggregate(simple_df.lazy()).collect()
    assert {"meta_aa_changes", "f1_MAD", "f2_MAD"}.issubset(set(result.columns))
    row_a = _get_row(result, "A")
    vals = np.array([1.0, 2.0, 3.0])
    expected_mad = np.median(np.abs(vals - np.median(vals)))
    assert row_a["f1_MAD"] == pytest.approx(expected_mad)


def test_std_aggregator(simple_df: pl.DataFrame) -> None:
    result = m.StdAggregator().aggregate(simple_df.lazy()).collect()
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
        result = agg_cls().aggregate(df.lazy()).collect()
        assert "WT" not in result["meta_aa_changes"].to_list()


# ---------------------------------------------------------------------------
# Native aggregators — edge cases
# ---------------------------------------------------------------------------


@pytest.fixture
def single_value_group_df() -> pl.DataFrame:
    """Group A has three values; group B has exactly one (std edge case)."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B"],
            "meta_is_control": [False, False, False, False],
            "f1": [1.0, 2.0, 3.0, 5.0],
        }
    )


@pytest.fixture
def nan_inf_group_df() -> pl.DataFrame:
    """Group A mixes finite values with None, NaN, and Inf."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "A"],
            "meta_is_control": [False, False, False, False],
            "f1": pl.Series([1.0, None, float("nan"), float("inf")], dtype=pl.Float64),
        }
    ).vstack(
        pl.DataFrame(
            {
                "meta_aa_changes": ["A"],
                "meta_is_control": [False],
                "f1": pl.Series([2.0], dtype=pl.Float64),
            }
        )
    )


def test_std_native_single_value_group_returns_null(
    single_value_group_df: pl.DataFrame,
) -> None:
    result = m.StdAggregator().aggregate(single_value_group_df.lazy()).collect()
    row_b = _get_row(result, "B")
    assert row_b["f1_std"] is None


def test_mad_native_matches_numpy_with_nan_inf_present(
    nan_inf_group_df: pl.DataFrame,
) -> None:
    result = m.MADAggregator().aggregate(nan_inf_group_df.lazy()).collect()
    row_a = _get_row(result, "A")
    finite_vals = np.array([1.0, 2.0])
    expected = np.median(np.abs(finite_vals - np.median(finite_vals)))
    assert row_a["f1_MAD"] == pytest.approx(expected)


def test_std_native_matches_numpy_with_nan_inf_present(
    nan_inf_group_df: pl.DataFrame,
) -> None:
    result = m.StdAggregator().aggregate(nan_inf_group_df.lazy()).collect()
    row_a = _get_row(result, "A")
    finite_vals = np.array([1.0, 2.0])
    expected = np.std(finite_vals, ddof=1)
    assert row_a["f1_std"] == pytest.approx(expected)


def test_reference_pool_not_collected_for_native_aggregators(
    simple_df: pl.DataFrame,
) -> None:
    for agg_cls in (
        m.MeanAggregator,
        m.MedianAggregator,
        m.StdAggregator,
        m.MADAggregator,
    ):
        with patch(
            "fisseq_data_pipeline.aggregate._collect_reference_pool",
            wraps=m._collect_reference_pool,
        ) as spy:
            agg_cls().aggregate(simple_df.lazy()).collect()
        assert spy.call_count == 0


def test_reference_pool_still_collected_for_reference_based_aggregators(
    toy_norm_df: pl.DataFrame,
) -> None:
    for agg_cls in (m.KSAggregator, m.QQCorrelationAggregator, m.AUROCAggregator):
        with patch(
            "fisseq_data_pipeline.aggregate._collect_reference_pool",
            wraps=m._collect_reference_pool,
        ) as spy:
            agg_cls().aggregate(toy_norm_df.lazy()).collect()
        assert spy.call_count == 1


def test_native_and_batching_combine_correctly(many_features_df: pl.DataFrame) -> None:
    unbatched = m.MeanAggregator().aggregate(many_features_df.lazy()).collect()
    batched = (
        m.MeanAggregator()
        .aggregate(many_features_df.lazy(), feature_batch_size=2)
        .collect()
    )
    assert _sorted(unbatched).equals(_sorted(batched))


# ---------------------------------------------------------------------------
# aggregate() function
# ---------------------------------------------------------------------------


def test_aggregate_ks_returns_expected_columns(toy_norm_df: pl.DataFrame) -> None:
    result = m.aggregate(
        toy_norm_df.lazy(), label_col="meta_aa_changes", aggregator_name="KS"
    ).collect()
    assert {"meta_aa_changes", "f1_KS", "f2_KS"}.issubset(set(result.columns))


def test_aggregate_mean_returns_expected_columns(simple_df: pl.DataFrame) -> None:
    result = m.aggregate(
        simple_df.lazy(), label_col="meta_aa_changes", aggregator_name="mean"
    ).collect()
    assert {"meta_aa_changes", "f1_mean", "f2_mean"}.issubset(set(result.columns))


def test_aggregate_ks_excludes_control_rows(toy_norm_df: pl.DataFrame) -> None:
    result = m.aggregate(
        toy_norm_df.lazy(), label_col="meta_aa_changes", aggregator_name="KS"
    ).collect()
    assert "WT" not in result["meta_aa_changes"].to_list()


def test_aggregate_unknown_raises() -> None:
    lf = pl.DataFrame(
        {"meta_aa_changes": ["A"], "meta_is_control": [False], "f1": [1.0]}
    ).lazy()
    with pytest.raises(ValueError, match="Unknown aggregator"):
        m.aggregate(lf, label_col="meta_aa_changes", aggregator_name="bogus")


# ---------------------------------------------------------------------------
# feature_batch_size
# ---------------------------------------------------------------------------


@pytest.fixture
def many_features_df() -> pl.DataFrame:
    """Two variant groups, six feature columns (f1..f6), no control rows."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B", "B"],
            "meta_is_control": [False, False, False, False, False, False],
            "f1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "f2": [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
            "f3": [7.0, 8.0, 9.0, 70.0, 80.0, 90.0],
            "f4": [1.5, 2.5, 3.5, 15.0, 25.0, 35.0],
            "f5": [2.0, 3.0, 4.0, 20.0, 30.0, 40.0],
            "f6": [0.5, 1.5, 2.5, 5.0, 15.0, 25.0],
        }
    )


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(sorted(df.columns)).sort("meta_aa_changes")


def test_feature_batch_size_none_matches_unbatched(simple_df: pl.DataFrame) -> None:
    unbatched = m.MeanAggregator().aggregate(simple_df.lazy()).collect()
    explicit_none = (
        m.MeanAggregator()
        .aggregate(simple_df.lazy(), feature_batch_size=None)
        .collect()
    )
    assert _sorted(unbatched).equals(_sorted(explicit_none))


def test_feature_batch_size_splits_and_rejoins_identically(
    many_features_df: pl.DataFrame,
) -> None:
    unbatched = m.MeanAggregator().aggregate(many_features_df.lazy()).collect()
    batched = (
        m.MeanAggregator()
        .aggregate(many_features_df.lazy(), feature_batch_size=2)
        .collect()
    )
    assert _sorted(unbatched).equals(_sorted(batched))


@pytest.mark.parametrize(
    "agg_cls", [m.KSAggregator, m.QQCorrelationAggregator, m.AUROCAggregator]
)
def test_feature_batch_size_with_reference_based_aggregator(
    toy_norm_df: pl.DataFrame, agg_cls
) -> None:
    unbatched = agg_cls().aggregate(toy_norm_df.lazy()).collect()
    batched = agg_cls().aggregate(toy_norm_df.lazy(), feature_batch_size=1).collect()
    assert _sorted(unbatched).equals(_sorted(batched))


def test_feature_batch_size_preserves_all_labels(
    many_features_df: pl.DataFrame,
) -> None:
    unbatched = m.MeanAggregator().aggregate(many_features_df.lazy()).collect()
    batched = (
        m.MeanAggregator()
        .aggregate(many_features_df.lazy(), feature_batch_size=2)
        .collect()
    )
    assert set(batched["meta_aa_changes"]) == set(unbatched["meta_aa_changes"])


@pytest.mark.parametrize("size", [0, -1])
def test_feature_batch_size_zero_or_negative_is_noop(
    simple_df: pl.DataFrame, size: int
) -> None:
    unbatched = m.MeanAggregator().aggregate(simple_df.lazy()).collect()
    result = (
        m.MeanAggregator()
        .aggregate(simple_df.lazy(), feature_batch_size=size)
        .collect()
    )
    assert _sorted(unbatched).equals(_sorted(result))


def test_aggregate_feature_batch_size_threads_through(
    many_features_df: pl.DataFrame,
) -> None:
    unbatched = m.aggregate(
        many_features_df.lazy(), label_col="meta_aa_changes", aggregator_name="mean"
    ).collect()
    batched = m.aggregate(
        many_features_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        feature_batch_size=2,
    ).collect()
    assert _sorted(unbatched).equals(_sorted(batched))


def test_aggregate_config_feature_batch_size_default_is_500() -> None:
    # Checked directly on the dataclass field rather than through
    # make_agg_cfg(), which always passes its own explicit
    # feature_batch_size=200 test-helper default regardless of what
    # AggregateConfig's own field default is.
    field = next(
        f for f in dataclasses.fields(AggregateConfig) if f.name == "feature_batch_size"
    )
    assert field.default == 500


def test_feature_type_aggregate_config_feature_batch_size_default_is_500() -> None:
    field = next(
        f
        for f in dataclasses.fields(m.FeatureTypeAggregateConfig)
        if f.name == "feature_batch_size"
    )
    assert field.default == 500


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
    compute_impact_score=True,
    feature_batch_size=200,
) -> OmegaConf:
    """Return a DictConfig for AggregateConfig with sensible test defaults."""
    return OmegaConf.structured(
        AggregateConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            input_file=str(tmp_path / "input.parquet"),
            save_normalizer=save_normalizer,
            aggregator=aggregator,
            block_list_file=block_list_file,
            compute_impact_score=compute_impact_score,
            feature_batch_size=feature_batch_size,
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
    assert (tmp_path / "out" / "input.parquet").exists()


def test_main_output_contains_label_column(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "meta_aa_changes" in result.columns


def test_main_synonymous_rows_normalized_to_zero_mean(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
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
    assert (tmp_path / "out" / "normalizer.parquet").exists()


def test_main_feature_batch_size_end_to_end(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path, feature_batch_size=1))
    batched = pl.read_parquet(tmp_path / "out" / "input.parquet")

    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    unbatched = pl.read_parquet(tmp_path / "out" / "input.parquet")

    assert set(batched.columns) == set(unbatched.columns)
    assert _sorted(batched).equals(_sorted(unbatched))


# ---------------------------------------------------------------------------
# Null handling — native aggregators
# ---------------------------------------------------------------------------


def test_mean_aggregator_ignores_nulls(null_df: pl.DataFrame) -> None:
    result = m.MeanAggregator().aggregate(null_df.lazy()).collect()
    row_a = _get_row(result, "A")
    assert row_a["f1_mean"] == pytest.approx(2.0)  # mean of [1, 3]
    assert row_a["f2_mean"] == pytest.approx(4.5)  # mean of [4, 5]


def test_mean_aggregator_all_null_returns_null(null_df: pl.DataFrame) -> None:
    result = m.MeanAggregator().aggregate(null_df.lazy()).collect()
    row_b = _get_row(result, "B")
    assert row_b["f1_mean"] is None


def test_median_aggregator_ignores_nulls(null_df: pl.DataFrame) -> None:
    result = m.MedianAggregator().aggregate(null_df.lazy()).collect()
    row_a = _get_row(result, "A")
    assert row_a["f1_median"] == pytest.approx(2.0)  # median of [1, 3]


def test_median_aggregator_all_null_returns_null(null_df: pl.DataFrame) -> None:
    result = m.MedianAggregator().aggregate(null_df.lazy()).collect()
    row_b = _get_row(result, "B")
    assert row_b["f1_median"] is None


def test_mad_aggregator_ignores_nulls(null_df: pl.DataFrame) -> None:
    result = m.MADAggregator().aggregate(null_df.lazy()).collect()
    row_a = _get_row(result, "A")
    vals = np.array([1.0, 3.0])
    expected_mad = np.median(np.abs(vals - np.median(vals)))
    assert row_a["f1_MAD"] == pytest.approx(expected_mad)


def test_mad_aggregator_all_null_returns_null(null_df: pl.DataFrame) -> None:
    result = m.MADAggregator().aggregate(null_df.lazy()).collect()
    row_b = _get_row(result, "B")
    assert row_b["f1_MAD"] is None


def test_std_aggregator_ignores_nulls(null_df: pl.DataFrame) -> None:
    result = m.StdAggregator().aggregate(null_df.lazy()).collect()
    row_a = _get_row(result, "A")
    assert row_a["f1_std"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


def test_std_aggregator_all_null_returns_null(null_df: pl.DataFrame) -> None:
    result = m.StdAggregator().aggregate(null_df.lazy()).collect()
    row_b = _get_row(result, "B")
    assert row_b["f1_std"] is None


# ---------------------------------------------------------------------------
# Null handling — reference-based aggregators
# ---------------------------------------------------------------------------


def _ref_based_null_df() -> pl.DataFrame:
    """Variant group A with one null; reference with one null. f2 all-null in variant."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "WT", "A1B", "A1B", "A1B"],
            "meta_is_control": [True, True, True, False, False, False],
            "f1": pl.Series([1.0, None, 3.0, 10.0, None, 30.0], dtype=pl.Float64),
            "f2": pl.Series([5.0, 6.0, 7.0, None, None, None], dtype=pl.Float64),
        }
    )


def test_ks_aggregator_ignores_nulls_in_variant() -> None:
    full = _ref_based_null_df()
    row = (
        m.KSAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    expected = scipy.stats.ks_2samp([10.0, 30.0], [1.0, 3.0]).statistic
    assert row["f1_KS"] == pytest.approx(expected)


def test_ks_aggregator_all_null_variant_returns_null() -> None:
    full = _ref_based_null_df()
    row = (
        m.KSAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    assert row["f2_KS"] is None


def test_ks_aggregator_all_null_reference_returns_null() -> None:
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False],
            "f1": pl.Series([None, None, 1.0, 2.0], dtype=pl.Float64),
        }
    )
    row = (
        m.KSAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    assert row["f1_KS"] is None


def test_qq_aggregator_ignores_nulls_in_variant() -> None:
    full = _ref_based_null_df()
    row = (
        m.QQCorrelationAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    assert row["f1_QQ"] is not None


def test_qq_aggregator_all_null_variant_returns_null() -> None:
    full = _ref_based_null_df()
    row = (
        m.QQCorrelationAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    assert row["f2_QQ"] is None


def test_qq_aggregator_all_null_reference_returns_null() -> None:
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False],
            "f1": pl.Series([None, None, 1.0, 2.0], dtype=pl.Float64),
        }
    )
    row = (
        m.QQCorrelationAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    assert row["f1_QQ"] is None


def test_auroc_aggregator_ignores_nulls_in_variant() -> None:
    full = _ref_based_null_df()
    row = (
        m.AUROCAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    assert row["f1_AUROC"] is not None


def test_auroc_aggregator_all_null_variant_returns_null() -> None:
    full = _ref_based_null_df()
    row = (
        m.AUROCAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    assert row["f2_AUROC"] is None


def test_auroc_aggregator_all_null_reference_returns_null() -> None:
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False],
            "f1": pl.Series([None, None, 1.0, 2.0], dtype=pl.Float64),
        }
    )
    row = (
        m.AUROCAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    assert row["f1_AUROC"] is None


# ---------------------------------------------------------------------------
# Infinity / NaN handling — reference-based aggregators
# ---------------------------------------------------------------------------


def test_auroc_aggregator_inf_in_variant_returns_null_not_exception() -> None:
    """inf in variant values must be silently dropped, not crash sklearn."""
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "WT", "A1B", "A1B", "A1B"],
            "meta_is_control": [True, True, True, False, False, False],
            "f1": pl.Series([1.0, 2.0, 3.0, float("inf"), 1.0, 2.0], dtype=pl.Float64),
        }
    )
    row = (
        m.AUROCAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    assert row["f1_AUROC"] is not None


def test_auroc_aggregator_all_inf_in_variant_returns_null() -> None:
    """When all variant values are inf, result must be null, not an exception."""
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False],
            "f1": pl.Series([1.0, 2.0, float("inf"), float("inf")], dtype=pl.Float64),
        }
    )
    row = (
        m.AUROCAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
    assert row["f1_AUROC"] is None


def test_auroc_aggregator_inf_in_reference_returns_null() -> None:
    """When all reference values are inf, result must be null, not an exception."""
    full = pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False],
            "f1": pl.Series([float("inf"), float("inf"), 1.0, 2.0], dtype=pl.Float64),
        }
    )
    row = (
        m.AUROCAggregator()
        .aggregate(full.lazy())
        .filter(pl.col("meta_aa_changes") == "A1B")
        .collect()
        .to_dicts()
        .pop()
    )
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
    ).collect()
    assert "f1_mean" not in result.columns


def test_aggregate_unblocked_feature_included(simple_df: pl.DataFrame) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list={"f1_mean"},
    ).collect()
    assert "f2_mean" in result.columns


def test_aggregate_none_block_list_no_effect(simple_df: pl.DataFrame) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list=None,
    ).collect()
    assert {"f1_mean", "f2_mean"}.issubset(set(result.columns))


def test_aggregate_unknown_feature_in_block_list_ignored(
    simple_df: pl.DataFrame,
) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list={"f1_does_not_exist"},
    ).collect()
    assert {"f1_mean", "f2_mean"}.issubset(set(result.columns))


@pytest.mark.parametrize("aggregator_name", ["KS", "QQ", "AUROC"])
def test_aggregate_block_list_with_reference_based_aggregator(
    toy_norm_df: pl.DataFrame, aggregator_name: str
) -> None:
    result = m.aggregate(
        toy_norm_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name=aggregator_name,
        block_list={f"f1_{aggregator_name}"},
    ).collect()
    assert f"f1_{aggregator_name}" not in result.columns
    assert f"f2_{aggregator_name}" in result.columns


def test_main_block_list_file_excludes_features(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    bl_path = tmp_path / "block_list.parquet"
    pl.DataFrame(
        {"feature": ["f1_mean", "f2_mean"], "feature_ok": [False, True]}
    ).write_parquet(bl_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path, block_list_file=str(bl_path)))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "f1_mean" not in result.columns
    assert "f2_mean" in result.columns


def test_main_output_contains_meta_num_cells(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "meta_num_cells" in result.columns


def test_main_meta_num_cells_reflects_cell_level_counts(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    counts = dict(
        zip(result["meta_aa_changes"].to_list(), result["meta_num_cells"].to_list())
    )
    assert counts["A1B"] == 3


def test_main_barcode_metadata_serializes_to_parquet(tmp_path) -> None:
    write_agg_input_parquet(tmp_path, with_barcode=True)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "meta_barcode_num_unique" in result.columns
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
    assert "meta_barcode_num_unique" in result.columns
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
            result["meta_barcode_num_unique"].to_list(),
        )
    )
    assert counts["A"] == 2
    assert counts["B"] == 1


# ---------------------------------------------------------------------------
# compute_impact_score — main() integration
# ---------------------------------------------------------------------------


def write_agg_input_parquet_asymmetric(tmp_path) -> None:
    """Cell-level data with 3 asymmetric synonymous controls.

    Three synonymous variants (A1A, A2A, A3A) with unevenly spaced feature
    values ensure the control median after Z-score normalization is non-zero,
    avoiding NaN impact scores.
    """
    pl.DataFrame(
        {
            "meta_aa_changes": (
                ["WT"] * 3 + ["A1A"] * 3 + ["A2A"] * 3 + ["A3A"] * 3 + ["A1B"] * 3
            ),
            "meta_is_control": [True] * 3 + [False] * 12,
            "f1": [0.0] * 3 + [1.0] * 3 + [2.0] * 3 + [6.0] * 3 + [20.0] * 3,
            "f2": [0.0] * 3 + [1.0] * 3 + [4.0] * 3 + [1.0] * 3 + [30.0] * 3,
        }
    ).write_parquet(tmp_path / "input.parquet")


def test_main_impact_score_column_present_by_default(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert IMPACT_SCORE_COL in result.columns


def test_main_impact_score_column_absent_when_disabled(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path, compute_impact_score=False))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert IMPACT_SCORE_COL not in result.columns


def test_main_impact_score_values_are_finite(tmp_path) -> None:
    # Asymmetric synonymous controls guarantee a non-zero control median after
    # Z-score normalization, so impact scores are finite rather than NaN.
    write_agg_input_parquet_asymmetric(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert result[IMPACT_SCORE_COL].is_finite().all()


def test_main_impact_score_in_unit_interval(tmp_path) -> None:
    write_agg_input_parquet_asymmetric(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    scores = result[IMPACT_SCORE_COL]
    assert (scores >= 0).all() and (scores <= 1).all()


def test_get_aggregate_meta_data_barcode_counts_not_null(
    meta_lf_with_barcode: pl.LazyFrame,
) -> None:
    result = m.get_aggregate_meta_data(
        meta_lf_with_barcode, "meta_aa_changes"
    ).collect()
    assert result["meta_barcode_counts"].null_count() == 0


# ---------------------------------------------------------------------------
# feature_type_main
# ---------------------------------------------------------------------------


def make_ft_cfg(
    tmp_path,
    *,
    output_root=None,
    aggregator="mean",
    index_file=None,
    feature_batch_size=200,
) -> OmegaConf:
    """Return a DictConfig for FeatureTypeAggregateConfig with test defaults."""
    return OmegaConf.structured(
        m.FeatureTypeAggregateConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            input_file=str(tmp_path / "input.parquet"),
            aggregator=aggregator,
            index_file=index_file,
            feature_batch_size=feature_batch_size,
        )
    )


def test_feature_type_main_output_has_only_label_and_stat_columns(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.feature_type_main.__wrapped__(make_ft_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert set(result.columns) == {"meta_aa_changes", "f1_mean", "f2_mean"}


def test_feature_type_main_index_file_none_aggregates_all_rows(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.feature_type_main.__wrapped__(make_ft_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    # All four groups (WT is control and excluded; A1A, A2A, A1B remain).
    assert set(result["meta_aa_changes"].to_list()) == {"A1A", "A2A", "A1B"}


def test_feature_type_main_index_file_filters_rows(tmp_path) -> None:
    # Custom dataset (unlike write_agg_input_parquet, whose per-group values
    # are constant, which would make a single-row filter indistinguishable
    # from the full-group aggregate): A1B has three distinct f1 values, so
    # filtering to a subset changes the aggregated mean.
    pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False, False],
            "f1": [0.0, 0.0, 10.0, 20.0, 30.0],
        }
    ).write_parquet(tmp_path / "input.parquet")
    # Row index 2 is the first A1B row (f1=10.0); rows 0-1 are WT.
    idx_path = tmp_path / "half1.parquet"
    pl.DataFrame({"tmp_cell_idx": [2]}).write_parquet(idx_path)

    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.feature_type_main.__wrapped__(
            make_ft_cfg(
                tmp_path,
                index_file=str(idx_path),
                output_root=str(tmp_path / "filtered"),
            )
        )
    filtered_result = pl.read_parquet(tmp_path / "filtered.input.parquet")

    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.feature_type_main.__wrapped__(
            make_ft_cfg(tmp_path, output_root=str(tmp_path / "unfiltered"))
        )
    unfiltered_result = pl.read_parquet(tmp_path / "unfiltered.input.parquet")

    # Filtering to a single A1B row means only that A1B row contributes, and
    # its single-cell mean must equal the raw feature value at that row exactly.
    assert set(filtered_result["meta_aa_changes"].to_list()) == {"A1B"}
    filtered_row = filtered_result.filter(pl.col("meta_aa_changes") == "A1B").row(
        0, named=True
    )
    assert filtered_row["f1_mean"] == pytest.approx(10.0)

    unfiltered_row = unfiltered_result.filter(pl.col("meta_aa_changes") == "A1B").row(
        0, named=True
    )
    assert filtered_row["f1_mean"] != pytest.approx(unfiltered_row["f1_mean"])


def test_feature_type_main_output_root_naming(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.feature_type_main.__wrapped__(make_ft_cfg(tmp_path, output_root=root))
    assert (tmp_path / "run1.input.parquet").exists()
