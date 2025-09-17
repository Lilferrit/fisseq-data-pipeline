# test_normalizer.py
import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.normalize import Normalizer, fit_normalizer, normalize


def make_feature_df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0],
        }
    )


def test_fit_normalizer_basic_stats():
    feature_df = make_feature_df()
    normalizer = fit_normalizer(feature_df)

    # Check type
    assert isinstance(normalizer, Normalizer)

    # Means should be correct
    expected_means = {"x": 2.0, "y": 4.0}
    for col, mean in expected_means.items():
        assert np.isclose(normalizer.means[col][0], mean)

    # Std devs should be correct
    expected_stds = {"x": np.std([1, 2, 3], ddof=1), "y": np.std([2, 4, 6], ddof=1)}
    for col, std in expected_stds.items():
        assert np.isclose(normalizer.stds[col][0], std)


def test_fit_only_on_control():
    feature_df = make_feature_df()
    meta_df = pl.DataFrame(
        {
            "_is_control": [True, False, True],
        }
    )

    normalizer = fit_normalizer(
        feature_df, meta_data_df=meta_df, fit_only_on_control=True
    )

    # Means should be computed only from rows 0 and 2
    assert np.isclose(normalizer.means["x"][0], np.mean([1.0, 3.0]))
    assert np.isclose(normalizer.means["y"][0], np.mean([2.0, 6.0]))


def test_fit_only_on_control_requires_meta():
    feature_df = make_feature_df()

    with pytest.raises(ValueError):
        fit_normalizer(feature_df, meta_data_df=None, fit_only_on_control=True)
