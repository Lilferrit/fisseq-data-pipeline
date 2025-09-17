# test_harmonizer.py
import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.harmonize import fit_harmonizer, harmonize


def make_toy_data(n_samples=6, n_features=3):
    # Features: two batches with distinct offsets
    batch_labels = ["A"] * (n_samples // 2) + ["B"] * (n_samples // 2)
    features = np.vstack(
        [
            np.random.normal(loc=0.0, scale=1.0, size=(n_samples // 2, n_features)),
            np.random.normal(loc=5.0, scale=1.0, size=(n_samples // 2, n_features)),
        ]
    )

    feature_df = pl.DataFrame(features, schema=[f"f{i}" for i in range(n_features)])
    meta_df = pl.DataFrame(
        {
            "_batch": batch_labels,
            "_is_control": [True] * n_samples,  # all controls for simplicity
        }
    )
    return feature_df, meta_df


def test_fit_and_harmonize_shapes():
    feature_df, meta_df = make_toy_data()

    # Fit harmonizer
    harmonizer = fit_harmonizer(feature_df, meta_df, fit_only_on_control=True)
    assert isinstance(harmonizer, dict)

    # Apply harmonization
    harmonized_df = harmonize(feature_df, meta_df, harmonizer)

    # Check shape and column alignment
    assert harmonized_df.shape == feature_df.shape
    assert harmonized_df.columns == feature_df.columns

    # Values should be floats
    for col in harmonized_df.columns:
        assert harmonized_df[col].dtype.is_float()


def test_harmonization_changes_values():
    feature_df, meta_df = make_toy_data()

    harmonizer = fit_harmonizer(feature_df, meta_df)
    harmonized_df = harmonize(feature_df, meta_df, harmonizer)

    # At least some values should be different after harmonization
    assert not np.allclose(feature_df.to_numpy(), harmonized_df.to_numpy())
