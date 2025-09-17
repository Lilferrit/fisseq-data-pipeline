import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.filter import clean_data


def test_removes_all_null_columns():
    feature_df = pl.DataFrame(
        {
            "good": [1.0, 2.0, 3.0],
            "all_null": [None, None, None],
        }
    )
    meta_df = pl.DataFrame({"id": [1, 2, 3]})

    cleaned_features, cleaned_meta = clean_data(feature_df, meta_df)

    assert "all_null" not in cleaned_features.columns
    assert "good" in cleaned_features.columns
    # No rows dropped
    assert cleaned_features.height == 3
    assert cleaned_meta.height == 3


def test_removes_zero_variance_columns():
    feature_df = pl.DataFrame(
        {
            "constant": [5.0, 5.0, 5.0],
            "varying": [1.0, 2.0, 3.0],
        }
    )
    meta_df = pl.DataFrame({"id": [1, 2, 3]})

    cleaned_features, _ = clean_data(feature_df, meta_df)

    assert "constant" not in cleaned_features.columns
    assert "varying" in cleaned_features.columns


def test_drops_rows_with_nulls():
    feature_df = pl.DataFrame(
        {
            "a": [2.0, 1.0, None, 3.0],
            "b": [2.0, 4.0, 5.0, None],
        }
    )
    meta_df = pl.DataFrame({"id": [9, 10, 11, 12]})

    cleaned_features, cleaned_meta = clean_data(feature_df, meta_df)

    # Only first row should survive (no nulls across columns)
    assert cleaned_features.height == 2
    assert cleaned_meta.height == 2
    assert cleaned_meta["id"].to_list() == [9, 10]


def test_combined_cleaning():
    feature_df = pl.DataFrame(
        {
            "all_null": [None, None, None],
            "constant": [7.0, 7.0, 7.0],
            "valid": [1.0, 2.0, 3.0],
        }
    )
    meta_df = pl.DataFrame({"id": [101, 102, 103]})

    cleaned_features, cleaned_meta = clean_data(feature_df, meta_df)

    assert "all_null" not in cleaned_features.columns
    assert "constant" not in cleaned_features.columns
    assert "valid" in cleaned_features.columns
    assert cleaned_features.height == 3
    assert cleaned_meta.height == 3
