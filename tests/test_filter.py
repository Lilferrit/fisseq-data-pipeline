import unittest.mock

import polars as pl

import fisseq_data_pipeline.filter as filt  # to tweak MINIMUM_CLASS_MEMBERS in tests
from fisseq_data_pipeline.filter import clean_data


def test_removes_all_null_columns():
    """Columns containing only NaN or None should be removed."""
    feature_df = pl.DataFrame(
        {
            "good": [1.0, 2.0, 3.0],
            "all_null": [None, None, None],
        },
        schema={"good": pl.Float32, "all_null": pl.Float32},
    )
    meta_df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "_label": ["A", "A", "A"],
            "_batch": ["B1", "B1", "B1"],  # single group passes threshold
        }
    )

    cleaned_features, cleaned_meta = clean_data(feature_df.lazy(), meta_df.lazy())
    cleaned_features = cleaned_features.collect()
    cleaned_meta = cleaned_meta.collect()

    # all_null column dropped
    assert "all_null" not in cleaned_features.columns
    assert "good" in cleaned_features.columns

    # row count unchanged
    assert cleaned_features.height == 3
    assert cleaned_meta.height == 3


def test_drops_rows_with_nulls():
    """Rows containing any nulls or NaNs across columns should be removed."""
    feature_df = pl.DataFrame(
        {
            "a": [2.0, 1.0, None, 3.0],
            "b": [2.0, 4.0, 5.0, None],
        }
    ).select([pl.col(c).cast(pl.Float32).alias(c) for c in ["a", "b"]])

    meta_df = pl.DataFrame(
        {
            "id": [9, 10, 11, 12],
            "_label": ["A", "A", "A", "A"],
            "_batch": ["B1", "B1", "B1", "B1"],
        }
    )

    cleaned_features, cleaned_meta = clean_data(feature_df.lazy(), meta_df.lazy())
    cleaned_features = cleaned_features.collect()
    cleaned_meta = cleaned_meta.collect()

    # only rows 0 and 1 have all finite values
    assert cleaned_features.height == 2
    assert cleaned_meta.height == 2
    assert cleaned_meta["id"].to_list() == [9, 10]


def test_combined_cleaning():
    """All cleaning stages should compose correctly in sequence."""
    feature_df = pl.DataFrame(
        {
            "all_null": [None, None, None],
            "constant": [7.0, 7.0, 7.0],
            "valid": [1.0, 2.0, 3.0],
        },
        schema_overrides={
            "all_null": pl.Float32,
            "constant": pl.Float32,
            "valid": pl.Float32,
        },
    )
    meta_df = pl.DataFrame(
        {
            "id": [101, 102, 103],
            "_label": ["A", "A", "A"],
            "_batch": ["B1", "B1", "B1"],
        }
    )

    cleaned_features, cleaned_meta = clean_data(feature_df.lazy(), meta_df.lazy())
    cleaned_features = cleaned_features.collect()
    cleaned_meta = cleaned_meta.collect()

    assert "all_null" not in cleaned_features.columns
    assert all(col in cleaned_features.columns for col in ("constant", "valid"))
    assert cleaned_features.height == 3
    assert cleaned_meta.height == 3


def test_enforces_minimum_members(monkeypatch):
    """Groups with sample counts < MINIMUM_CLASS_MEMBERS should be dropped."""
    monkeypatch.setattr(filt, "MINIMUM_CLASS_MEMBERS", 2, raising=False)

    # Build 5 rows across three (label, batch) groups:
    #  ("A","B1") -> 2 samples → keep
    #  ("B","B1") -> 1 sample  → drop
    #  ("B","B2") -> 2 samples → keep
    feature_df = pl.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5]})
    meta_df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "_label": ["A", "A", "B", "B", "B"],
            "_batch": ["B1", "B1", "B1", "B2", "B2"],
        }
    )

    # run full cleaning pipeline
    cleaned_features, cleaned_meta = clean_data(feature_df.lazy(), meta_df.lazy())
    # then explicitly enforce min-members rule (redundant but explicit)
    cleaned_features, cleaned_meta = drop_rows_infrequent_pairs(
        cleaned_features, cleaned_meta
    )

    cleaned_features = cleaned_features.collect()
    cleaned_meta = cleaned_meta.collect()

    # expect to drop the lone ("B","B1") row with id==3
    assert cleaned_features.height == 4
    assert cleaned_meta.height == 4
    assert cleaned_meta["id"].to_list() == [1, 2, 4, 5]


def test_custom_filter():
    mock_filter = unittest.mock.MagicMock()
    data_df = unittest.mock.MagicMock(spec_set=[])
    meta_df = unittest.mock.MagicMock(spec_set=[])
    mock_filter.side_effect = lambda *args: args

    data_df, meta_df = clean_data(data_df, meta_df, stages=[mock_filter])
    mock_filter.assert_called_once()
