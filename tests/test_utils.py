# test_pipeline_utils.py
import numpy as np
import polars as pl

from fisseq_data_pipeline.utils import get_data_lf, get_feature_selector


class DummyConfig:
    def __init__(
        self,
        feature_cols,
        batch_col_name="_batch_src",
        label_col_name="_label_src",
        control_sample_query="_is_ctrl_src",
    ):
        self.feature_cols = feature_cols
        self.batch_col_name = batch_col_name
        self.label_col_name = label_col_name
        # Polars SQL expression string; should evaluate to a boolean per-row.
        self.control_sample_query = control_sample_query


def test_get_feature_selector_regex():
    lf = pl.LazyFrame(
        {
            "f_a": [1, 2],
            "f_b": [3, 4],
            "meta": [0, 1],
        }
    )
    cfg = DummyConfig(feature_cols=r"^f_")
    sel = get_feature_selector(lf, cfg)

    # Apply the selector to ensure it picks only f_a and f_b
    out = lf.select(sel).collect()
    assert out.columns == ["f_a", "f_b"]


def test_get_feature_selector_list_preserves_order_and_ignores_missing():
    lf = pl.LazyFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )
    cfg = DummyConfig(feature_cols=["a", "missing", "b"])
    sel = get_feature_selector(lf, cfg)

    out = lf.select(sel).collect()
    # Missing column gets ignored; order of present ones is preserved
    assert out.columns == ["a", "b"]
