"""Shared column-name constants and Polars selectors used across the pipeline.

Defines the ``meta_*`` column name constants, the ``FEATURE_SELECTOR`` /
``META_SELECTOR`` Polars selectors distinguishing feature columns from metadata
columns, and the floating-point epsilon used for near-zero-variance checks.
"""

from typing import Any

import numpy as np
import polars as pl
from polars import selectors as cs

FEATURE_SELECTOR: pl.Expr = cs.exclude("^meta_.*$")
META_SELECTOR: pl.Expr = cs.matches("^meta_.*$")
EPS: np.floating[Any] = np.finfo(np.float32).eps
CONTROL_COLUMN_NAME: str = "meta_is_control"
CONTROL_COLUMN: pl.Expr = pl.col(CONTROL_COLUMN_NAME)
META_BARCODE_COL: str = "meta_barcode"
META_BATCH_COL: str = "meta_batch"
META_EDIT_DISTANCE_COL: str = "meta_edit_distance"
META_VARIANT_TAG_COL: str = "meta_variant_tag"
META_VARIANT_CLASS: str = "meta_variant_class"
IMPACT_SCORE_COL: str = "meta_impact_score"
