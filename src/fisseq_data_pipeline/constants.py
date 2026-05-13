from typing import Any

import numpy as np
import polars as pl
from polars import selectors as cs

FEATURE_SELECTOR: pl.Expr = cs.exclude("^meta_.*$")
META_SELECTOR: pl.Expr = cs.matches("^meta_.*$")
EPS: np.floating[Any] = np.finfo(np.float32).eps
CONTROL_COLUMN_NAME: str = "meta_is_control"
CONTROL_COLUMN: pl.Expr = pl.col(CONTROL_COLUMN_NAME)
