"""Generate a synthetic cell-level parquet at production shape.

No real FISSEQ data is checked in, and the unit fixtures are far too small to
exercise the aggregation stage's memory behaviour (a handful of cells and two
feature columns, against ~1731 features and hundreds of thousands of cells in
production). This writes a stand-in with the same shape and the same null/NaN
density, streamed out in row groups so the generator itself does not need to
hold the full table in memory.
"""

from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fisseq_data_pipeline.utils.constants import CONTROL_COLUMN_NAME

LABEL_COL = "meta_aa_changes"
#: Bytes of float64 feature data per written row group.
_ROW_GROUP_BYTES = 3e8


def generate(
    path: str,
    n_cells: int,
    n_features: int,
    n_variants: int,
    control_frac: float = 0.05,
    seed: int = 0,
) -> int:
    """
    Write a synthetic cell-level parquet and return its row count.

    Variant group sizes are drawn from a lognormal, which reproduces the
    heavily skewed per-variant cell counts of a real batch (a handful of large
    variants, a long tail of small ones) -- group size drives the per-label
    list lengths the aggregators build, so a uniform split would understate
    the peak.

    Parameters
    ----------
    path : str
        Destination parquet path.
    n_cells : int
        Approximate total cell count (rounding across groups moves it
        slightly).
    n_features : int
        Number of feature columns. Production is ~1731.
    n_variants : int
        Number of distinct non-control variant labels.
    control_frac : float
        Fraction of cells labelled control. Defaults to ``0.05``.
    seed : int
        Seed for the value draw. Defaults to ``0``.

    Returns
    -------
    int
        Number of rows written.
    """
    rng = np.random.default_rng(seed)
    weights = rng.lognormal(0.0, 0.9, n_variants)
    weights /= weights.sum()
    n_control = int(n_cells * control_frac)
    counts = np.maximum(1, (weights * (n_cells - n_control)).astype(np.int64))

    labels = np.repeat(np.array([f"V{i}" for i in range(n_variants)]), counts)
    labels = np.concatenate([labels, np.array(["SYN"] * n_control)])
    # Shuffle so rows are not pre-grouped by label, as real cell tables are
    # not: a pre-sorted frame lets Polars group far more cheaply than it can
    # in production, which would flatter the benchmark.
    labels = labels[rng.permutation(labels.shape[0])]
    is_control = labels == "SYN"
    n_rows = labels.shape[0]

    schema = pa.schema(
        [(LABEL_COL, pa.string()), (CONTROL_COLUMN_NAME, pa.bool_())]
        + [(f"FEAT_{i:05d}", pa.float64()) for i in range(n_features)]
    )
    batch_rows = max(1, int(_ROW_GROUP_BYTES / (n_features * 8)))
    with pq.ParquetWriter(path, schema, compression="snappy") as writer:
        for start in range(0, n_rows, batch_rows):
            stop = min(n_rows, start + batch_rows)
            size = stop - start
            arrays = [pa.array(labels[start:stop]), pa.array(is_control[start:stop])]
            for i in range(n_features):
                values = rng.standard_normal(size)
                if i % 17 == 0:  # a sparse scatter of NaN, as real features have
                    values[rng.random(size) < 0.01] = np.nan
                arrays.append(pa.array(values))
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
    return n_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--n-cells", type=int, default=150_000)
    parser.add_argument("--n-features", type=int, default=2048)
    parser.add_argument("--n-variants", type=int, default=1000)
    parser.add_argument("--control-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_rows = generate(
        args.path,
        args.n_cells,
        args.n_features,
        args.n_variants,
        args.control_frac,
        args.seed,
    )
    print(
        f"wrote {args.path}: {n_rows} rows x {args.n_features} features, "
        f"{args.n_variants} variants"
    )


if __name__ == "__main__":
    main()
