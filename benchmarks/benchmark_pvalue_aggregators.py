"""Benchmark harness for the reference-based p-value aggregators in
:mod:`fisseq_data_pipeline.aggregate` (KS, AUROC, and their searchsorted-
restructured variants).

This is a standalone script, not a pytest file: it lives in a top-level
``benchmarks/`` directory (a sibling of ``tests/``, not nested inside it) so
``uv run pytest tests`` can never collect it regardless of ``testpaths``
configuration. There is no ``pytest-benchmark``/``psutil``/``memory-profiler``
dependency in this project, so it measures wall time via
``time.perf_counter``, peak traced Python allocations via ``tracemalloc``,
and RSS delta via ``resource.getrusage`` (Linux) directly.

Usage::

    uv run python benchmarks/benchmark_pvalue_aggregators.py \\
        --n-labels 300 --n-features 300 --n-cells-per-group 30 --n-ref 30 \\
        --aggregators KS AUROC

Reports one line per aggregator in the same
"{name} took {s:.2f}s / {kb:.1f}KB peak traced Python allocations
({mb:.1f}MB RSS delta)" format cited in the KSNegLogPValueAggregator /
AUROCNegLogPValueAggregator docstrings in ``aggregate.py``, so before/after
comparisons stay apples-to-apples with that prose.
"""

from __future__ import annotations

import argparse
import resource
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import polars as pl

from fisseq_data_pipeline.aggregate import _AGGREGATORS


def make_synthetic_lf(
    n_labels: int,
    n_features: int,
    n_cells_per_group: int,
    n_ref: int,
    seed: int = 0,
) -> pl.LazyFrame:
    """
    Synthetic cell-level frame: one ``"WT"`` control group of ``n_ref``
    rows (``meta_is_control=True``) plus ``n_labels`` variant groups of
    ``n_cells_per_group`` rows each, with ``n_features`` iid
    ``standard_normal`` feature columns. Mirrors the shape
    :class:`fisseq_data_pipeline.aggregate.ReferenceBasedAggregator`
    subclasses expect: a ``meta_aa_changes`` label column and a
    ``meta_is_control`` boolean column alongside the feature columns.
    """
    rng = np.random.default_rng(seed)
    n_variant_rows = n_labels * n_cells_per_group
    n_rows = n_ref + n_variant_rows

    labels = np.empty(n_rows, dtype=object)
    labels[:n_ref] = "WT"
    variant_labels = np.repeat([f"V{i}" for i in range(n_labels)], n_cells_per_group)
    labels[n_ref:] = variant_labels

    is_control = np.zeros(n_rows, dtype=bool)
    is_control[:n_ref] = True

    data: dict[str, object] = {
        "meta_aa_changes": labels.tolist(),
        "meta_is_control": is_control.tolist(),
    }
    for i in range(n_features):
        data[f"f{i}"] = rng.standard_normal(n_rows)

    return pl.DataFrame(data).lazy()


@dataclass
class BenchmarkResult:
    name: str
    seconds: float
    traced_peak_bytes: int
    rss_delta_kb: int

    def format(self) -> str:
        kb = self.traced_peak_bytes / 1024
        mb = self.rss_delta_kb / 1024
        return (
            f"{self.name} took {self.seconds:.2f}s / {kb:.1f}KB peak traced "
            f"Python allocations ({mb:.1f}MB RSS delta)"
        )


def run_one(key: str, lf: pl.LazyFrame, label_col: str = "meta_aa_changes") -> BenchmarkResult:
    """Time and memory-profile one aggregator (by ``_AGGREGATORS`` key) end
    to end, from ``.aggregate(lf)`` through ``.collect()``."""
    agg_cls = _AGGREGATORS[key]
    agg = agg_cls(label_col=label_col)

    tracemalloc.start()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()

    agg.aggregate(lf).collect()

    elapsed = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        name=key,
        seconds=elapsed,
        traced_peak_bytes=peak,
        rss_delta_kb=rss_after - rss_before,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-labels", type=int, default=300)
    parser.add_argument("--n-features", type=int, default=300)
    parser.add_argument("--n-cells-per-group", type=int, default=30)
    parser.add_argument(
        "--n-ref",
        type=int,
        default=None,
        help="Control-group row count. Defaults to --n-cells-per-group "
        "(balanced regime). Set e.g. 10x --n-cells-per-group for the "
        "high-n_ref regime.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--aggregators",
        nargs="+",
        default=None,
        help="Subset of _AGGREGATORS keys to run (default: all registered).",
    )
    args = parser.parse_args()

    n_ref = args.n_ref if args.n_ref is not None else args.n_cells_per_group
    keys = args.aggregators if args.aggregators is not None else list(_AGGREGATORS)

    unknown = [k for k in keys if k not in _AGGREGATORS]
    if unknown:
        raise SystemExit(f"Unknown aggregator key(s): {unknown}. Choose from: {sorted(_AGGREGATORS)}")

    print(
        f"scale: n_labels={args.n_labels} n_features={args.n_features} "
        f"n_cells_per_group={args.n_cells_per_group} n_ref={n_ref}"
    )
    lf = make_synthetic_lf(
        n_labels=args.n_labels,
        n_features=args.n_features,
        n_cells_per_group=args.n_cells_per_group,
        n_ref=n_ref,
        seed=args.seed,
    ).collect().lazy()  # materialize synthetic data once, outside the timed region

    for key in keys:
        result = run_one(key, lf)
        print(result.format())


if __name__ == "__main__":
    main()
