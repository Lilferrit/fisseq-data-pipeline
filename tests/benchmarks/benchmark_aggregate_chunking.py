"""Sweep ``feature_chunk_size`` and report wall time and peak RSS.

Each case runs in its own subprocess, guarded by an RSS watchdog. The watchdog
matters: the failure this harness exists to measure is an OOM kill, and letting
the real kernel OOM killer decide makes the result depend on whatever else is
running on the box (and can take the parent down instead). Exceeding
``--budget-mb`` is reported as ``OOM`` and the case moves on.

Run directly -- ``pyproject.toml``'s ``testpaths`` excludes this directory from
the default pytest run. See ``tests/benchmarks/README.md``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
#: How often the watchdog samples its own RSS. Fine enough to catch the spike
#: inside a single Polars collect, cheap enough to not perturb the timing.
_POLL_SECONDS = 0.05
_OOM_EXIT_CODE = 42


def _rss_mb() -> float:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * _PAGE_SIZE / 1024 / 1024


def _start_rss_watchdog(budget_mb: float) -> None:
    """Exit with ``_OOM_EXIT_CODE`` as soon as RSS passes ``budget_mb``."""

    def poll() -> None:
        while True:
            rss = _rss_mb()
            if rss > budget_mb:
                print(f"PEAK\t{rss:.0f}", flush=True)
                os._exit(_OOM_EXIT_CODE)
            time.sleep(_POLL_SECONDS)

    threading.Thread(target=poll, daemon=True).start()


def _run_one_case() -> None:
    """Child-process entry point: aggregate once and report time and peak RSS."""
    import resource

    import polars as pl

    from fisseq_data_pipeline.aggregate import aggregate

    path, aggregator, chunk_size, budget_mb, label_col = (
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
        float(sys.argv[5]),
        sys.argv[6],
    )
    _start_rss_watchdog(budget_mb)

    lf = pl.scan_parquet(path)
    started = time.perf_counter()
    # sink_parquet to /dev/null, not collect(): this is the call that was
    # dying in production, and discarding the bytes keeps disk out of the
    # measurement.
    aggregate(lf, label_col, aggregator, feature_chunk_size=chunk_size).sink_parquet(
        "/dev/null"
    )
    elapsed = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"OK\t{elapsed:.2f}\t{peak:.0f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="cell-level parquet from generate_benchmark_data")
    parser.add_argument(
        "--aggregators", nargs="+", default=["median", "AUROC", "KS", "QQ"]
    )
    parser.add_argument(
        "--chunk-sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--budget-mb",
        type=float,
        default=5000.0,
        help="memory the cluster grants one task; exceeding it reports OOM",
    )
    parser.add_argument("--label-col", default="meta_aa_changes")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    print("aggregator\tchunk\tsecs\tpeak_rss_mb\tstatus")
    for aggregator in args.aggregators:
        for chunk_size in args.chunk_sizes:
            proc = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--child",
                    args.path,
                    aggregator,
                    str(chunk_size),
                    str(args.budget_mb),
                    args.label_col,
                ],
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            line = next(
                (
                    ln
                    for ln in proc.stdout.splitlines()
                    if ln.startswith(("OK\t", "PEAK\t"))
                ),
                "",
            )
            if line.startswith("OK\t"):
                _, secs, peak = line.split("\t")
                status = f"{aggregator}\t{chunk_size}\t{secs}\t{peak}\tok"
            elif proc.returncode == _OOM_EXIT_CODE:
                peak = line.split("\t")[1] if line else ""
                status = f"{aggregator}\t{chunk_size}\t\t{peak}\tOOM"
            else:
                status = f"{aggregator}\t{chunk_size}\t\t\tFAILED(rc={proc.returncode})"
                print(proc.stderr.strip()[-2000:], file=sys.stderr)
            print(status, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _run_one_case()
    else:
        main()
