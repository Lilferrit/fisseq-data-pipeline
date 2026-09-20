# Aggregation benchmarks

Not collected by `pytest` — `pyproject.toml` sets
`testpaths = ["tests/unit", "tests/integration"]`, so this directory is
excluded from the default run. Run the scripts directly.

These exist because the aggregation stage's cost is a *memory* problem, not a
throughput one, and peak RSS is invisible to the unit tests. Before this
harness, `aggregate.py`'s docstrings cited benchmark numbers from a file
(`benchmark_pvalue_aggregators.py`) that was never committed.

## Why

On the 111925 cluster run, every `KS`, `AUROC`, `KSnegLogP` and
`AUROCnegLogP` task was OOM-killed (exit 137) inside `sink_parquet` — 714 of
947 aggregation tasks produced no output, silently, because every process
carries `errorStrategy 'ignore'`. `BaseAggregator.aggregate` now evaluates
`feature_chunk_size` features per query instead of all ~1731 at once.

## Usage

```bash
# 1. Generate a synthetic cell-level parquet at production shape.
#    (500k cells x 1731 features is ~8.5 GB on disk; start smaller.)
uv run python tests/benchmarks/generate_benchmark_data.py \
    /tmp/bench.parquet --n-cells 150000 --n-features 2048 --n-variants 1000

# 2. Sweep chunk sizes, one subprocess per case, with an RSS watchdog that
#    emulates the cluster's OOM kill instead of waiting for the real one.
uv run python tests/benchmarks/benchmark_aggregate_chunking.py \
    /tmp/bench.parquet --aggregators median AUROC KS QQ \
    --chunk-sizes 8 16 32 64 128 --budget-mb 5000
```

`--budget-mb` should be set to whatever memory the cluster actually grants
these tasks (they declare `label 'process_medium'`, which `nextflow.config`
deliberately leaves unsized).

## Reference numbers

150k cells x 2048 features x 1000 variants, 7500 control cells, 5 GB budget.
`chunk = 2048` is a single query, i.e. the pre-chunking behaviour — it OOMs
for every aggregator, which is the production failure reproduced.

| chunk | mean | median | std | MAD | AUROC | KS | QQ |
|---|---|---|---|---|---|---|---|
| 4 | | | | | | 561 s / 2893 | 964 s / 1913 |
| 8 | | | | | | 577 s / 3902 | 919 s / 1718 |
| 16 | | | | | | 649 s / 4203 | 939 s / 2043 |
| 32 | 1.3 s / 706 | 1.1 s / 673 | 1.1 s / 655 | 1.7 s / 666 | 160 s / 3041 | 602 s / 4698 | 856 s / 2919 |
| 64 | 0.9 s / 920 | 1.0 s / 914 | 0.9 s / 908 | 1.5 s / 914 | 162 s / 3094 | 629 s / 4797 | 973 s / 3034 |
| 128 | 0.8 s / 1349 | 1.0 s / 1295 | 0.8 s / 1358 | 1.5 s / 1273 | 162 s / 3102 | OOM | 1033 s / 3773 |
| 256 | 0.9 s / 2007 | 1.0 s / 1935 | 0.9 s / 1940 | 1.8 s / 1856 | 169 s / 3621 | OOM | OOM |
| 512 | 1.1 s / 3020 | 1.2 s / 2995 | 1.1 s / 3144 | 2.4 s / 2507 | 167 s / 4247 | OOM | OOM |
| 1024 | 2.4 s / 4020 | 1.8 s / 4132 | 1.7 s / 4133 | 3.7 s / 4350 | OOM | OOM | OOM |
| 2048 | OOM | OOM | OOM | OOM | OOM | OOM | OOM |

Seconds / peak RSS in MB. Two things this table is meant to settle:

1. **Runtime is flat in chunk size for the expensive aggregators** (AUROC
   moved 160 s -> 167 s across a 16x range). Chunk size is a memory dial, not
   a speed/memory trade-off — halving it when a task is killed costs
   essentially nothing.
2. **`KS` pins the default.** It OOMs at 128 where `AUROC` still has headroom.
   `params.aggregate_feature_chunk_size` defaults to 64, the widest chunk at
   which every aggregator completed here.

Peak memory scales with `chunk_size x n_variant_labels` (times the control
pool, for the reference-based aggregators), so these numbers do not transfer
to a differently-shaped batch. Production batches are *larger* on both axes
than this one (1622-1947 labels, WT pools of 2051-12535 cells), so re-run the
sweep against real shapes before trusting a default.
