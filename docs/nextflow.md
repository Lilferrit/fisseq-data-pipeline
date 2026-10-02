# Nextflow Workflow

## Running

```bash
# Local, no container
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml -profile local

# With the published container (the default)
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml

# Resume after interruption (Nextflow task caching)
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml -resume
```

`-params-file params.yaml` is effectively mandatory: `nextflow.config` holds no
parameter defaults, so without it the workflow fails fast on the first required
parameter. See [Configuration](configuration.md).

## Layout

| Path | Role |
| ---- | ---- |
| `main.nf` | Entry point. Includes and calls one workflow. |
| `workflows/fisseq.nf` | The DAG, plus `experiments:` validation and channel wiring. |
| `modules/local/*.nf` | One process per pipeline stage, each wrapping a `python -m fisseq_data_pipeline.<module>` call. |
| `params.yaml` | Every parameter default. |
| `nextflow.config` | Executor, profile and container settings only. |
| `Dockerfile` | The single image every process runs in. |

## Processes

| Process | Cardinality | Gate |
| ------- | ----------- | ---- |
| `INPUT` | per experiment | always |
| `QC_FILTER` | per experiment | always |
| `NORMALIZE` | per experiment | always |
| `OVWT_BATCHWISE` | per experiment | `run_ovwt` |
| `AGGREGATE_FEATURE_TYPE` | per experiment × feature type (selected and passthrough) | `run_feature_selection` |
| `GENERATE_SPLIT` | per experiment × bootstrap | `run_feature_selection` |
| `AGGREGATE_HALF` | per experiment × bootstrap × half × feature type | `run_feature_selection` |
| `CORRELATE_FEATURES` | per experiment × bootstrap × feature type | `run_feature_selection` |
| `BLOCKLIST` | per experiment × feature type | `run_feature_selection` |
| `COMBINE_BLOCKLISTS` | per experiment | `run_feature_selection` |
| `FINALIZE_FEATURE_SELECT` | per experiment | `run_feature_selection` |

`BLOCKLIST`'s `groupTuple` — gathering every bootstrap replicate for one feature
type before computing a median-`r` threshold — is the pipeline's only
cross-bootstrap synchronization point. Everything else in the
split/aggregate/correlate chain is fully parallel.

Every process runs per experiment; there is no cross-experiment stage. Combining
experiments is done downstream by
[fisseqborn](https://github.com/FowlerLab/fisseqborn) from the published
per-experiment outputs (see
[Architecture](architecture.md#cross-experiment-aggregation)).

## Module conventions

Every `modules/local/*.nf` process declares:

- `errorStrategy 'ignore'` — a failed stage drops that experiment from the run
  rather than aborting everything. A "missing" output may therefore mean its
  task failed, not that it was never requested; check the run log.
- a resource `label` (`process_low` / `process_medium` / `process_high`) —
  currently inert placeholders, sized by no profile
- `container "${params.container_image}"`
- `publishDir` under `params.pipeline_dir`
- `when: task.ext.when == null || task.ext.when` — lets a config disable a
  process without editing the workflow
- a script block whose last argument is `random_seed=${params.random_seed}`
- a named `emit:`

Pipeline-wide scalars are read directly off `params.*` inside the process
script. Only genuinely per-task values (an experiment's `input_paths`, a feature
type, a bootstrap index, a publish subpath) travel through the input tuple. This
keeps `-resume` cache keys narrow: a process's key then depends only on what
actually varies per task.

## Channel idioms

A few conventions in `workflows/fisseq.nf` worth knowing before editing it.

**Filter channels, don't `if`.** Optional fan-out is expressed as an empty
channel rather than a conditional. `channel.fromList(params.feature_select_passthrough_types)`
is empty under the default `[]`, so the passthrough `AGGREGATE_FEATURE_TYPE`
simply runs zero tasks — no explicit gate needed. (The pipeline-wide run gates
`run_ovwt` / `run_feature_selection` are the exception: they wrap whole
subgraphs in an `if`.)

**A `.join()` whose right side may be empty needs `remainder: true`.** The
finalize stage joins the passthrough aggregates; with the default empty
passthrough list, a plain `.join()` emits nothing and `FINALIZE_FEATURE_SELECT`
never runs for any experiment, silently.

**`.join()` is not a broadcast operator.** For a many-to-one key relationship it
silently keeps one match per key and drops the rest, starving every downstream
stage. Use `.combine(other, by: 0)` to broadcast one per-experiment value across
a fan-out (as `AGGREGATE_HALF`'s input does), and reserve `.join()` for cases
where both sides are already collapsed to exactly one item per key (as the
finalize-stage joins are).

**Collect real path channels, not glob strings.** A `val` glob passed to a
process hashes only the glob *text*, not the resolved file set, so `-resume`
fails to invalidate when the underlying files change. The gathering processes
(`BLOCKLIST`, `COMBINE_BLOCKLISTS`, `FINALIZE_FEATURE_SELECT`) therefore take
`path(...)` inputs built with `groupTuple`, never a glob string.

**Never bind a variable named `channel`.** It is a reserved Nextflow binding —
the lowercase alias for the `Channel` class — and silently resolves to
`nextflow.Channel` itself rather than failing loudly. Use a name like `chan`
instead. (The `channel.fromList(...)` *factory call* is fine; that is not a
variable binding.)

## Linting

```bash
nextflow lint .
```

Not run in CI or pre-commit. Run it after any change to `workflows/*.nf`,
`modules/local/*.nf`, or `nextflow.config`.

Note that DSL2 forbids bare statements at script scope: a top-level helper must
be a `def someFunction() { ... }`, not a `def x = { ... }` closure binding.
