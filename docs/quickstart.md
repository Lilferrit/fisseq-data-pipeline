# Quickstart

The fastest path from a fresh checkout to a first pipeline run, including
setting up a cluster config. For more depth, see
[Installation](installation.md) (full environment setup),
[Configuration](configuration.md) (every parameter, the `pipeline_dir` layout,
and global groups), and
[Walkthrough](walkthrough.md) (stage-by-stage detail on what each part of the
pipeline does).

## 1. Install

```bash
git clone https://github.com/Lilferrit/fisseq-data-pipeline.git
cd fisseq-data-pipeline
uv sync --group dev
```

Or, without cloning:

```bash
pip install git+https://github.com/Lilferrit/fisseq-data-pipeline.git
```

See [Installation](installation.md) for requirements and the pip-only path in
more detail.

## 2. Get a `nextflow.config`

The repo ships a `nextflow.config` at its root with default `params` and a set
of commented-out profile stubs (`venv`, `conda`, `singularity`, `sge`). You can
either run with it as-is, or copy it and use it as a template for your own
site.

If you've already cloned the repo, you have it at `nextflow.config`. To fetch
just the config file on its own — e.g. onto a cluster head node where you
don't want a full clone — download it directly from GitHub:

```bash
wget https://raw.githubusercontent.com/Lilferrit/fisseq-data-pipeline/main/nextflow.config -O your.config
```

### Making your own profile

Nextflow profiles control *where* and *how* each process runs (executor,
queue, resources, environment setup) — separate from the pipeline `params` in
the same file. To add one:

1. Copy `your.config` (or `nextflow.config`) somewhere you can edit it.
2. Uncomment one of the stub blocks in `profiles { }` — the `sge` stub is the
   right starting point for a Sun Grid Engine cluster:

   ```groovy
   sge {
       process {
           executor       = 'sge'
           queue          = 'all.q'           // SGE queue name
           clusterOptions = '-V'              // forward current environment to each job
           cpus           = 4
           memory         = '16 GB'
           time           = '4h'
           beforeScript   = "source /path/to/.venv/bin/activate"
       }
   }
   ```

3. Rename the block to something specific to your site if you like (e.g.
   `my_sge`) and fill in your actual queue name, resource limits, and a
   `beforeScript` that makes `fisseq_data_pipeline` importable on each compute
   node — either activating a pre-built venv (recommended) or installing it
   fresh on every run. See [Installation: Cluster / HPC](installation.md#cluster-hpc)
   for both `beforeScript` options in full.
4. Pass your config and profile name at run time with `-c your.config -profile
   my_sge` (see the run commands below).

## 3. Run commands

Run the full pipeline directly from GitHub, no local clone required:

```bash
nextflow run Lilferrit/fisseq-data-pipeline \
    -c your.config \
    -profile my_sge \
    --pipeline_dir /path/to/experiment
```

Or from a local clone:

```bash
nextflow run . -c your.config -profile my_sge --pipeline_dir /path/to/experiment
```

Resume a previously interrupted run instead of starting over:

```bash
nextflow run . -c your.config -profile my_sge --pipeline_dir /path/to/experiment -resume
```

`--pipeline_dir` must contain a `configs/` subdirectory of per-batch YAML
config files — see [Configuration](configuration.md#pipeline-directory-layout).
See [Configuration: Parameters](configuration.md#parameters) for every
`--param` the pipeline accepts.

### Running a single step directly

Every pipeline stage is also runnable on its own as a Python CLI, without
Nextflow — useful for debugging or rerunning one step against different
inputs. For example, aggregating cell-level features to one row per variant:

```bash
uv run python -m fisseq_data_pipeline.aggregate \
    output_dir=./out \
    'input_file=data/batches/*.parquet' \
    aggregator=KS
```

See the [CLI Reference](cli/aggregate.md) for every tool's config fields and
example command.

## 4. Example cluster launcher script

A minimal launcher script for submitting the pipeline as an SGE job, kept
up-to-date and re-installable on every run. Save it in the directory you want
the run's working files and `.nextflow` cache in, adjust `SCRIPT_DIR` and
`-profile`, and submit it as your SGE job script:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/path/to/your/analysis/dir"
module load nextflow
export NXF_HOME="${SCRIPT_DIR}/.nextflow"

# NOTE: for SGE jobs, replace $SCRIPT_DIR with a hard-coded absolute path,
# since SGE may not run from the expected working directory.
cd "${SCRIPT_DIR}"

if [[ ! -d "${SCRIPT_DIR}/.venv" ]]; then
    echo "No venv found at ${SCRIPT_DIR}/.venv — creating one with uv"
    uv venv "${SCRIPT_DIR}/.venv"
fi
source "${SCRIPT_DIR}/.venv/bin/activate"

RESUME_FLAG="-resume"
if [[ "${CLEAN:-false}" == "true" ]]; then
    RESUME_FLAG=""
fi

# Get the latest version in the python environment
REPO_URL="https://github.com/Lilferrit/fisseq-data-pipeline.git"
uv pip install --upgrade "git+${REPO_URL}"

# Get the latest workflow version
nextflow pull Lilferrit/fisseq-data-pipeline -r main

nextflow run Lilferrit/fisseq-data-pipeline \
    -c "${SCRIPT_DIR}/your.config" \
    -profile my_sge \
    --pipeline_dir "${SCRIPT_DIR}" \
    ${RESUME_FLAG} \
    "$@"
```

Set `CLEAN=true` before running it to force a fresh run instead of resuming
(e.g. `CLEAN=true ./run.sh`). Any extra arguments passed to the script (`"$@"`)
are forwarded straight to `nextflow run`, so you can override any pipeline
parameter without editing the script, e.g. `./run.sh --barcode_count_threshold 15`
(list-valued params like `--global_groups` need a `-c`/`-params-file` override
instead — see [Configuration: Global groups](configuration.md#global-groups)).

## Next steps

- [Walkthrough](walkthrough.md) — a complete end-to-end run, stage by stage.
- [Nextflow Workflow](nextflow.md) — every process and profile.
- [Configuration](configuration.md) — every parameter, the `pipeline_dir`
  layout, and global groups.
- [Architecture](architecture.md) — the full pipeline DAG and output layout.
- [CLI Reference](cli/qcfilter.md) — config fields and examples for every
  standalone Python tool.
