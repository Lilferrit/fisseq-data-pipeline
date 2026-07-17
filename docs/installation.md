# Installation

## Requirements

- **Python 3.13** (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for Python dependency and environment management
- **[Nextflow](https://www.nextflow.io/) ≥ 23.10** (only needed to run the full
  pipeline via `main.nf`, not for standalone Python CLI usage)
- No required environment variables

## Install

```bash
# Clone the repo
git clone https://github.com/Lilferrit/fisseq-data-pipeline.git
cd fisseq-data-pipeline

# Install all runtime + dev dependencies into .venv
uv sync --group dev

# Install pre-commit hooks (one-time)
uv run pre-commit install
```

The package installs in editable mode, so `fisseq-*` CLI commands are immediately
available via `uv run fisseq-qc-filter`, `uv run fisseq-normalize`, etc. See the
[CLI Reference](cli/qcfilter.md) for every available command.

### Install with pip (no clone)

If you just need the `fisseq-*` CLI tools available on `PATH` — e.g. on a
compute node, in a container, or for a quick one-off install — `pip install`
directly from GitHub instead of cloning and using `uv sync`:

```bash
pip install git+https://github.com/Lilferrit/fisseq-data-pipeline.git
```

Pin a branch, tag, or commit with `@`:

```bash
pip install git+https://github.com/Lilferrit/fisseq-data-pipeline.git@main
```

To run the pipeline directly from GitHub without cloning, see the
[Nextflow Workflow](nextflow.md) page's Quickstart section — Nextflow can pull and
cache the repository itself.

## Cluster / HPC

The repo ships a `nextflow.config` at the root with default `params` values and
commented-out profile stubs for `venv`, `conda`, `singularity`, and `sge` executors.
To run on a cluster:

1. Write your own config (or copy and adapt `nextflow.config`).
2. Uncomment and fill in a profile block — pick one `beforeScript` option to make
   the `fisseq-*` CLI tools available on each compute node:

   ```groovy
   // Option A: activate a pre-existing venv (recommended for shared clusters)
   beforeScript = 'source /path/to/your/venv/bin/activate'

   // Option B: install from GitHub on each run (simpler, slower)
   beforeScript = 'uv pip install git+https://github.com/your-org/fisseq-data-pipeline.git@main --system'
   ```

3. Pass it at run time:

   ```bash
   nextflow run . -c your.config -profile sge --input_dir /path/to/experiment
   ```

See [Nextflow Workflow](nextflow.md) for the full parameter reference.
