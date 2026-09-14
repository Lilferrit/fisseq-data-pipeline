# Dockerfile -- the single image every Nextflow process runs in.
#
# CPU-only: this pipeline is polars + XGBoost + pycytominer, with no CUDA
# dependency anywhere, so a plain ubuntu base is enough (the sibling
# fisseq-embeddings-pipeline needs nvidia/cuda for its Cell-DINO stage; nothing
# here does).
FROM ubuntu:22.04

# ca-certificates/git: needed by the uv project install below. No compiler
# toolchain is required -- every runtime dependency resolves to a prebuilt
# wheel on both amd64 and arm64.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# uv as a standalone static binary (Astral's own published image), not via
# `pip install uv` -- this needs no system Python or pip pre-installed at all,
# since uv provisions its own interpreter. ubuntu 22.04 (jammy) only ships
# Python 3.10 by default, which does not satisfy pyproject.toml's
# `requires-python` regardless.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /opt/fisseq-data-pipeline

# .python-version (which pins 3.13 exactly, not just pyproject.toml's
# `>=3.13,<3.14` range) MUST be copied in before the first `uv sync`. Without
# it uv has nothing here to prefer 3.13 over any newer interpreter it can
# fetch, and Hydra 1.3.x's `get_args_parser()` crashes on Python 3.14's
# stricter argparse (`TypeError: argument of type 'LazyCompletionHelp' is not a
# container or iterable`) -- which breaks every single stage's
# `python -m fisseq_data_pipeline.<module>` invocation outright, not just
# `--help`. pyproject.toml's `<3.14` upper bound is the belt to this braces.
#
# Dependencies-only layer, cached independently of source changes. `uv sync`
# also builds and installs the local project by default, which fails here
# because src/ and README.md are not present yet (hatchling's metadata
# validation needs both) -- hence --no-install-project. --no-dev leaves the
# mkdocs/pytest/ruff/pre-commit tooling out of the runtime image; every
# process runs `python -m fisseq_data_pipeline.<module>` directly, never
# pytest or ruff.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev

# Now the project itself -- the deps layer above stays cached across
# source-only changes.
COPY README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev

# `uv sync` installs into a project-local .venv/, not any system Python -- put
# it on PATH so the bare `python -m fisseq_data_pipeline.<module>` that every
# modules/local/*.nf script block invokes actually resolves there. Without
# this, `python` would not resolve at all (no system Python is installed
# above), and every process would fail with `python: command not found`.
ENV PATH="/opt/fisseq-data-pipeline/.venv/bin:${PATH}"

ENTRYPOINT []
