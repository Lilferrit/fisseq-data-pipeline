#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NOTE: for SGE jobs, replace $SCRIPT_DIR with a hard-coded absolute path,
# since SGE may not run from the expected working directory.
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/env/bin/activate"

nextflow run "${SCRIPT_DIR}/workflow.nf" \
    -profile sge \
    --input_dir "${SCRIPT_DIR}" \
    "$@"
