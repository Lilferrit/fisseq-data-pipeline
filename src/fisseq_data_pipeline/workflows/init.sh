#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"

USE_UV=0
for arg in "$@"; do
    if [[ "${arg}" == "--uv" ]]; then
        USE_UV=1
    fi
done

if [[ ${USE_UV} -eq 1 ]]; then
    uv venv "${ENV_DIR}"
    uv pip install --upgrade --python "${ENV_DIR}/bin/python" -e "${REPO_ROOT}"
else
    python3 -m venv "${ENV_DIR}"
    "${ENV_DIR}/bin/pip" install --upgrade -e "${REPO_ROOT}"
fi
