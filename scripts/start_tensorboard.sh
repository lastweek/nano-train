#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/start_tensorboard.sh [--logdir PATH] [--port PORT] [--host HOST] \\
    [--reload_interval SECONDS] [--version]

Defaults:
  --logdir outputs
  --port  6006
  --host  localhost
  --reload_interval 1

Notes:
  - If your system `tensorboard` is too old (e.g. imports TensorFlow and crashes), this script
    bootstraps a local venv at `.venv_tensorboard/` and runs a modern TensorBoard from there.

Examples:
  ./scripts/start_tensorboard.sh
  ./scripts/start_tensorboard.sh --logdir outputs/nano_train_mvp
  ./scripts/start_tensorboard.sh --port 7007
  ./scripts/start_tensorboard.sh --reload_interval 1
  ./scripts/start_tensorboard.sh --version
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOGDIR="outputs"
PORT="6006"
HOST="localhost"
RELOAD_INTERVAL="1"
PRINT_VERSION="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --version)
            PRINT_VERSION="1"
            shift 1
            ;;
        --logdir)
            LOGDIR="${2:-}"
            shift 2
            ;;
        --port)
            PORT="${2:-}"
            shift 2
            ;;
        --host)
            HOST="${2:-}"
            shift 2
            ;;
        --reload_interval)
            RELOAD_INTERVAL="${2:-}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${LOGDIR}" ]]; then
    echo "Error: --logdir must be non-empty" >&2
    exit 2
fi

cd "${REPO_ROOT}"

if [[ "${PRINT_VERSION}" != "1" ]]; then
    if [[ ! -d "${LOGDIR}" ]]; then
        echo "Error: logdir not found: ${LOGDIR}" >&2
        echo "Run training first to generate event files (default: outputs/<run_name>/)." >&2
        exit 1
    fi
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv_tensorboard}"

check_tensorboard_python() {
    "${1}" - <<'PY' >/dev/null 2>&1
from __future__ import annotations

from tensorboard import version as tb_version

version = getattr(tb_version, "VERSION", "")
major = int(str(version).split(".")[0])
if major < 2:
    raise SystemExit(1)

import pkg_resources  # noqa: F401  # tensorboard.default imports this
PY
}

bootstrap_tensorboard_venv() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        echo "Creating TensorBoard venv: ${VENV_DIR}"
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi

    local venv_python="${VENV_DIR}/bin/python"
    echo "Installing TensorBoard (no TensorFlow) into: ${VENV_DIR}"
    "${venv_python}" -m pip install --upgrade pip >/dev/null
    "${venv_python}" -m pip install --upgrade "tensorboard>=2.15.0" "setuptools<82"
}

TB_PYTHON="${PYTHON_BIN}"
if ! check_tensorboard_python "${TB_PYTHON}"; then
    if [[ -x "${VENV_DIR}/bin/python" ]] && check_tensorboard_python "${VENV_DIR}/bin/python"; then
        TB_PYTHON="${VENV_DIR}/bin/python"
    else
        bootstrap_tensorboard_venv
        TB_PYTHON="${VENV_DIR}/bin/python"
        if ! check_tensorboard_python "${TB_PYTHON}"; then
            echo "Error: TensorBoard bootstrap failed." >&2
            echo "Try: ${TB_PYTHON} -m pip install -U 'tensorboard>=2.15.0' 'setuptools<82'" >&2
            exit 1
        fi
    fi
fi

if [[ "${PRINT_VERSION}" == "1" ]]; then
    exec env PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:pkg_resources is deprecated:UserWarning}" \
        "${TB_PYTHON}" -m tensorboard.main --version_tb
fi

echo "Starting TensorBoard..."
echo "  logdir: ${LOGDIR}"
echo "  url:    http://${HOST}:${PORT}"

exec env PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:pkg_resources is deprecated:UserWarning}" \
    "${TB_PYTHON}" -m tensorboard.main \
    --logdir="${LOGDIR}" \
    --port="${PORT}" \
    --host="${HOST}" \
    --reload_interval="${RELOAD_INTERVAL}"
