# Shared job setup. Sourced by every script in jobs/, never run directly.
#
# Responsibilities: strict shell mode, site configuration, required-variable
# checks, allocation reporting, and a single place that decides how Python is
# invoked (directly or inside a container).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Site configuration. Absent on a laptop, which is fine.
if [[ -f "$REPO_ROOT/env.sh" ]]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/env.sh"
fi

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/outputs"

require_vars() {
    local missing=()
    local name
    for name in "$@"; do
        if [[ -z "${!name:-}" ]]; then
            missing+=("$name")
        fi
    done
    if (( ${#missing[@]} > 0 )); then
        echo "ERROR required variable(s) not set: ${missing[*]}" >&2
        echo "Copy env.example.sh to env.sh and edit it." >&2
        exit 2
    fi
}

report_allocation() {
    echo "=== allocation ==="
    echo "HOSTNAME=$(hostname)"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
    echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-none}"
    echo "SLURM_NNODES=${SLURM_NNODES:-1}"
    echo "SLURM_NTASKS=${SLURM_NTASKS:-1}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
    echo "SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-none}"
    echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset}"
    echo "TUTORIAL_ROOT=${TUTORIAL_ROOT:-unset}"
    echo "TUTORIAL_CONTAINER=${TUTORIAL_CONTAINER:-none}"
    echo "=================="
}

# One thread per process. Without this, every DataLoader worker would try to use
# the whole allocation, and the resulting oversubscription is easily mistaken for
# a storage problem.
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Interpreter used outside a container. Override to test with a virtual
# environment: TUTORIAL_PYTHON=.venv/bin/python bash jobs/run_loader.sh ...
PYTHON="${TUTORIAL_PYTHON:-python3}"

python_is_recent() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null
}

# LUMI's system python3 is 3.6, older than this project supports. When no container
# is configured, fall back to the cray-python module, which provides 3.11.
if [[ -z "${TUTORIAL_CONTAINER:-}" ]] && ! python_is_recent "$PYTHON"; then
    if ! command -v module >/dev/null 2>&1 && [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        # shellcheck source=/dev/null
        source /usr/share/lmod/lmod/init/bash
    fi
    if command -v module >/dev/null 2>&1; then
        module load cray-python 2>/dev/null || true
        PYTHON=python3
    fi
fi

if [[ -z "${TUTORIAL_CONTAINER:-}" ]] && ! python_is_recent "$PYTHON"; then
    echo "ERROR $PYTHON is older than Python 3.9, which this project requires." >&2
    echo "On LUMI: module load cray-python, or set TUTORIAL_CONTAINER in env.sh." >&2
    exit 2
fi

# Run Python either directly or inside the configured container.
run_python() {
    if [[ -n "${TUTORIAL_CONTAINER:-}" ]]; then
        # The repository is always bound, because the scripts live there. Anything
        # in TUTORIAL_CONTAINER_BINDS is added to it rather than replacing it.
        local binds="$REPO_ROOT${TUTORIAL_CONTAINER_BINDS:+,$TUTORIAL_CONTAINER_BINDS}"
        # $WITH_CONDA is provided by LUMI's PyTorch containers to activate the
        # bundled environment. Harmless when the container does not define it.
        singularity exec -B "$binds" "$TUTORIAL_CONTAINER" bash -c \
            "\${WITH_CONDA:-true}; cd '$REPO_ROOT' && python3 $(printf '%q ' "$@")"
    else
        "$PYTHON" "$@"
    fi
}
