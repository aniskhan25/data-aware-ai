#!/bin/bash
# Measure one storage placement.
#
#   sbatch jobs/run_storage_comparison.sh configs/staging/scratch.yaml
#   sbatch jobs/run_storage_comparison.sh configs/staging/tmp.yaml
#
# --mem is doubly important for the /tmp rung: node-local /tmp lives in memory and is
# charged against this allocation, so it is both the space the dataset is copied into
# and the number the safety check is measured against. Requesting more memory than the
# workload needs would flatter staging; requesting too little makes the job refuse to
# stage, which is the safe direction.
#
# The run summary is written to output.directory on shared storage, so it survives the
# node-local cleanup that happens when the job ends.

#SBATCH --job-name=daai-storage
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Fail before anything else, so a broken setup cannot report success.
set -euo pipefail

COMMON="${SLURM_SUBMIT_DIR:-$(dirname "$0")}/jobs/common.sh"
[[ -f "$COMMON" ]] || COMMON="$(dirname "$0")/common.sh"
if [[ ! -f "$COMMON" ]]; then
    echo "ERROR cannot find jobs/common.sh; submit from the repository root" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "$COMMON"

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    echo "usage: sbatch jobs/run_storage_comparison.sh CONFIG [section.option=value ...]" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR configuration not found: $CONFIG" >&2
    exit 2
fi
shift

require_vars TUTORIAL_ROOT
report_allocation

echo "=== node-local space ==="
echo "SLURM_TMPDIR=${SLURM_TMPDIR:-unset}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset}"
df -h /tmp 2>/dev/null | tail -1 || true
echo "======================="

# Create the node-local directory on the host and bind it into the container.
# Relying on the container's own TMPDIR is not enough: inside a container that is
# usually plain /tmp, which may not be the host's node-local filesystem at all.
NODE_LOCAL="/tmp/daai-${SLURM_JOB_ID:-$$}"
mkdir -p "$NODE_LOCAL"
export TUTORIAL_CONTAINER_BINDS="${TUTORIAL_CONTAINER_BINDS:+$TUTORIAL_CONTAINER_BINDS,}$NODE_LOCAL"
echo "NODE_LOCAL=$NODE_LOCAL"

OVERRIDES=(--set "storage.tmp_dir=$NODE_LOCAL")
for assignment in "$@"; do
    if [[ "$assignment" != *=* ]]; then
        echo "ERROR override must look like section.option=value, got '$assignment'" >&2
        exit 2
    fi
    OVERRIDES+=(--set "$assignment")
done

# Remove any node-local leftovers if the job is killed mid-run. The Python staging
# context manager already cleans up on normal exit and on exceptions; this covers
# SIGTERM from Slurm, which Python never sees as an exception.
cleanup() {
    if [[ -e "$NODE_LOCAL" ]]; then
        echo "TRAP removing leftover node-local data at $NODE_LOCAL"
        rm -rf "$NODE_LOCAL" || true
    fi
}
trap cleanup EXIT TERM INT

echo "CONFIG=$CONFIG"
run_python scripts/benchmark_loader.py --config "$CONFIG" \
    ${OVERRIDES[@]+"${OVERRIDES[@]}"}

echo "DONE ${SLURM_JOB_ID:-local}"
