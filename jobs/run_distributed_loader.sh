#!/bin/bash
# Validate distributed reading with eight ranks.
#
#   sbatch jobs/run_distributed_loader.sh configs/distributed/healthy.yaml
#   sbatch jobs/run_distributed_loader.sh configs/distributed/duplicate_samples.yaml
#
# Eight tasks with 7 CPUs each reproduces the rank count and nominal per-rank CPU share
# of a full LUMI-G job. It does NOT reproduce LUMI-G's NUMA layout, CPU-GPU binding, or
# memory placement: this runs on a CPU partition and never touches an accelerator.
#
# That is the right trade for validating a data path, and spending GPU hours to read
# files would be waste. When you need to validate placement on the real accelerator node,
# use standard-g with eight GCDs and explicit CPU/GPU binding.
#
# Exit code 4 means a correctness problem was found. For the deliberately broken
# challenges that is the expected outcome, and their configs set
# validate_unique_samples: false so the job reports the damage rather than failing.

#SBATCH --job-name=daai-distributed
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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
    echo "usage: sbatch jobs/run_distributed_loader.sh CONFIG [section.option=value ...]" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR configuration not found: $CONFIG" >&2
    exit 2
fi
shift

require_vars TUTORIAL_ROOT
report_allocation

OVERRIDES=()
for assignment in "$@"; do
    if [[ "$assignment" != *=* ]]; then
        echo "ERROR override must look like section.option=value, got '$assignment'" >&2
        exit 2
    fi
    OVERRIDES+=(--set "$assignment")
done

# Every rank must agree on the rendezvous address and port. The address comes from the
# allocation; the port is derived from the job ID so two concurrent jobs on one node
# cannot collide on it.
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 10000 ))
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NTASKS=${SLURM_NTASKS:-unset} CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

# One thread per rank's main process, as in the single-process benchmark: eight ranks
# each trying to use the whole node would be oversubscription mistaken for slow storage.
export OMP_NUM_THREADS=1

# --kill-on-bad-exit stops the job if any rank dies. Without it, a rank that fails
# during measurement leaves the others blocked in the correctness gather until the
# wall-clock limit, which wastes the whole allocation to report nothing.
STATUS=0
srun --kill-on-bad-exit=1 jobs/_rank_task.sh \
    --config "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"} || STATUS=$?

echo "SRUN_STATUS=$STATUS"
if [[ "$STATUS" == "4" ]]; then
    echo "NOTE exit 4 means a partitioning problem was reported. For the deliberately"
    echo "     broken challenges that is the expected result."
fi
echo "DONE ${SLURM_JOB_ID:-local}"
exit "$STATUS"
