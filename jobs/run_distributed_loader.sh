#!/bin/bash
# Validate distributed reading with eight ranks.
#
#   sbatch jobs/run_distributed_loader.sh configs/distributed/healthy.yaml
#   sbatch jobs/run_distributed_loader.sh configs/distributed/duplicate_samples.yaml
#
# Eight tasks with 7 CPUs each mirrors one LUMI-G node's shape: 8 GCDs, 7 cores per GCD.
# No GPU is requested, because this benchmark validates the *data path* and never
# touches one — spending GPU hours to read files would be waste. To reproduce the exact
# NUMA and binding of a GPU node, switch the partition to standard-g and add
# --gpus-per-node=8; the ranks and CPU share stay the same.
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
