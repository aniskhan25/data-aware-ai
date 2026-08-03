#!/bin/bash
# Generate the tutorial dataset on project storage.
#
#   sbatch jobs/prepare_dataset.sh configs/datasets/balanced.yaml
#
# Generation is CPU work and needs no GPU. It is a one-off preparation step: the
# dataset is deterministic, so it never has to be regenerated to be reproduced.

#SBATCH --job-name=daai-prepare
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Fail before anything else, so a broken setup cannot report success.
set -euo pipefail

# Slurm copies the batch script into a spool directory, so $0 does not point into
# the repository. SLURM_SUBMIT_DIR is where sbatch was invoked, i.e. the repo root.
COMMON="${SLURM_SUBMIT_DIR:-$(dirname "$0")}/jobs/common.sh"
[[ -f "$COMMON" ]] || COMMON="$(dirname "$0")/common.sh"
if [[ ! -f "$COMMON" ]]; then
    echo "ERROR cannot find jobs/common.sh; submit from the repository root" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "$COMMON"

PROFILE_CONFIG="${1:-configs/datasets/balanced.yaml}"

require_vars TUTORIAL_ROOT
if [[ ! -f "$PROFILE_CONFIG" ]]; then
    echo "ERROR dataset profile not found: $PROFILE_CONFIG" >&2
    exit 2
fi

report_allocation

DATASET_DIR="$TUTORIAL_ROOT/source"
mkdir -p "$DATASET_DIR"

WORKERS="${SLURM_CPUS_PER_TASK:-4}"

echo "PROFILE_CONFIG=$PROFILE_CONFIG"
echo "DATASET_DIR=$DATASET_DIR"
echo "GENERATION_WORKERS=$WORKERS"

# The manifest is left at its default location inside the dataset directory, which
# is where every configs/baseline/*.yaml expects to find it. Keeping it beside the
# data also means the dataset and its manifest cannot drift apart.
run_python scripts/generate_dataset.py \
    --profile-config "$PROFILE_CONFIG" \
    --output "$DATASET_DIR" \
    --workers "$WORKERS" \
    --overwrite

echo "DONE dataset ready at $DATASET_DIR"
echo "Next: sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml"
