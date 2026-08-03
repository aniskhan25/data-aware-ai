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

source "$(dirname "$0")/common.sh"

PROFILE_CONFIG="${1:-configs/datasets/balanced.yaml}"

require_vars TUTORIAL_ROOT
if [[ ! -f "$PROFILE_CONFIG" ]]; then
    echo "ERROR dataset profile not found: $PROFILE_CONFIG" >&2
    exit 2
fi

report_allocation

DATASET_DIR="$TUTORIAL_ROOT/source"
MANIFEST_DIR="$TUTORIAL_ROOT/manifests"
mkdir -p "$DATASET_DIR" "$MANIFEST_DIR"

PROFILE_NAME="$(basename "$PROFILE_CONFIG" .yaml)"
WORKERS="${SLURM_CPUS_PER_TASK:-4}"

echo "PROFILE_CONFIG=$PROFILE_CONFIG"
echo "DATASET_DIR=$DATASET_DIR"
echo "GENERATION_WORKERS=$WORKERS"

run_python scripts/generate_dataset.py \
    --profile-config "$PROFILE_CONFIG" \
    --output "$DATASET_DIR" \
    --manifest "$MANIFEST_DIR/$PROFILE_NAME.jsonl" \
    --workers "$WORKERS" \
    --overwrite

echo "DONE dataset ready at $DATASET_DIR"
echo "Next: sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml"
