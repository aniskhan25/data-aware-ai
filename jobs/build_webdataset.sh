#!/bin/bash
# Pack the dataset into tar shards.
#
#   sbatch jobs/build_webdataset.sh
#   sbatch jobs/build_webdataset.sh 1000 work           # samples/shard, balancing key
#   sbatch jobs/build_webdataset.sh 1000 count 6        # imbalanced by a factor of 6
#   sbatch jobs/build_webdataset.sh 25000 count 1 shards-few   # into another directory
#
# Arguments: SAMPLES_PER_SHARD, BALANCE_BY, IMBALANCE_FACTOR, OUTPUT_SUBDIR.
# The last two exist for Part VI, which needs a deliberately imbalanced set and a
# deliberately under-sharded one alongside the healthy shards.
#
# Shards are written with fixed member metadata, so rebuilding from the same manifest
# and plan produces byte-identical archives.
#
# Shard count matters for Part VI: there must be at least as many shards as readers,
# or some readers sit idle. With the defaults below and the balanced profile, 20 000
# samples at 1 000 per shard gives 20 shards, comfortably more than one node's ranks.

#SBATCH --job-name=daai-build-shards
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
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

require_vars TUTORIAL_ROOT
report_allocation

SAMPLES_PER_SHARD="${1:-1000}"
BALANCE_BY="${2:-count}"
IMBALANCE_FACTOR="${3:-1.0}"
OUTPUT_SUBDIR="${4:-shards}"
SOURCE_DIR="$TUTORIAL_ROOT/source"
SHARD_DIR="$TUTORIAL_ROOT/$OUTPUT_SUBDIR"
MANIFEST="$SOURCE_DIR/manifest.jsonl"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR manifest not found: $MANIFEST" >&2
    echo "Generate the dataset first: sbatch jobs/prepare_dataset.sh" >&2
    exit 2
fi

echo "SOURCE_DIR=$SOURCE_DIR"
echo "SHARD_DIR=$SHARD_DIR"
echo "SAMPLES_PER_SHARD=$SAMPLES_PER_SHARD"
echo "BALANCE_BY=$BALANCE_BY"
echo "IMBALANCE_FACTOR=$IMBALANCE_FACTOR"

run_python scripts/build_webdataset.py \
    --source "$SOURCE_DIR" \
    --manifest "$MANIFEST" \
    --output "$SHARD_DIR" \
    --samples-per-shard "$SAMPLES_PER_SHARD" \
    --balance-by "$BALANCE_BY" \
    --imbalance-factor "$IMBALANCE_FACTOR" \
    --overwrite

echo "DONE ${SLURM_JOB_ID:-local}"
echo "Next: sbatch jobs/run_loader.sh configs/baseline/webdataset.yaml"
