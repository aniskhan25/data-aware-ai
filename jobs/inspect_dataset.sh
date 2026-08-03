#!/bin/bash
# Characterise a dataset before allocating any GPUs.
#
#   sbatch jobs/inspect_dataset.sh                        # inspects $TUTORIAL_ROOT/source
#   sbatch jobs/inspect_dataset.sh /scratch/.../my-data
#
# Inspection reads metadata only, never file contents, so it needs no GPU. Run it
# as a job rather than on a login node when the tree is large: walking millions of
# entries is a sustained metadata workload that logins should not carry.
#
# --mem matters here beyond the walk itself: it is what the staging advice is
# measured against, so request the memory you actually intend to train with if you
# want that advice to mean anything.

#SBATCH --job-name=daai-inspect
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

if [[ -n "${1:-}" ]]; then
    DATASET_PATH="$1"
    shift
else
    require_vars TUTORIAL_ROOT
    DATASET_PATH="$TUTORIAL_ROOT/source"
fi

if [[ ! -e "$DATASET_PATH" ]]; then
    echo "ERROR path does not exist: $DATASET_PATH" >&2
    echo "Generate the dataset first: sbatch jobs/prepare_dataset.sh" >&2
    exit 2
fi

report_allocation

if [[ -n "${TUTORIAL_ROOT:-}" ]]; then
    REPORT_DIR="$TUTORIAL_ROOT/outputs/inspection"
else
    REPORT_DIR="outputs/inspection"
fi
mkdir -p "$REPORT_DIR"

echo "DATASET_PATH=$DATASET_PATH"
echo "REPORT_DIR=$REPORT_DIR"

run_python scripts/inspect_dataset.py \
    --path "$DATASET_PATH" \
    --output "$REPORT_DIR/dataset_report.json" \
    --progress-every 100000 \
    --verbose \
    ${@+"$@"}

echo "DONE ${SLURM_JOB_ID:-local}"
echo "Next: sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml"
