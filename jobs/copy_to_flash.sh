#!/bin/bash
# Copy the dataset artifacts to project flash, for the Part V flash rung.
#
#   sbatch jobs/copy_to_flash.sh            # copies the shard directory
#   sbatch jobs/copy_to_flash.sh source     # copies the loose tree instead
#
# Flash is a placement, not a staging target: the copy persists between jobs, so it is
# made once here rather than inside every measured run. That is also why the flash
# rung reports no staging cost - the copy is not part of the job being measured.
#
# Requires TUTORIAL_FLASH_ROOT in env.sh. Flash is much smaller than scratch; check
# your quota with lumi-workspaces before copying a large dataset.

#SBATCH --job-name=daai-copy-flash
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
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

require_vars TUTORIAL_ROOT TUTORIAL_FLASH_ROOT
report_allocation

WHAT="${1:-shards}"
SOURCE="$TUTORIAL_ROOT/$WHAT"
DESTINATION="$TUTORIAL_FLASH_ROOT/$WHAT"

if [[ ! -e "$SOURCE" ]]; then
    echo "ERROR nothing to copy: $SOURCE" >&2
    exit 2
fi

mkdir -p "$TUTORIAL_FLASH_ROOT"
echo "SOURCE=$SOURCE"
echo "DESTINATION=$DESTINATION"
echo "SOURCE_BYTES=$(du -sb "$SOURCE" | cut -f1)"

START=$SECONDS
rm -rf "$DESTINATION"
cp -r "$SOURCE" "$DESTINATION"
echo "COPY_SECONDS=$(( SECONDS - START ))"

SOURCE_FILES=$(find "$SOURCE" -type f | wc -l)
DEST_FILES=$(find "$DESTINATION" -type f | wc -l)
echo "SOURCE_FILES=$SOURCE_FILES"
echo "DEST_FILES=$DEST_FILES"
if [[ "$SOURCE_FILES" != "$DEST_FILES" ]]; then
    echo "ERROR copy is incomplete: $DEST_FILES of $SOURCE_FILES files" >&2
    exit 1
fi

echo "DONE flash copy ready at $DESTINATION"
echo "Next: sbatch jobs/run_storage_comparison.sh configs/staging/flash.yaml"
