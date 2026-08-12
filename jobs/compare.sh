#!/bin/bash
# Compare finished runs and write the report to disk.
#
#   sbatch jobs/compare.sh workers "$TUTORIAL_ROOT/outputs/workers/*/run_summary.json"
#
# Normally submitted for you, with an afterok dependency on the runs it reads, so
# that one command produces both the measurements and the report. The glob must be
# quoted: it is expanded here, once the runs have actually written their summaries.

#SBATCH --job-name=daai-compare
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
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

KIND="${1:-}"
PATTERN="${2:-}"
if [[ -z "$KIND" || -z "$PATTERN" ]]; then
    echo "usage: sbatch jobs/compare.sh {layouts|workers|storage} 'GLOB' [OUTPUT]" >&2
    exit 2
fi
OUTPUT="${3:-$TUTORIAL_ROOT/outputs/$KIND-comparison/summary.json}"

# shellcheck disable=SC2206 - the glob is meant to expand into separate paths here.
SUMMARIES=($PATTERN)
if [[ ${#SUMMARIES[@]} -lt 2 ]]; then
    echo "ERROR $PATTERN matched ${#SUMMARIES[@]} summaries; at least two are needed" >&2
    exit 2
fi

run_python scripts/compare.py "$KIND" "${SUMMARIES[@]}" --output "$OUTPUT"
