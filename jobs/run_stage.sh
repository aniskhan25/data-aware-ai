#!/bin/bash
# Submit a build step and its measurement as a dependent pair.
#
#   ./jobs/run_stage.sh squashfs
#   ./jobs/run_stage.sh webdataset
#   ./jobs/run_stage.sh flash
#
# sbatch returns as soon as a job is queued, so submitting a build and a benchmark back
# to back is a race: the benchmark can start before the artifact exists. Slurm's
# afterok dependency is the fix - the second job runs only if the first succeeded, and
# is cancelled if it did not.
#
# Run this from a login node; it calls sbatch rather than being one.

set -euo pipefail

COMMON="$(dirname "$0")/common.sh"
# shellcheck source=/dev/null
source "$COMMON"

require_vars TUTORIAL_ROOT

STAGE="${1:-}"
case "$STAGE" in
    squashfs)
        BUILD=(jobs/build_squashfs.sh)
        MEASURE=(jobs/run_loader.sh configs/baseline/squashfs.yaml)
        ;;
    webdataset)
        BUILD=(jobs/build_webdataset.sh 1250 count 1.0 shards)
        MEASURE=(jobs/run_loader.sh configs/baseline/webdataset.yaml)
        ;;
    flash)
        BUILD=(jobs/copy_to_flash.sh shards)
        MEASURE=(jobs/run_storage_comparison.sh configs/staging/flash.yaml)
        ;;
    *)
        echo "usage: ./jobs/run_stage.sh {squashfs|webdataset|flash}" >&2
        echo "Chains a build job and its measurement with an afterok dependency." >&2
        exit 2
        ;;
esac

if [[ -z "${SBATCH_ACCOUNT:-}" ]]; then
    echo "ERROR SBATCH_ACCOUNT is not set. Run 'source env.sh' first: sbatch resolves" >&2
    echo "the account at submission time, so setting it inside the job is too late." >&2
    exit 2
fi

BUILD_ID=$(sbatch --parsable "${BUILD[@]}")
echo "BUILD_JOB=$BUILD_ID  (${BUILD[*]})"

MEASURE_ID=$(sbatch --parsable --dependency="afterok:${BUILD_ID}" "${MEASURE[@]}")
echo "MEASURE_JOB=$MEASURE_ID  (${MEASURE[*]})"
echo
echo "The measurement runs only if the build succeeds, and is cancelled otherwise."
echo "Watch both with: squeue -j $BUILD_ID,$MEASURE_ID"
