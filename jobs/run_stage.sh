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

if [[ -z "${SBATCH_ACCOUNT:-}" ]]; then
    echo "ERROR SBATCH_ACCOUNT is not set. Run 'source env.sh' first: sbatch resolves" >&2
    echo "the account at submission time, so setting it inside the job is too late." >&2
    exit 2
fi

# Submit one build and its measurement, and echo the measurement's job id. The
# measurement runs only if the build succeeds, and is cancelled otherwise.
chain() {
    local build_id measure_id
    build_id=$(sbatch --parsable "${BUILD[@]}")
    measure_id=$(sbatch --parsable --dependency="afterok:${build_id}" "${MEASURE[@]}")
    echo "  ${1}: build $build_id -> measure $measure_id" >&2
    printf '%s' "$measure_id"
}

# The composite targets submit every run for a step plus the comparison that reads
# them, so one command produces the measurements and the report together.
case "$STAGE" in
    layouts|storage)
        IDS=()
        if [[ "$STAGE" == layouts ]]; then
            KIND=layouts
            PATTERN="$TUTORIAL_ROOT/outputs/*/run_summary.json"
            OUTPUT="$TUTORIAL_ROOT/outputs/layout-comparison/summary.json"
            IDS+=("$(sbatch --parsable jobs/run_loader.sh configs/baseline/loose_files.yaml)")
            echo "  loose-files: measure ${IDS[-1]}" >&2
            BUILD=(jobs/build_squashfs.sh)
            MEASURE=(jobs/run_loader.sh configs/baseline/squashfs.yaml)
            IDS+=("$(chain squashfs)")
            BUILD=(jobs/build_webdataset.sh 1250 count 1.0 shards)
            MEASURE=(jobs/run_loader.sh configs/baseline/webdataset.yaml)
            IDS+=("$(chain webdataset)")
        else
            KIND=storage
            PATTERN="$TUTORIAL_ROOT/outputs/storage/*/run_summary.json"
            OUTPUT="$TUTORIAL_ROOT/outputs/storage-comparison/summary.json"
            for CONFIG in configs/staging/scratch.yaml configs/staging/tmp.yaml; do
                IDS+=("$(sbatch --parsable jobs/run_storage_comparison.sh "$CONFIG")")
                echo "  $(basename "$CONFIG" .yaml): measure ${IDS[-1]}" >&2
            done
        fi
        COMPARE=$(sbatch --parsable \
            --dependency="afterok:$(IFS=:; echo "${IDS[*]}")" \
            jobs/compare.sh "$KIND" "$PATTERN" "$OUTPUT")
        echo
        echo "MEASURE_JOBS=$(IFS=,; echo "${IDS[*]}")"
        echo "COMPARE_JOB=$COMPARE"
        echo "REPORT=$OUTPUT"
        exit 0
        ;;
esac

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
        echo "usage: ./jobs/run_stage.sh {layouts|storage|squashfs|webdataset|flash}" >&2
        echo "Chains a build job and its measurement with an afterok dependency." >&2
        exit 2
        ;;
esac

BUILD_ID=$(sbatch --parsable "${BUILD[@]}")
echo "BUILD_JOB=$BUILD_ID  (${BUILD[*]})"

MEASURE_ID=$(sbatch --parsable --dependency="afterok:${BUILD_ID}" "${MEASURE[@]}")
echo "MEASURE_JOB=$MEASURE_ID  (${MEASURE[*]})"
echo
echo "The measurement runs only if the build succeeds, and is cancelled otherwise."
echo "Watch both with: squeue -j $BUILD_ID,$MEASURE_ID"
