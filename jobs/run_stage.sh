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
        # Repeats matter as much here as in the worker ladder: a single run on a
        # shared filesystem can be off by a factor of four.
        REPEATS="${2:-3}"
        shift 2 2>/dev/null || shift $#
        EXTRA=("$@")
        IDS=()
        if [[ "$STAGE" == layouts ]]; then
            KIND=layouts
            OUTPUT="$TUTORIAL_ROOT/outputs/layout-comparison/summary.json"
            BUILD=(jobs/build_squashfs.sh)
            IMAGE_BUILD=$(sbatch --parsable "${BUILD[@]}")
            BUILD=(jobs/build_webdataset.sh 1250 count 1.0 shards)
            SHARD_BUILD=$(sbatch --parsable "${BUILD[@]}")
            echo "  builds: squashfs $IMAGE_BUILD, shards $SHARD_BUILD" >&2
            # Runs are chained one after another, not fired off together. Concurrent
            # repeats read the same source tree and warm each other's page cache, so
            # they are not independent measurements: run side by side, the loose-file
            # baseline reads mostly from cache and looks four times faster than it is.
            PREV="afterany:$IMAGE_BUILD:$SHARD_BUILD"
            for (( R=1; R<=REPEATS; R++ )); do
                for L in loose_files squashfs webdataset; do
                    OUT="$TUTORIAL_ROOT/outputs/layouts/$L-r$R"
                    ID=$(sbatch --parsable --dependency="$PREV" jobs/run_loader.sh \
                        "configs/baseline/$L.yaml" "output.directory=$OUT" \
                        ${EXTRA[@]+"${EXTRA[@]}"})
                    IDS+=("$ID")
                    PREV="afterany:$ID"
                done
            done
            PATTERN="$TUTORIAL_ROOT/outputs/layouts/*/run_summary.json"
        else
            KIND=storage
            PATTERN="$TUTORIAL_ROOT/outputs/storage/*/run_summary.json"
            OUTPUT="$TUTORIAL_ROOT/outputs/storage-comparison/summary.json"
            PREV=""
            for (( R=1; R<=REPEATS; R++ )); do
                for CONFIG in configs/staging/scratch.yaml configs/staging/tmp.yaml; do
                    ID=$(sbatch --parsable ${PREV:+--dependency="$PREV"} \
                        jobs/run_storage_comparison.sh "$CONFIG" \
                        ${EXTRA[@]+"${EXTRA[@]}"})
                    IDS+=("$ID")
                    PREV="afterany:$ID"
                done
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
        echo "usage: ./jobs/run_stage.sh {layouts|storage} [REPEATS] [key=value ...]" >&2
        echo "       ./jobs/run_stage.sh {squashfs|webdataset|flash}" >&2
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
