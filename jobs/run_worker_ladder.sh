#!/bin/bash
# Submit the whole Part IV worker ladder, one job per rung.
#
#   ./jobs/run_worker_ladder.sh                       # loose-files, 1 run per rung
#   ./jobs/run_worker_ladder.sh squashfs 3            # squashfs layout, 3 repeats
#   ./jobs/run_worker_ladder.sh webdataset 2 run.measured_batches=1000
#
# Trailing key=value arguments are applied to every rung. Use them to lengthen the
# measured window: at high throughput 200 batches can be under a second of
# measurement, which is far too short to be stable.
#
# Run this from a login node: it calls sbatch, it is not itself a batch script.
# Each rung is an ordinary jobs/run_loader.sh job, so a single rung can be re-run
# on its own afterwards.
#
# Repeats matter here more than in Part III: the ladder is read as a *shape*, and a
# single noisy rung can invent a plateau or hide a regression.

set -euo pipefail

COMMON="$(dirname "$0")/common.sh"
# shellcheck source=/dev/null
source "$COMMON"

require_vars TUTORIAL_ROOT

LAYOUT="${1:-loose-files}"
REPEATS="${2:-1}"
shift 2 2>/dev/null || shift $# || true
# Whatever remains is applied to every rung, so the ladder stays controlled.
COMMON_OVERRIDES=("$@")

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || (( REPEATS < 1 )); then
    echo "ERROR repeats must be a positive integer, got '$REPEATS'" >&2
    exit 2
fi
if [[ -z "${SBATCH_ACCOUNT:-}" ]]; then
    echo "ERROR SBATCH_ACCOUNT is not set in this shell." >&2
    echo "Run 'source env.sh' before submitting: sbatch resolves the account at" >&2
    echo "submission time, so setting it inside the job is too late." >&2
    exit 2
fi

# The ladder configs are written for the loose-file layout, so a different layout
# needs its own root as well as its own loader settings. Getting only half of that
# right points the run at the wrong data, so both are set together here.
EXTRA=()
case "$LAYOUT" in
    loose-files)
        ;;
    webdataset)
        # Shards live elsewhere, and a streaming layout has no index to permute —
        # the configuration rejects shuffle: true for it.
        EXTRA+=("dataset.root=$TUTORIAL_ROOT/shards" "loader.shuffle=false")
        ;;
    squashfs)
        # The image must already be bound at dataset.root; see jobs/build_squashfs.sh.
        EXTRA+=("dataset.root=$TUTORIAL_ROOT/mnt/source")
        ;;
    *)
        echo "ERROR unknown layout '$LAYOUT'" >&2
        echo "Expected one of: loose-files, squashfs, webdataset" >&2
        exit 2
        ;;
esac

RUNGS=(workers_0 workers_2 workers_7 workers_13 workers_oversubscribed)
IDS=()

for RUNG in "${RUNGS[@]}"; do
    CONFIG="configs/workers/$RUNG.yaml"
    [[ -f "$CONFIG" ]] || { echo "ERROR missing $CONFIG" >&2; exit 2; }
    for (( R=1; R<=REPEATS; R++ )); do
        OUT="$TUTORIAL_ROOT/outputs/workers/$RUNG-$LAYOUT-r$R"
        JOB=$(sbatch --parsable jobs/run_loader.sh "$CONFIG" \
            "dataset.layout=$LAYOUT" "output.directory=$OUT" \
            ${EXTRA[@]+"${EXTRA[@]}"} \
            ${COMMON_OVERRIDES[@]+"${COMMON_OVERRIDES[@]}"})
        IDS+=("$JOB")
        echo "SUBMITTED $RUNG repeat $R -> job $JOB"
    done
done

echo
echo "LADDER_JOBS=$(IFS=,; echo "${IDS[*]}")"
echo "LAYOUT=$LAYOUT"
echo "REPEATS=$REPEATS"
echo
echo "When they finish:"
echo "  python3 scripts/compare_workers.py \\"
echo "      \"\$TUTORIAL_ROOT\"/outputs/workers/*-$LAYOUT-r*/run_summary.json"
