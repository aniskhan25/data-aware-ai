#!/bin/bash
# Package the dataset into one SquashFS image.
#
#   sbatch jobs/build_squashfs.sh
#
# Packaging is CPU and I/O work with no GPU involved, and it is a one-off: the image
# is read-only, so it only needs rebuilding when the dataset changes.
#
# The default arguments store data blocks uncompressed, because the tutorial dataset
# is JPEG and PNG whose bytes are already compressed. Compressing them again would
# spend CPU on every read for almost no space saved.

#SBATCH --job-name=daai-build-squashfs
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$(dirname "$0")/common.sh"

require_vars TUTORIAL_ROOT
report_allocation

SOURCE_DIR="${1:-$TUTORIAL_ROOT/source}"
IMAGE_PATH="${2:-$TUTORIAL_ROOT/source.squashfs}"
MOUNT_POINT="$TUTORIAL_ROOT/mnt/source"

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "ERROR dataset not found: $SOURCE_DIR" >&2
    echo "Generate it first: sbatch jobs/prepare_dataset.sh" >&2
    exit 2
fi

echo "SOURCE_DIR=$SOURCE_DIR"
echo "IMAGE_PATH=$IMAGE_PATH"

run_python scripts/build_squashfs.py \
    --source "$SOURCE_DIR" \
    --image "$IMAGE_PATH" \
    --overwrite

mkdir -p "$MOUNT_POINT"
cat <<NEXT

The image is built. To measure it, make it readable at
  $MOUNT_POINT
which is what configs/baseline/squashfs.yaml expects (squashfs_mode: prebound).

Two ways, both documented at
https://docs.lumi-supercomputer.eu/storage/formats/FUSE/

  1. Bind it into your container at launch, which needs no privileges. Add to
     env.sh (this is added to the repository bind, not a replacement):
       export TUTORIAL_CONTAINER_BINDS="$IMAGE_PATH:$MOUNT_POINT:image-src=/"

  2. Let the loader mount it with squashfuse, by setting in the config:
       dataset:
         squashfs_mode: squashfuse
         image: $IMAGE_PATH

Then: sbatch jobs/run_loader.sh configs/baseline/squashfs.yaml
NEXT
