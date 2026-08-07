# Site configuration. Copy to env.sh and edit. env.sh is deliberately gitignored:
# it holds your project allocation, which must never be committed.
#
#   cp env.example.sh env.sh
#   $EDITOR env.sh
#
# Job scripts source env.sh automatically when it exists.

# Your LUMI project. Find it with: groups | tr ' ' '\n' | grep project_
export LUMI_PROJECT=project_XXXXXXXXX

# Slurm bills to this account. Setting SBATCH_ACCOUNT means you do not have to
# pass --account on every sbatch command.
export SBATCH_ACCOUNT="$LUMI_PROJECT"

# Where the tutorial keeps datasets, packaged images, and results.
# Project scratch is the documented location for job input and output I/O.
export TUTORIAL_ROOT="/scratch/$LUMI_PROJECT/data-aware-ai"

# Project flash, used only by the optional Part V flash experiment. Comment this
# out if your project has no flash allocation.
export TUTORIAL_FLASH_ROOT="/flash/$LUMI_PROJECT/data-aware-ai"

# Optional: a container to run the Python workload inside. Leave unset to use
# whatever python is already on PATH (a module, a venv, or a conda environment).
#
# The LUMI AI Factory image carries pyarrow, datasets, and h5py, so all three
# optional format tracks run without installing anything. The sif-images PyTorch
# containers carry the first two but not h5py.
#
# The "latest" symlink moves when a new image is published. Pin the versioned
# path if you want a run to stay reproducible.
# export TUTORIAL_CONTAINER="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
# export TUTORIAL_CONTAINER="/scratch/$LUMI_PROJECT/containers/pytorch.sif"

# Extra bind mounts for the container, comma separated. These are added to the
# repository bind, not a replacement for it.
#
# Anything the job reads or writes must appear here, or it will be invisible inside
# the container. That includes project flash if you run the Part V flash rung - its
# absence shows up as a confusing "file not found" for a path that plainly exists.
export TUTORIAL_CONTAINER_BINDS="/scratch/$LUMI_PROJECT,/flash/$LUMI_PROJECT"
#
# To read a SquashFS image through a container bind (Part III), point one of these
# at the image with image-src, matching dataset.root in the config:
# export TUTORIAL_CONTAINER_BINDS="$TUTORIAL_ROOT/source.squashfs:$TUTORIAL_ROOT/mnt/source:image-src=/"
