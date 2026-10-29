#!/bin/bash

#SBATCH -M wice
#SBATCH --account=lp_h_pancreatic_surgery
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=126000MB
#SBATCH --time=12:00:00
#SBATCH --exclude=k28g30
#SBATCH --job-name=CAPS_ALPS_nnUnet_validate
#SBATCH --output=log/CAPS_ALPS_nnUNet_validate_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=inigo.bermejo@uhasselt.be

set -euo pipefail


# ==========================================================
# USER SETTINGS (edit these for your own VSC account)
# ==========================================================

# VSC project number
PROJECT="365"

# VSC username
VSC_USER="vsc36501"

# Data directory
DATA_DIR="/data/leuven/${PROJECT}/${VSC_USER}"

source ${DATA_DIR}/miniconda3/etc/profile.d/conda.sh

conda activate nnunet

# ==========================================================
# nnU-Net environment variables
# ==========================================================
# Scratch directory
SCRATCH_DIR="/scratch/leuven/${PROJECT}/${VSC_USER}/CAPS"

# The following directories are required by nnU-Net.
# For inference only, nnUNet_raw and nnUNet_preprocessed may remain empty.
export nnUNet_raw="${SCRATCH_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${SCRATCH_DIR}/nnUNet_preprocessed"
export nnUNet_results="${SCRATCH_DIR}/nnUNet_results"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export nnUNet_n_proc_DA=4
export nnUNet_compile=F

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

DATASET_ID=1

PLANNER="nnUNetPlannerResEncL"
PLANS="nnUNetResEncUNetLPlans"
CONFIG="3d_fullres"

FOLD=4

# ------------------------------------------------------------
# Environment diagnostics
# ------------------------------------------------------------

echo "============================================================"
echo "JOB START"
echo "============================================================"

echo "Node:"
hostname

echo "Starting"
date

echo "Python:"
which python
python --version

echo "nnU-Net:"
which nnUNetv2_train

date
hostname

echo "GPU:"
nvidia-smi

echo "CPU:"
nproc



# ============================================================
# SETTINGS
# ============================================================

DATASET_NAME="Dataset001_PancreasTumor"

PLANS="nnUNetResEncUNetLPlans"
CONFIG="3d_fullres"

IMAGES_DIR="$DATA_DIR/CAPS/Jessa/nnUNet/nnUNet_raw/$DATASET_NAME/imagesTr"
SPLITS_FILE="$nnUNet_preprocessed/$DATASET_NAME/splits_final.json"

MODEL_DIR="$nnUNet_results/$DATASET_NAME/nnUNetTrainer__${PLANS}__${CONFIG}/fold_${FOLD}"

OUTPUT_DIR="$nnUNet_results/${DATASET_NAME}/OOF_best/fold_${FOLD}"

# ============================================================
# CHECK CHECKPOINT
# ============================================================

if [[ ! -f "$MODEL_DIR/checkpoint_best.pth" &&
      ! -f "$MODEL_DIR/checkpoint_best.safetensors" ]]; then
    echo "ERROR: No best checkpoint for fold $FOLD"
    exit 1
fi

# ============================================================
# CREATE TEMPORARY INPUT FOLDER
# ============================================================

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Read the held-out case IDs and create symbolic links.
# No need to copy the large CT images.

python - "$SPLITS_FILE" "$IMAGES_DIR" "$TEMP_DIR" "$FOLD" <<'PY'
import json
import sys
from pathlib import Path

splits_file = Path(sys.argv[1])
images_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
fold = int(sys.argv[4])

with open(splits_file) as f:
    splits = json.load(f)

cases = splits[fold]["val"]

for case in cases:
    source = images_dir / f"{case}_0000.nii.gz"

    if not source.exists():
        raise FileNotFoundError(source)

    destination = output_dir / source.name
    destination.symlink_to(source.resolve())

print(f"Fold {fold}: {len(cases)} held-out images")
PY

# ============================================================
# PREDICT USING THE BEST CHECKPOINT
# ============================================================

mkdir -p "$OUTPUT_DIR"

nnUNetv2_predict \
    -i "$TEMP_DIR" \
    -o "$OUTPUT_DIR" \
    -d "$DATASET_ID" \
    -c "$CONFIG" \
    -p "$PLANS" \
    -f "$FOLD" \
    -chk checkpoint_best.pth \
    -npp 2 \
    -nps 2

echo "Predictions saved to: $OUTPUT_DIR"
date
