#!/bin/bash -l

#SBATCH -M wice
#SBATCH --account=lp_h_pancreatic_surgery
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=126000MB
#SBATCH --time=12:00:00
#SBATCH --job-name=CAPS_ALPS_train_nnUnet
#SBATCH --output=CAPS_ALPS_nnUNet_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=inigo.bermejo@uhasselt.be

set -e

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

echo "============================================================"
echo "STARTING PRETRAINING"
echo "============================================================"

# ============================================================
# Stage 2: Binary pancreatic lesion/tumour segmentation
# ============================================================

DATASET_ID=4
DATASET_NAME="Dataset004_PancreaticLesionCropped"
PLANNER="nnUNetPlannerResEncL"
PLANS="nnUNetResEncUNetLPlans"
CONFIG="3d_fullres"

MODE="train"
FOLD="0"

STAGE2_ROOT="${DATA_DIR}/CAPS/Jessa/nnUNet/nnUNet_raw/Dataset004_PancreaticLesionCropped"

if [[ -z "$MODE" ]]; then
    echo "Usage:"
    echo "  $0 setup"
    echo "  $0 train FOLD"
    echo "  $0 resume FOLD"
    echo "  $0 validate FOLD"
    exit 1
fi

: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"

RAW_DATASET="${nnUNet_raw}/${DATASET_NAME}"
PREP_DATASET="${nnUNet_preprocessed}/${DATASET_NAME}"

# ============================================================
# SETUP
# ============================================================

if [[ "$MODE" == "setup" ]]; then

    : "${STAGE2_ROOT:?Set STAGE2_ROOT to your cropped dataset folder}"

    python - "$STAGE2_ROOT" "$RAW_DATASET" <<'PY'
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

source = Path(sys.argv[1])
target = Path(sys.argv[2])

images_dir = source / "imagesTr"
labels_dir = source / "labelsTr"
folds_file = source / "folds.json"

for path in [images_dir, labels_dir, folds_file]:
    if not path.exists():
        raise FileNotFoundError(path)

images = sorted(images_dir.glob("*_0000.nii.gz"))

if not images:
    raise RuntimeError("No cropped CT images found")

suffix = "_0000.nii.gz"
case_ids = [p.name[:-len(suffix)] for p in images]
cases = set(case_ids)

if len(cases) != len(case_ids):
    raise RuntimeError("Duplicate case identifiers")

# ------------------------------------------------------------
# Convert existing folds.json into nnU-Net splits
# ------------------------------------------------------------

with open(folds_file) as f:
    folds_data = json.load(f)

# Expected structure from the earlier cropping script:
# {"n_folds": 5, "seed": 42,
#  "folds": {"0": [case_ids], ..., "4": [case_ids]}}

folds = folds_data["folds"]
splits = []
all_validation = []

for fold in range(5):
    val = sorted(set(folds[str(fold)]))

    unknown = set(val) - cases
    if unknown:
        raise RuntimeError(
            f"Fold {fold} has unknown case IDs: {sorted(unknown)}"
        )

    train = sorted(cases - set(val))

    if not val or not train:
        raise RuntimeError(f"Empty split in fold {fold}")

    splits.append({"train": train, "val": val})
    all_validation.extend(val)

if len(all_validation) != len(cases) or set(all_validation) != cases:
    raise RuntimeError(
        "The five validation folds must partition all cases "
        "exactly once"
    )

# ------------------------------------------------------------
# Check that labelmaps really are binary and spatially aligned
# ------------------------------------------------------------

n_positive = 0
n_negative = 0

for image_path in images:
    case_id = image_path.name[:-len(suffix)]
    label_path = labels_dir / f"{case_id}.nii.gz"

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    ct = sitk.ReadImage(str(image_path))
    seg = sitk.ReadImage(str(label_path))

    same_geometry = (
        ct.GetSize() == seg.GetSize()
        and np.allclose(ct.GetSpacing(), seg.GetSpacing())
        and np.allclose(ct.GetOrigin(), seg.GetOrigin(), atol=1e-3)
        and np.allclose(ct.GetDirection(), seg.GetDirection())
    )

    if not same_geometry:
        raise RuntimeError(
            f"Image/label geometry mismatch: {case_id}"
        )

    values = np.unique(sitk.GetArrayViewFromImage(seg))

    if not np.isin(values, [0, 1]).all():
        raise RuntimeError(
            f"Unexpected labels in {case_id}: {values}"
        )

    if 1 in values:
        n_positive += 1
    else:
        n_negative += 1

print(f"Cases:             {len(cases)}")
print(f"Tumour-positive:   {n_positive}")
print(f"Tumour-negative:   {n_negative}")

for i, split in enumerate(splits):
    print(
        f"Fold {i}: "
        f"{len(split['train'])} training, "
        f"{len(split['val'])} validation"
    )

# ------------------------------------------------------------
# Copy into nnUNet_raw/Dataset004...
# ----------------------------------------------------------
        

dataset_json = {
    "channel_names": {"0": "CT"},
    "labels": {
        "background": 0,
        "pancreatic_lesion_tumour": 1
    },
    "numTraining": len(cases),
    "file_ending": ".nii.gz"
}

with open(target / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

with open(target / "nnunet_folds.json", "w") as f:
    json.dump(splits, f, indent=2)

print(f"Created {target}")
PY

    echo "Planning and preprocessing Dataset004..."

    nnUNetv2_plan_and_preprocess \
        -d "$DATASET_ID" \
        -c "$CONFIG" \
        -pl "$PLANNER" \
        --verify_dataset_integrity

    # nnU-Net reads these splits during training.
    cp \
        "${RAW_DATASET}/nnunet_folds.json" \
        "${PREP_DATASET}/splits_final.json"

    echo "Dataset004 setup complete."
    exit 0
fi

# ============================================================
# TRAIN / RESUME / VALIDATE
# ============================================================

if [[ ! "$FOLD" =~ ^[0-4]$ ]]; then
    echo "Specify a fold between 0 and 4"
    exit 1
fi

MODEL_DIR="${nnUNet_results}/${DATASET_NAME}/nnUNetTrainer_500epochs__${PLANS}__${CONFIG}/fold_${FOLD}"

case "$MODE" in

    train)
        if [[ -f "${MODEL_DIR}/checkpoint_latest.pth" ||
              -f "${MODEL_DIR}/checkpoint_final.pth" ]]; then
            echo "A checkpoint already exists for fold $FOLD."
            echo "Use 'resume' to continue the existing run."
            exit 1
        fi

        nnUNetv2_train \
            "$DATASET_ID" "$CONFIG" "$FOLD" \
            -tr nnUNetTrainer_500epochs \
            -p "$PLANS"
        ;;

    resume)
        if [[ ! -f "${MODEL_DIR}/checkpoint_latest.pth" &&
              ! -f "${MODEL_DIR}/checkpoint_final.pth" &&
              ! -f "${MODEL_DIR}/checkpoint_best.pth" ]]; then
            echo "No checkpoint found for fold $FOLD"
            exit 1
        fi

        nnUNetv2_train \
            "$DATASET_ID" "$CONFIG" "$FOLD" \
            -tr nnUNetTrainer_500epochs \
            -p "$PLANS" --c
        ;;

    validate)
        nnUNetv2_train \
            "$DATASET_ID" "$CONFIG" "$FOLD" \
            -tr nnUNetTrainer_500epochs \
            -p "$PLANS" --val --val_best
        ;;

    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac
