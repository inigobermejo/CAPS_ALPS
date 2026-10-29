#!/bin/bash -l

#SBATCH -M wice
#SBATCH --account=lp_h_pancreatic_surgery
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=126000MB
#SBATCH --time=12:00:00
#SBATCH --job-name=CAPS_ALPS_TL_nnUnet_MSD
#SBATCH --output=CAPS_ALPS_TL_nnUnet_MSD_%j.out
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

# ============================================================
# ARGUMENTS
#
# Examples:
#
#   ./nnunet_transfer.sh setup
#   ./nnunet_transfer.sh pretrain
#   ./nnunet_transfer.sh finetune 0
#   ./nnunet_transfer.sh validate 0
#
# ============================================================

MODE="finetune"
FOLD=4

echo "============================================================"
echo "STARTING PRETRAINING"
echo "============================================================"


# ============================================================
# DATASETS
# ============================================================

# Your 50 CAPS scans: dataset we ultimately care about
TARGET_ID=1
TARGET_NAME="Dataset001_PancreasTumor"

# MSD Task07: pretraining dataset
SOURCE_ID=2
SOURCE_NAME="Dataset002_MSDPancreas"


# ============================================================
# nnU-Net plans
# ============================================================

# Current ResEnc-L planner
PLANNER="nnUNetPlannerResEncL"

# Plans produced by that planner
TARGET_PLANS="nnUNetResEncUNetLPlans"

# Copy of the CAPS plans used to preprocess/train MSD.
# Give it a different name so we don't overwrite MSD's own plans.
SOURCE_PLANS="CAPSResEncLPretrainPlans"

CONFIG="3d_fullres"


# ============================================================
# BASIC CHECKS
# ============================================================

echo "============================================================"
echo "nnU-Net transfer-learning pipeline"
echo "============================================================"

echo "Mode:               $MODE"
echo "Target dataset:     $TARGET_ID ($TARGET_NAME)"
echo "Source dataset:     $SOURCE_ID ($SOURCE_NAME)"
echo "Planner:            $PLANNER"
echo "Target plans:       $TARGET_PLANS"
echo "Source plans:       $SOURCE_PLANS"
echo "Configuration:      $CONFIG"

echo
echo "nnUNet_raw:          $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results:      $nnUNet_results"
echo

which nnUNetv2_train
python --version


# ============================================================
# STEP 1:
# PLAN + PREPROCESS TARGET CAPS DATA
# ============================================================

if [ "$MODE" = "setup" ]; then

    echo
    echo "============================================================"
    echo "1. Planning CAPS with ResEnc-L"
    echo "============================================================"

    nnUNetv2_plan_and_preprocess \
        -d "$TARGET_ID" \
        -c "$CONFIG" \
        -pl "$PLANNER" \
        --verify_dataset_integrity


    # ========================================================
    # STEP 2:
    # EXTRACT MSD FINGERPRINT
    # ========================================================

    echo
    echo "============================================================"
    echo "2. Extracting MSD fingerprint"
    echo "============================================================"

    nnUNetv2_extract_fingerprint \
        -d "$SOURCE_ID"


    # ========================================================
    # STEP 3:
    # TRANSFER CAPS PLANS -> MSD
    #
    # This forces the MSD pretraining model to have exactly
    # the architecture/preprocessing expected by CAPS.
    # ========================================================

    echo
    echo "============================================================"
    echo "3. Transferring CAPS ResEnc-L plans to MSD"
    echo "============================================================"

    nnUNetv2_move_plans_between_datasets \
        -s "$TARGET_ID" \
        -t "$SOURCE_ID" \
        -sp "$TARGET_PLANS" \
        -tp "$SOURCE_PLANS"


    # ========================================================
    # STEP 4:
    # PREPROCESS MSD USING THE TRANSFERRED CAPS PLANS
    # ========================================================

    echo
    echo "============================================================"
    echo "4. Preprocessing MSD using CAPS plans"
    echo "============================================================"

    nnUNetv2_preprocess \
        -d "$SOURCE_ID" \
        -c "$CONFIG" \
        -plans_name "$SOURCE_PLANS"


    echo
    echo "============================================================"
    echo "SETUP COMPLETE"
    echo "============================================================"

    echo
    echo "Next run:"
    echo "  $0 pretrain"

    exit 0
fi


# ============================================================
# STEP 5:
# PRETRAIN ON ALL MSD CASES
# ============================================================

if [ "$MODE" = "pretrain" ]; then

    echo
    echo "============================================================"
    echo "Pretraining on MSD"
    echo "============================================================"

    echo "Dataset: $SOURCE_ID"
    echo "Plans:   $SOURCE_PLANS"
    echo "Fold:    all"

    nnUNetv2_train \
        "$SOURCE_ID" \
        "$CONFIG" \
        all \
        -tr nnUNetTrainer_500epochs \
        -p "$SOURCE_PLANS"


    echo
    echo "============================================================"
    echo "MSD PRETRAINING COMPLETE"
    echo "============================================================"

    CHECKPOINT_DIR="${nnUNet_results}/${SOURCE_NAME}/nnUNetTrainer__${SOURCE_PLANS}__${CONFIG}/fold_all"

    echo
    echo "Expected checkpoints:"
    echo "  ${CHECKPOINT_DIR}/checkpoint_best.pth"
    echo "  ${CHECKPOINT_DIR}/checkpoint_final.pth"

    exit 0
fi


# ============================================================
# CHECKPOINT USED FOR FINE-TUNING
# ============================================================

CHECKPOINT_DIR="${nnUNet_results}/${SOURCE_NAME}/nnUNetTrainer_500epochs__${SOURCE_PLANS}__${CONFIG}/fold_all"

PRETRAINED_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint_best.pth"


# ============================================================
# STEP 6:
# FINE-TUNE ONE CAPS FOLD
# ============================================================

if [ "$MODE" = "finetune" ]; then


    if [ -z "$FOLD" ]; then
        echo "Please specify fold 0-4"
        echo
        echo "Example:"
        echo "  $0 finetune 0"
        exit 1
    fi

    if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then

        echo "ERROR: pretrained checkpoint not found:"
        echo
        echo "$PRETRAINED_CHECKPOINT"
        echo
        echo "Run:"
        echo "  $0 pretrain"
        echo
        echo "first."

        exit 1
    fi


    echo
    echo "============================================================"
    echo "Fine-tuning CAPS fold $FOLD"
    echo "============================================================"

    echo "Initializing from:"
    echo "$PRETRAINED_CHECKPOINT"
    echo

    nnUNetv2_train \
        "$TARGET_ID" \
        "$CONFIG" \
        "$FOLD" \
        -p "$TARGET_PLANS" \
        -pretrained_weights "$PRETRAINED_CHECKPOINT"

    exit 0
fi


# ============================================================
# OPTIONAL:
# RUN VALIDATION OF BEST CHECKPOINT
# ============================================================

if [ "$MODE" = "validate" ]; then

    FOLD=${2:-}

    if [ -z "$FOLD" ]; then
        echo "Please specify fold 0-4"
        exit 1
    fi

    echo
    echo "============================================================"
    echo "Validating best CAPS checkpoint: fold $FOLD"
    echo "============================================================"

    nnUNetv2_train \
        "$TARGET_ID" \
        "$CONFIG" \
        "$FOLD" \
        -p "$TARGET_PLANS" \
        --val_best

    exit 0
fi


echo "Unknown mode: $MODE"
exit 1

echo "Finished"
date
