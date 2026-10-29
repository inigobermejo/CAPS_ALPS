from pathlib import Path
import os
import json
import shutil
import subprocess

import torch
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import SimpleITK as sitk

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt
)


# ============================================================
# SETTINGS
# ============================================================
# VSC project number
PROJECT="365"

# VSC username
VSC_USER="vsc36501"

# Data directory
DATA_DIR= f"/data/leuven/{PROJECT}/{VSC_USER}"
CAPS_DATA_DIR= f"{DATA_DIR}/CAPS"

# Common dataset 
#COMMON_DATASET = Path(f"{DATA_DIR}/CAPS/Jessa/Preprocessed")
#COMMON_DATASET = Path(f"{DATA_DIR}/CAPS/MSD_Task07")
COMMON_DATASET = Path(f"{DATA_DIR}/CAPS/Jessa/nnUNet/nnUNet_raw/Dataset003_JessaMultiOrgan")

# nnU-Net folders
NNUNET_RAW = Path(
    os.environ["nnUNet_raw"]
)

NNUNET_PREPROCESSED = Path(
    os.environ["nnUNet_preprocessed"]
)

NNUNET_RESULTS = Path(
    os.environ["nnUNet_results"]
)

PREDICTIONS_ROOT = Path(f"{NNUNET_RESULTS}/predictions") 

# Dataset number/name
#DATASET_ID = 1
#DATASET_NAME = "PancreasTumor"
#DATASET_ID = 2
#DATASET_NAME = "MSDPancreas"
DATASET_ID = 3
DATASET_NAME = "JessaMultiOrgan"

DATASET_FOLDER = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"

# nnU-Net configuration
CONFIGURATION = "3d_fullres"

# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["nnUNet_raw"] = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
os.environ["nnUNet_results"] = str(NNUNET_RESULTS)


# ============================================================
# COMMAND HELPER
# ============================================================

def run(cmd):

    print()
    print("=" * 70)
    print(" ".join(map(str, cmd)))
    print("=" * 70)

    subprocess.run(
        cmd,
        check=True
    )


# ============================================================
# 1. CREATE nnU-Net RAW DATASET
# ============================================================

def prepare_raw_dataset():

    target = NNUNET_RAW / DATASET_FOLDER

    images_target = target / "imagesTr"
    labels_target = target / "labelsTr"

    images_target.mkdir(
        parents=True,
        exist_ok=True
    )

    labels_target.mkdir(
        parents=True,
        exist_ok=True
    )

    # Copy images
    for src in (COMMON_DATASET / "imagesTr").glob("*.nii.gz"):

        dst = images_target / src.name

        if not dst.exists():
            shutil.copy2(src, dst)

    # Copy labels
    for src in (COMMON_DATASET / "labelsTr").glob("*.nii.gz"):

        dst = labels_target / src.name

        if not dst.exists():
            shutil.copy2(src, dst)

    n_cases = len(
        list(labels_target.glob("*.nii.gz"))
    )
    
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
    
        "labels": {
            "background": 0,
            "aorta": 1,
            "celiac_artery": 2,
            "common_bile_duct": 3,
            "duodenum": 4,
            "pancreas": 5,
            "pancreatic_duct": 6,
            "IVC": 7,
            "spleen": 8,
            "SMA": 9,
            "pancreatic_lesion_tumor": 10,
            "portal_vein": 11,
            "SMV": 12,
            "splenic_vein": 13,
            "IMV": 14
        },
    
        "numTraining": n_cases,
    
        "file_ending": ".nii.gz"
    }    

    with open(
        target / "dataset.json",
        "w"
    ) as f:

        json.dump(
            dataset_json,
            f,
            indent=2
        )

    print(
        f"Prepared {n_cases} cases in\n"
        f"{target}"
    )


# ============================================================
# 2. PLAN + PREPROCESS
# ============================================================

def preprocess():

    run([
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(DATASET_ID),
        "-c",
        "3d_fullres",
        "--verify_dataset_integrity",
        "-pl",
        "nnUNetPlannerResEncL",
    ])


# ============================================================
# 3. CONVERT OUR folds.json → nnU-Net splits_final.json
# ============================================================

def create_nnunet_splits():

    with open(
        COMMON_DATASET / "folds.json"
    ) as f:

        folds_json = json.load(f)

    folds = folds_json["folds"]

    all_cases = sorted(
        p.name.replace(".nii.gz", "")
        for p in
        (COMMON_DATASET / "labelsTr").glob("*.nii.gz")
    )

    splits = []

    for fold_number in range(5):

        val_cases = folds[str(fold_number)]

        train_cases = [
            c for c in all_cases
            if c not in val_cases
        ]

        splits.append({
            "train": train_cases,
            "val": val_cases
        })

        print(
            f"Fold {fold_number}: "
            f"{len(train_cases)} train / "
            f"{len(val_cases)} val"
        )

    split_file = (
        NNUNET_PREPROCESSED
        / DATASET_FOLDER
        / "splits_final.json"
    )

    with open(
        split_file,
        "w"
    ) as f:

        json.dump(
            splits,
            f,
            indent=2
        )

    print(
        f"\nCustom folds written to:\n{split_file}"
    )

# ============================================================
# LOGGING + PLOTTING
# ============================================================

def export_training_history(fold):
    """
    Extract nnU-Net's epoch-by-epoch training history from the
    final checkpoint and save it as CSV + plots.

    Tracks:
        - training loss
        - validation loss
        - pancreas pseudo-Dice
        - tumour pseudo-Dice
        - mean foreground pseudo-Dice
        - EMA pseudo-Dice
        - learning rate
        - epoch duration
    """

    fold_dir = (
        NNUNET_RESULTS
        / DATASET_FOLDER
        / f"nnUNetTrainer__nnUNetPlans__{CONFIGURATION}"
        / f"fold_{fold}"
    )

    checkpoint_file = fold_dir / "checkpoint_final.pth"

    if not checkpoint_file.exists():
        raise RuntimeError(
            f"Checkpoint not found:\n{checkpoint_file}"
        )

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False
    )

    log = checkpoint["logging"]

    train_loss = log["train_losses"]
    val_loss = log["val_losses"]
    mean_dice = log["mean_fg_dice"]
    ema_dice = log["ema_fg_dice"]
    dice_per_class = log["dice_per_class_or_region"]
    lrs = log["lrs"]

    start_times = log["epoch_start_timestamps"]
    end_times = log["epoch_end_timestamps"]

    n_epochs = len(train_loss)

    rows = []

    for epoch in range(n_epochs):

        class_dice = dice_per_class[epoch]

        pancreas_dice = (
            float(class_dice[0])
            if len(class_dice) > 0
            else np.nan
        )

        tumor_dice = (
            float(class_dice[1])
            if len(class_dice) > 1
            else np.nan
        )

        epoch_seconds = (
            float(end_times[epoch] - start_times[epoch])
        )

        rows.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss[epoch]),
            "val_loss": float(val_loss[epoch]),
            "pancreas_pseudo_dice": pancreas_dice,
            "tumor_pseudo_dice": tumor_dice,
            "mean_pseudo_dice": float(mean_dice[epoch]),
            "ema_pseudo_dice": float(ema_dice[epoch]),
            "learning_rate": float(lrs[epoch]),
            "epoch_seconds": epoch_seconds,
        })

    df = pd.DataFrame(rows)

    csv_file = fold_dir / "training_history.csv"

    df.to_csv(
        csv_file,
        index=False
    )

    # --------------------------------------------------------
    # Loss plot
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Training loss"
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"nnU-Net fold {fold}: loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fold_dir / "loss_history.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Dice plot
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        df["epoch"],
        df["pancreas_pseudo_dice"],
        label="Pancreas pseudo-Dice"
    )

    plt.plot(
        df["epoch"],
        df["tumor_pseudo_dice"],
        label="Tumour pseudo-Dice"
    )

    plt.plot(
        df["epoch"],
        df["mean_pseudo_dice"],
        label="Mean pseudo-Dice"
    )

    plt.plot(
        df["epoch"],
        df["ema_pseudo_dice"],
        label="EMA mean pseudo-Dice"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Pseudo-Dice")
    plt.ylim(0, 1)
    plt.title(f"nnU-Net fold {fold}: validation performance")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fold_dir / "dice_history.png",
        dpi=150
    )

    plt.close()

    print(
        f"Saved training history for fold {fold}:\n"
        f"    {csv_file}"
    )

def copy_nnunet_predictions(fold):

    source_dir = (
        NNUNET_RESULTS
        / DATASET_FOLDER
        / f"nnUNetTrainer__nnUNetPlans__{CONFIGURATION}"
        / f"fold_{fold}"
        / "validation"
    )

    target_dir = (
        PREDICTIONS_ROOT
        / "nnunet"
        / f"fold_{fold}"
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions = list(
        source_dir.glob("*.nii.gz")
    )

    if not predictions:
        raise RuntimeError(
            f"No validation predictions found in:\n"
            f"{source_dir}"
        )

    for src in predictions:

        shutil.copy2(
            src,
            target_dir / src.name
        )

    print(
        f"Copied {len(predictions)} predictions "
        f"to {target_dir}"
    )
# ============================================================
# 4. TRAIN FIVE FOLDS
# ============================================================
    
def train_fold(fold):

    run([
        "nnUNetv2_train",
        str(DATASET_ID),
        CONFIGURATION,
        str(fold),
        "--val_best",
        "-p",
        "nnUNetResEncUNetLPlans"
    ])
    export_training_history(fold)

    copy_nnunet_predictions(fold)

def train():

    for fold in range(5):

        train_fold(fold)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run nnU-Net planning and preprocessing"
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        choices=range(5)
    )

    args = parser.parse_args()

    if args.preprocess:
        prepare_raw_dataset()
        preprocess()
        create_nnunet_splits()

    train_fold(args.fold)