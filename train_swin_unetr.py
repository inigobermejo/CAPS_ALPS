from pathlib import Path
import os
import json
import csv

import torch
import numpy as np

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

import SimpleITK as sitk

from monai.transforms import (
    Compose,
    Invertd,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    AsDiscrete,
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

# Model
MODEL_NAME = "SwinUNETR"

# Common dataset 
DATASET_DIR = Path(f"{DATA_DIR}/CAPS/Jessa/Preprocessed")

RESULTS_DIR = Path(os.environ["SwinUNETR_results"])

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

N_FOLDS = 5

MAX_EPOCHS = 300
VAL_EVERY = 5

BATCH_SIZE = 1
NUM_WORKERS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

ROI_SIZE = (96, 96, 96)

# MONAI Swin UNETR abdominal CT defaults
SPACING = (1.5, 1.5, 2.0)

HU_MIN = -175
HU_MAX = 250

FEATURE_SIZE = 48

NUM_CLASSES = 3

SW_BATCH_SIZE = 2
INFER_OVERLAP = 0.5

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# LOAD FOLDS
# ============================================================

with open(
    DATASET_DIR / "folds.json"
) as f:
    fold_data = json.load(f)

folds = fold_data["folds"]

all_cases = sorted(
    p.name.replace(".nii.gz", "")
    for p in (DATASET_DIR / "labelsTr").glob("*.nii.gz")
)


def make_case(case_id):

    return {
        "image":
            str(
                DATASET_DIR
                / "imagesTr"
                / f"{case_id}_0000.nii.gz"
            ),

        "label":
            str(
                DATASET_DIR
                / "labelsTr"
                / f"{case_id}.nii.gz"
            ),

        "case_id":
            case_id
    }


# ============================================================
# TRANSFORMS
# ============================================================

train_transforms = Compose([

    LoadImaged(
        keys=["image", "label"]
    ),

    EnsureChannelFirstd(
        keys=["image", "label"]
    ),

    Orientationd(
        keys=["image", "label"],
        axcodes="RAS"
    ),

    Spacingd(
        keys=["image", "label"],
        pixdim=SPACING,
        mode=("bilinear", "nearest")
    ),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=HU_MIN,
        a_max=HU_MAX,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    CropForegroundd(
        keys=["image", "label"],
        source_key="image"
    ),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=ROI_SIZE,
        pos=2,
        neg=1,
        num_samples=2,
        image_key="image",
        image_threshold=0
    ),

    RandFlipd(
        keys=["image", "label"],
        spatial_axis=0,
        prob=0.2
    ),

    RandFlipd(
        keys=["image", "label"],
        spatial_axis=1,
        prob=0.2
    ),

    RandFlipd(
        keys=["image", "label"],
        spatial_axis=2,
        prob=0.2
    ),

    RandRotate90d(
        keys=["image", "label"],
        prob=0.2,
        max_k=3
    ),

    RandScaleIntensityd(
        keys=["image"],
        factors=0.1,
        prob=0.1
    ),

    RandShiftIntensityd(
        keys=["image"],
        offsets=0.1,
        prob=0.1
    ),

    EnsureTyped(
        keys=["image", "label"]
    )
])


val_transforms = Compose([

    LoadImaged(
        keys=["image", "label"]
    ),

    EnsureChannelFirstd(
        keys=["image", "label"]
    ),

    Orientationd(
        keys=["image", "label"],
        axcodes="RAS"
    ),

    Spacingd(
        keys=["image", "label"],
        pixdim=SPACING,
        mode=("bilinear", "nearest")
    ),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=HU_MIN,
        a_max=HU_MAX,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    CropForegroundd(
        keys=["image", "label"],
        source_key="image"
    ),

    EnsureTyped(
        keys=["image", "label"],
        track_meta=True
    )
])

post_transforms = Compose([

    Invertd(
        keys="pred",
        transform=val_transforms,
        orig_keys="image",
        nearest_interp=True,
        to_tensor=True
    )

])

# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

def save_validation_predictions(
    model,
    val_loader,
    fold,
    device
):

    print()
    print(
        f"Saving validation predictions "
        f"for fold {fold}..."
    )

    output_dir = (
        RESULTS_DIR.parent
        / "predictions"
        / "swin_unetr"
        / f"fold_{fold}"
    )

    # Or, if using the common environment variable:
    output_dir = (
        Path(
            os.environ.get(
                "PREDICTIONS_ROOT",
                "./predictions"
            )
        )
        / "swin_unetr"
        / f"fold_{fold}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model.eval()

    with torch.no_grad():

        for batch in val_loader:

            case_id = batch["case_id"][0]

            images = batch["image"].to(device)

            # ----------------------------------------------
            # Prediction in Swin preprocessing space
            # ----------------------------------------------

            with torch.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=ROI_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=INFER_OVERLAP
                )

            # Convert probabilities/logits to discrete labels
            pred = torch.argmax(
                logits,
                dim=1,
                keepdim=True
            )

            # Store prediction in the MONAI dictionary so
            # Invertd can use the image transform history.
            batch["pred"] = pred

            # batch_size = 1 here
            item = decollate_batch(batch)[0]

            # ----------------------------------------------
            # Reverse:
            # crop -> spacing -> orientation
            # ----------------------------------------------

            item = post_transforms(item)

            pred_original = (
                item["pred"]
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # ----------------------------------------------
            # Force exact geometry of our common GT
            # ----------------------------------------------

            gt_file = (
                DATASET_DIR
                / "labelsTr"
                / f"{case_id}.nii.gz"
            )

            gt_img = sitk.ReadImage(
                str(gt_file)
            )

            # MONAI/NIfTI array order here is X,Y,Z.
            # SimpleITK GetImageFromArray expects Z,Y,X.
            expected_shape = tuple(
                gt_img.GetSize()
            )

            if pred_original.shape != expected_shape:

                raise RuntimeError(
                    f"{case_id}: restored prediction "
                    f"shape {pred_original.shape} "
                    f"does not match original "
                    f"{expected_shape}"
                )

            pred_zyx = np.transpose(
                pred_original,
                (2, 1, 0)
            )

            pred_img = sitk.GetImageFromArray(
                pred_zyx
            )

            # This guarantees spacing/origin/direction are
            # identical to our common dataset.
            pred_img.CopyInformation(
                gt_img
            )

            output_file = (
                output_dir
                / f"{case_id}.nii.gz"
            )

            sitk.WriteImage(
                pred_img,
                str(output_file),
                useCompression=True
            )

            print(
                f"  saved {case_id}"
            )


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(fold):

    print()
    print("=" * 70)
    print(f"TRAINING FOLD {fold}")
    print("=" * 70)

    val_cases = folds[str(fold)]

    train_cases = [
        c for c in all_cases
        if c not in val_cases
    ]

    print(
        f"Training cases:   {len(train_cases)}"
    )

    print(
        f"Validation cases: {len(val_cases)}"
    )

    train_files = [
        make_case(c)
        for c in train_cases
    ]

    val_files = [
        make_case(c)
        for c in val_cases
    ]

    # --------------------------------------------------------
    # Datasets / loaders
    # --------------------------------------------------------

    train_ds = Dataset(
        data=train_files,
        transform=train_transforms
    )

    val_ds = Dataset(
        data=val_files,
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        use_checkpoint=True
    ).to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS
    )

    # --------------------------------------------------------
    # Dice metric
    # --------------------------------------------------------

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch"
    )

    post_pred = AsDiscrete(
        argmax=True,
        to_onehot=NUM_CLASSES
    )

    post_label = AsDiscrete(
        to_onehot=NUM_CLASSES
    )

    # Mixed precision
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    best_dice = -1
    best_epoch = -1

    fold_dir = (
        RESULTS_DIR
        / f"fold_{fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(MAX_EPOCHS):

        model.train()

        epoch_loss = 0

        for batch in train_loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = loss_function(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()

        epoch_loss /= len(train_loader)

        print(
            f"Fold {fold} | "
            f"Epoch {epoch + 1}/{MAX_EPOCHS} | "
            f"Loss = {epoch_loss:.4f}"
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        if (
            (epoch + 1) % VAL_EVERY != 0
            and epoch != MAX_EPOCHS - 1
        ):
            continue

        model.eval()

        dice_metric.reset()

        with torch.no_grad():

            for batch in val_loader:

                images = batch["image"].to(device)

                labels = batch["label"].to(device)

                with torch.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):

                    outputs = sliding_window_inference(
                        inputs=images,
                        roi_size=ROI_SIZE,
                        sw_batch_size=SW_BATCH_SIZE,
                        predictor=model,
                        overlap=INFER_OVERLAP
                    )

                predictions = [
                    post_pred(x)
                    for x in outputs
                ]

                labels_list = [
                    post_label(x)
                    for x in labels
                ]

                dice_metric(
                    y_pred=predictions,
                    y=labels_list
                )

        class_dice = (
            dice_metric
            .aggregate()
            .cpu()
            .numpy()
        )

        dice_metric.reset()

        # class_dice contains:
        # [pancreas, tumour]
        pancreas_dice = float(
            class_dice[0]
        )

        tumor_dice = float(
            class_dice[1]
        )

        mean_dice = float(
            np.nanmean(class_dice)
        )

        print(
            f"Validation | "
            f"Pancreas DSC = {pancreas_dice:.4f} | "
            f"Tumour DSC = {tumor_dice:.4f} | "
            f"Mean = {mean_dice:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "pancreas_dice": pancreas_dice,
            "tumor_dice": tumor_dice,
            "mean_dice": mean_dice
        })

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if mean_dice > best_dice:

            best_dice = mean_dice
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "best_dice":
                        best_dice
                },
                fold_dir / "best_model.pt"
            )

            print(
                f"New best model: "
                f"DSC={best_dice:.4f}"
            )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    with open(
        fold_dir / "history.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "loss",
                "pancreas_dice",
                "tumor_dice",
                "mean_dice"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    # ========================================================
    # Load best model
    # ========================================================

    checkpoint = torch.load(
        fold_dir / "best_model.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # Save validation predictions
    # ========================================================

    save_validation_predictions(
        model=model,
        val_loader=val_loader,
        fold=fold,
        device=device
    )

    print()
    print(
        f"Fold {fold} completed. "
        f"Best mean DSC={best_dice:.4f} "
        f"at epoch {best_epoch}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for fold in range(N_FOLDS):
        train_fold(fold)