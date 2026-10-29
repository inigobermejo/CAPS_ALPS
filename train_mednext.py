from pathlib import Path
import os
import json
import csv
import warnings

warnings.filterwarnings(
    "ignore",
    message="no available indices of class 2 to crop"
)

import numpy as np
import torch
import SimpleITK as sitk

from nnunet_mednext import create_mednext_v1

from monai.data import (
    Dataset,
    DataLoader,
    PersistentDataset,
    decollate_batch,
)

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss

from monai.transforms import (
    AsDiscreted,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    Invertd,
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

# Common dataset 
DATASET_DIR = Path(f"{DATA_DIR}/CAPS/Jessa/Preprocessed")

RESULTS_DIR = Path(os.environ["MedNeXt_results"])

PREDICTIONS_ROOT = Path(f"{RESULTS_DIR}/predictions") 

CACHE_DIR = Path(f"{RESULTS_DIR}/mednext_cache") 

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

N_FOLDS = 5

MAX_EPOCHS = 300
VAL_EVERY = 5

BATCH_SIZE = 1
NUM_SAMPLES = 2
NUM_WORKERS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_CLASSES = 3

# Same preprocessing as Swin UNETR
ROI_SIZE = (128, 128, 96)
SPACING = (1.0, 1.0, 2.0)

HU_MIN = -175
HU_MAX = 250

SW_BATCH_SIZE = 2
INFER_OVERLAP = 0.5

# MedNeXt architecture
# S, B, M, L are supported
MODEL_ID = "B"

KERNEL_SIZE = 3

# Keep False for simple apples-to-apples training with Swin.
DEEP_SUPERVISION = False

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# LOAD FOLDS
# ============================================================

with open(DATASET_DIR / "folds.json") as f:
    fold_data = json.load(f)

folds = fold_data["folds"]

all_cases = sorted(
    p.name.replace(".nii.gz", "")
    for p in (DATASET_DIR / "labelsTr").glob("*.nii.gz")
)


def make_case(case_id):

    return {
        "image": str(
            DATASET_DIR
            / "imagesTr"
            / f"{case_id}_0000.nii.gz"
        ),
        "label": str(
            DATASET_DIR
            / "labelsTr"
            / f"{case_id}.nii.gz"
        ),
        "case_id": case_id,
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

    SpatialPadd(
        keys=["image", "label"],
        spatial_size=ROI_SIZE
    ),

    RandCropByLabelClassesd(
        keys=["image", "label"],
        label_key="label",
        spatial_size=ROI_SIZE,
    
        # background, pancreas, tumour
        ratios=[0.1, 0.2, 0.7],
    
        num_classes=3,
        num_samples=4,
    
        # Do not restrict sampling using image intensity
        image_key=None,
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
        keys=["image", "label"],
        track_meta=True
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

    SpatialPadd(
        keys=["image", "label"],
        spatial_size=ROI_SIZE
    ),

    EnsureTyped(
        keys=["image", "label"],
        track_meta=True
    )
])


# ============================================================
# DICE FOR TRAINING MONITORING
# ============================================================

def binary_dice(gt, pred):
    """
    Same basic convention as our common evaluator.

    If both GT and prediction are empty, returns NaN.
    If GT exists but prediction is empty, returns 0.
    """

    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    if not gt.any() and not pred.any():
        return np.nan

    denominator = gt.sum() + pred.sum()

    if denominator == 0:
        return np.nan

    return float(
        2.0 * np.logical_and(gt, pred).sum()
        / denominator
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    val_loader,
    device
):

    model.eval()

    pancreas_scores = []
    tumor_scores = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(device)

            labels = batch["label"].to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):

                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=ROI_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=INFER_OVERLAP
                )

            pred = torch.argmax(
                outputs,
                dim=1
            )

            gt = labels[:, 0]

            pred_np = pred.cpu().numpy()
            gt_np = gt.cpu().numpy()

            for i in range(pred_np.shape[0]):

                # ------------------------------------------
                # Anatomical whole pancreas:
                # label 1 + tumour label 2
                # ------------------------------------------

                gt_pancreas = (
                    (gt_np[i] == 1)
                    | (gt_np[i] == 2)
                )

                pred_pancreas = (
                    (pred_np[i] == 1)
                    | (pred_np[i] == 2)
                )

                pancreas_scores.append(
                    binary_dice(
                        gt_pancreas,
                        pred_pancreas
                    )
                )

                # ------------------------------------------
                # Tumour
                # ------------------------------------------

                tumor_scores.append(
                    binary_dice(
                        gt_np[i] == 2,
                        pred_np[i] == 2
                    )
                )

    pancreas_dice = float(
        np.nanmean(pancreas_scores)
    )

    tumor_dice = float(
        np.nanmean(tumor_scores)
    )

    mean_dice = float(
        np.nanmean([
            pancreas_dice,
            tumor_dice
        ])
    )

    return (
        pancreas_dice,
        tumor_dice,
        mean_dice
    )


# ============================================================
# SAVE PREDICTIONS IN ORIGINAL GEOMETRY
# ============================================================

def save_predictions(
    model,
    val_loader,
    fold,
    device
):

    print()
    print(
        f"Saving predictions for fold {fold}"
    )

    output_dir = (
        PREDICTIONS_ROOT
        / "mednext"
        / f"fold_{fold}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Reverse preprocessing:
    # padding -> crop -> spacing -> orientation
    post_transform = Compose([
        Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="image",
    
            # logits are continuous
            nearest_interp=False,
    
            to_tensor=True,
        ),
    
        AsDiscreted(
            keys="pred",
            argmax=True,
        ),
    ])

    model.eval()

    with torch.no_grad():

        for batch in val_loader:

            case_id = batch["case_id"][0]

            images = batch["image"].to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):

                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=ROI_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=INFER_OVERLAP
                )

            # Discrete segmentation in preprocessed space
            batch["pred"] = logits.detach().cpu()

            # Invertd operates case-by-case
            item = decollate_batch(batch)[0]

            item = post_transform(item)

            pred_original = (
                item["pred"]
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # ----------------------------------------------
            # Ground-truth NIfTI defines exact target geometry
            # ----------------------------------------------

            gt_file = (
                DATASET_DIR
                / "labelsTr"
                / f"{case_id}.nii.gz"
            )

            gt_img = sitk.ReadImage(
                str(gt_file)
            )

            expected_xyz = tuple(
                gt_img.GetSize()
            )

            if pred_original.shape != expected_xyz:

                raise RuntimeError(
                    f"{case_id}: inverted prediction "
                    f"shape={pred_original.shape}, "
                    f"expected={expected_xyz}"
                )

            # MONAI/NIfTI spatial order = X,Y,Z
            # SimpleITK array order       = Z,Y,X
            pred_zyx = np.transpose(
                pred_original,
                (2, 1, 0)
            )

            pred_img = sitk.GetImageFromArray(
                pred_zyx
            )

            # Exact size/spacing/origin/direction of common GT
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
    print(f"MEDNEXT - FOLD {fold}")
    print("=" * 70)

    val_cases = folds[str(fold)]

    train_cases = [
        c
        for c in all_cases
        if c not in val_cases
    ]

    print(
        f"Training:   {len(train_cases)}"
    )

    print(
        f"Validation: {len(val_cases)}"
    )

    train_files = [
        make_case(c)
        for c in train_cases
    ]

    val_files = [
        make_case(c)
        for c in val_cases
    ]

    # ========================================================
    # DATASETS
    #
    # PersistentDataset avoids resampling every CT again
    # on every epoch.
    # ========================================================

    train_ds = PersistentDataset(
        data=train_files,
        transform=train_transforms,
        cache_dir=(
            CACHE_DIR
            / f"fold_{fold}"
            / "train"
        )
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
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ========================================================
    # MEDNEXT
    # ========================================================

    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=NUM_CLASSES,
        model_id=MODEL_ID,
        kernel_size=KERNEL_SIZE,
        deep_supervision=DEEP_SUPERVISION
    ).to(device)

    print(
        f"MedNeXt-{MODEL_ID}, "
        f"kernel={KERNEL_SIZE}"
    )

    # ========================================================
    # LOSS / OPTIMIZER
    # ========================================================

    dice_loss_function = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False
    )
    
    ce_loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            [
                0.2,   # background
                1.0,   # pancreas
                4.0,   # tumour
            ],
            device=device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    fold_dir = (
        RESULTS_DIR
        / f"fold_{fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_dice = -1.0
    best_epoch = -1

    history = []

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(MAX_EPOCHS):

        model.train()

        epoch_loss = 0.0

        for batch in train_loader:

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):

                outputs = model(images)
                
                dice_loss = dice_loss_function(
                    outputs,
                    labels
                )
                
                ce_loss = ce_loss_function(
                    outputs,
                    labels[:, 0].long()
                )
                
                loss = dice_loss + ce_loss

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            epoch_loss += float(
                loss.item()
            )

        scheduler.step()

        epoch_loss /= len(train_loader)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Fold {fold} | "
            f"Epoch {epoch + 1:03d}/{MAX_EPOCHS} | "
            f"loss={epoch_loss:.4f} | "
            f"lr={lr:.6g}"
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        if (
            (epoch + 1) % VAL_EVERY != 0
            and epoch + 1 != MAX_EPOCHS
        ):
            continue

        (
            pancreas_dice,
            tumor_dice,
            mean_dice
        ) = validate(
            model,
            val_loader,
            device
        )

        print(
            f"Validation | "
            f"whole pancreas DSC={pancreas_dice:.4f} | "
            f"tumour DSC={tumor_dice:.4f} | "
            f"mean={mean_dice:.4f}"
        )

        history.append({
            "epoch":
                epoch + 1,

            "train_loss":
                epoch_loss,

            "whole_pancreas_dice":
                pancreas_dice,

            "tumor_dice":
                tumor_dice,

            "mean_dice":
                mean_dice,

            "learning_rate":
                lr,
        })

        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if mean_dice > best_dice:

            best_dice = mean_dice
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch":
                        best_epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_dice":
                        best_dice,

                    "model_id":
                        MODEL_ID,

                    "kernel_size":
                        KERNEL_SIZE,
                },
                fold_dir / "best_model.pt"
            )

            print(
                f"New best checkpoint: "
                f"{best_dice:.4f}"
            )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_file = (
        fold_dir
        / "training_history.csv"
    )

    with open(
        history_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "whole_pancreas_dice",
                "tumor_dice",
                "mean_dice",
                "learning_rate",
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print()
    print(
        f"Best fold {fold} mean DSC: "
        f"{best_dice:.4f} "
        f"(epoch {best_epoch})"
    )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        fold_dir / "best_model.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # SAVE VALIDATION PREDICTIONS
    # ========================================================

    save_predictions(
        model=model,
        val_loader=val_loader,
        fold=fold,
        device=device
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for fold in range(N_FOLDS):

        train_fold(fold)