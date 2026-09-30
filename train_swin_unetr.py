# train_swin_unetr.py

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
)

from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = (96, 96, 96)

NUM_CLASSES = 3
IN_CHANNELS = 1

FEATURE_SIZE = 48

BATCH_SIZE = 1
NUM_SAMPLES_PER_VOLUME = 4

MAX_EPOCHS = 500

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

VAL_EVERY = 5

NUM_WORKERS = 4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ============================================================
# DATA
# ============================================================

def get_case_files(data_dir):

    data_dir = Path(data_dir)

    images_dir = data_dir / "imagesTr"
    labels_dir = data_dir / "labelsTr"

    cases = []

    for image_file in sorted(images_dir.glob("*_0000.nii.gz")):

        case_id = image_file.name.replace("_0000.nii.gz", "")

        label_file = labels_dir / f"{case_id}.nii.gz"

        if not label_file.exists():
            raise FileNotFoundError(
                f"Missing label for {case_id}: {label_file}"
            )

        cases.append(
            {
                "image": str(image_file),
                "label": str(label_file),
                "case_id": case_id,
            }
        )

    print(f"Found {len(cases)} cases")

    return cases


# ============================================================
# TRANSFORMS
# ============================================================

def get_train_transforms():

    return Compose([

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
            pixdim=(1.0, 1.0, 2.0),
            mode=("bilinear", "nearest")
        ),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=-150,
            a_max=250,
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
            spatial_size=PATCH_SIZE,
            pos=2,
            neg=1,
            num_samples=NUM_SAMPLES_PER_VOLUME,
            image_key="image",
            image_threshold=0
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=0,
            prob=0.5
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=1,
            prob=0.5
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=2,
            prob=0.5
        ),

        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=3
        ),

        RandShiftIntensityd(
            keys=["image"],
            offsets=0.1,
            prob=0.5
        ),

        EnsureTyped(
            keys=["image", "label"]
        )
    ])


def get_val_transforms():

    return Compose([

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
            pixdim=(1.0, 1.0, 2.0),
            mode=("bilinear", "nearest")
        ),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=-150,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),

        CropForegroundd(
            keys=["image", "label"],
            source_key="image"
        ),

        EnsureTyped(
            keys=["image", "label"]
        )
    ])


# ============================================================
# MODEL
# ============================================================

def create_model(device):

    model = SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        use_checkpoint=True,
        spatial_dims=3
    )

    return model.to(device)


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    scaler,
    device
):

    model.train()

    epoch_loss = 0

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():

            outputs = model(images)

            loss = loss_function(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device
):

    model.eval()

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=False
    )

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = sliding_window_inference(
            inputs=images,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
            keepdim=True
        )

        predictions_onehot = torch.nn.functional.one_hot(
            predictions[:, 0].long(),
            num_classes=NUM_CLASSES
        )

        predictions_onehot = predictions_onehot.permute(
            0, 4, 1, 2, 3
        ).float()

        labels_onehot = torch.nn.functional.one_hot(
            labels[:, 0].long(),
            num_classes=NUM_CLASSES
        )

        labels_onehot = labels_onehot.permute(
            0, 4, 1, 2, 3
        ).float()

        dice_metric(
            predictions_onehot,
            labels_onehot
        )

    dice = dice_metric.aggregate().item()

    dice_metric.reset()

    return dice


# ============================================================
# MAIN TRAINING
# ============================================================

def train_fold(
    train_cases,
    val_cases,
    fold,
    output_dir,
    device
):

    print()
    print("=" * 70)
    print(f"TRAINING FOLD {fold}")
    print("=" * 70)

    train_dataset = Dataset(
        data=train_cases,
        transform=get_train_transforms()
    )

    val_dataset = Dataset(
        data=val_cases,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model = create_model(device)

    loss_function = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True
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

    scaler = torch.cuda.amp.GradScaler()

    fold_dir = Path(output_dir) / f"fold_{fold}"
    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_dice = -1

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            scaler,
            device
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": loss
        }

        if epoch % VAL_EVERY == 0:

            dice = validate(
                model,
                val_loader,
                device
            )

            record["val_dice"] = dice

            print(
                f"Fold {fold} | "
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | "
                f"Dice {dice:.4f}"
            )

            if dice > best_dice:

                best_dice = dice

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "dice": dice
                    },
                    fold_dir / "best_model.pt"
                )

                print(
                    f"  New best model: Dice={dice:.4f}"
                )

        else:

            print(
                f"Fold {fold} | "
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f}"
            )

        history.append(record)

    with open(
        fold_dir / "history.json",
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    print(
        f"\nFold {fold} best Dice: {best_dice:.4f}"
    )

    return best_dice


# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

def create_folds(cases, seed=42):

    from sklearn.model_selection import KFold

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    folds = []

    for train_idx, val_idx in kfold.split(cases):

        train_cases = [
            cases[i]
            for i in train_idx
        ]

        val_cases = [
            cases[i]
            for i in val_idx
        ]

        folds.append(
            {
                "train": train_cases,
                "val": val_cases
            }
        )

    return folds


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="./results"
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None
    )

    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    cases = get_case_files(
        args.data_dir
    )

    folds = create_folds(
        cases,
        SEED
    )

    # Save fold assignments
    split_dir = Path(args.data_dir) / "splits"
    split_dir.mkdir(exist_ok=True)

    for i, fold in enumerate(folds):

        with open(
            split_dir / f"fold_{i}.json",
            "w"
        ) as f:

            json.dump(
                {
                    "train": [
                        x["case_id"]
                        for x in fold["train"]
                    ],
                    "val": [
                        x["case_id"]
                        for x in fold["val"]
                    ]
                },
                f,
                indent=2
            )

    if args.fold is not None:

        fold = args.fold

        train_fold(
            folds[fold]["train"],
            folds[fold]["val"],
            fold,
            args.output_dir,
            device
        )

    else:

        results = []

        for fold in range(5):

            dice = train_fold(
                folds[fold]["train"],
                folds[fold]["val"],
                fold,
                args.output_dir,
                device
            )

            results.append(dice)

        print()
        print("=" * 70)
        print("5-FOLD RESULTS")
        print("=" * 70)

        for i, dice in enumerate(results):

            print(
                f"Fold {i}: {dice:.4f}"
            )

        print(
            f"\nMean Dice: {np.mean(results):.4f}"
        )

        print(
            f"SD Dice:   {np.std(results):.4f}"
        )


if __name__ == "__main__":
    main()