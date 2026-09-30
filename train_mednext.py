# train_mednext.py

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

import nibabel as nib

from sklearn.model_selection import StratifiedKFold


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = (96, 96, 96)

NUM_CLASSES = 3
IN_CHANNELS = 1

BATCH_SIZE = 1

NUM_PATCHES_PER_CASE = 4

MAX_EPOCHS = 500

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

VAL_EVERY = 5

NUM_WORKERS = 4

SEED = 42

# MedNeXt variant
MEDNEXT_VARIANT = "S"

# Number of blocks
MEDNEXT_BLOCKS = 5


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
# DATASET
# ============================================================

class PancreasDataset(Dataset):

    def __init__(
        self,
        cases,
        training=True
    ):

        self.cases = cases
        self.training = training

    def __len__(self):

        return len(self.cases)

    def load_case(self, case):

        image = nib.load(
            case["image"]
        ).get_fdata().astype(
            np.float32
        )

        label = nib.load(
            case["label"]
        ).get_fdata().astype(
            np.uint8
        )

        return image, label

    def normalize_ct(self, image):

        # Soft-tissue window
        image = np.clip(
            image,
            -150,
            250
        )

        image = (
            image + 150
        ) / 400.0

        return image

    def get_random_patch(
        self,
        image,
        label
    ):

        sx, sy, sz = image.shape

        px, py, pz = PATCH_SIZE

        # ----------------------------------------------------
        # Positive sampling
        # ----------------------------------------------------

        foreground = np.argwhere(
            label > 0
        )

        if len(foreground) > 0:

            center = foreground[
                np.random.randint(
                    len(foreground)
                )
            ]

        else:

            center = np.array([
                sx // 2,
                sy // 2,
                sz // 2
            ])

        start = []

        for c, s, p in zip(
            center,
            image.shape,
            PATCH_SIZE
        ):

            low = max(
                0,
                int(c - p // 2)
            )

            high = min(
                low + p,
                s
            )

            low = max(
                0,
                high - p
            )

            start.append(low)

        x, y, z = start

        image_patch = image[
            x:x + px,
            y:y + py,
            z:z + pz
        ]

        label_patch = label[
            x:x + px,
            y:y + py,
            z:z + pz
        ]

        # Pad if necessary
        pad_width = []

        for dim, p in zip(
            image_patch.shape,
            PATCH_SIZE
        ):

            pad_width.append(
                (0, max(0, p - dim))
            )

        if any(
            p[1] > 0
            for p in pad_width
        ):

            image_patch = np.pad(
                image_patch,
                pad_width,
                mode="constant"
            )

            label_patch = np.pad(
                label_patch,
                pad_width,
                mode="constant"
            )

        return image_patch, label_patch

    def augment(
        self,
        image,
        label
    ):

        # Random flips
        for axis in range(3):

            if np.random.rand() < 0.5:

                image = np.flip(
                    image,
                    axis=axis
                ).copy()

                label = np.flip(
                    label,
                    axis=axis
                ).copy()

        # Random intensity shift
        if np.random.rand() < 0.5:

            image += np.random.uniform(
                -0.05,
                0.05
            )

            image = np.clip(
                image,
                0,
                1
            )

        return image, label

    def __getitem__(self, idx):

        case = self.cases[idx]

        image, label = self.load_case(
            case
        )

        image = self.normalize_ct(
            image
        )

        if self.training:

            image, label = self.get_random_patch(
                image,
                label
            )

            image, label = self.augment(
                image,
                label
            )

        image = torch.from_numpy(
            image.copy()
        ).float()

        label = torch.from_numpy(
            label.copy()
        ).long()

        # NCDHW
        image = image.unsqueeze(0)

        # CDHW
        label = label.unsqueeze(0)

        return {
            "image": image,
            "label": label,
            "case_id": case["case_id"]
        }


# ============================================================
# MEDNEXT
# ============================================================

def create_model(device):

    # MedNeXt implementation
    from dynamic_network_architectures.architectures.mednext import (
        MedNeXt
    )

    if MEDNEXT_VARIANT == "S":

        kernel_size = 3
        exp_r = 2

    elif MEDNEXT_VARIANT == "B":

        kernel_size = 3
        exp_r = 4

    else:

        kernel_size = 3
        exp_r = 4

    model = MedNeXt(
        in_channels=IN_CHANNELS,
        n_channels=32,
        n_classes=NUM_CLASSES,
        exp_r=exp_r,
        kernel_size=kernel_size,
        deep_supervision=False,
        do_res=True,
        do_connect=True,
        block_counts=[
            MEDNEXT_BLOCKS,
            MEDNEXT_BLOCKS,
            MEDNEXT_BLOCKS,
            MEDNEXT_BLOCKS,
            MEDNEXT_BLOCKS
        ],
        block_type="MedNeXtBlock"
    )

    return model.to(device)


# ============================================================
# LOSS
# ============================================================

def dice_loss(
    logits,
    labels,
    smooth=1e-5
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    target = F.one_hot(
        labels[:, 0],
        num_classes=NUM_CLASSES
    )

    target = target.permute(
        0,
        4,
        1,
        2,
        3
    ).float()

    intersection = (
        probabilities * target
    ).sum(
        dim=(2, 3, 4)
    )

    denominator = (
        probabilities.sum(
            dim=(2, 3, 4)
        )
        +
        target.sum(
            dim=(2, 3, 4)
        )
    )

    dice = (
        2 * intersection + smooth
    ) / (
        denominator + smooth
    )

    # Ignore background
    dice = dice[:, 1:]

    return 1 - dice.mean()


def combined_loss(
    logits,
    labels
):

    ce = F.cross_entropy(
        logits,
        labels[:, 0]
    )

    dice = dice_loss(
        logits,
        labels
    )

    return ce + dice


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device
):

    model.train()

    total_loss = 0

    for batch in loader:

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

        with torch.cuda.amp.autocast():

            logits = model(
                images
            )

            loss = combined_loss(
                logits,
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# SLIDING-WINDOW INFERENCE
# ============================================================

@torch.no_grad()
def predict_volume(
    model,
    image,
    device,
    overlap=0.5
):

    model.eval()

    sx, sy, sz = image.shape

    px, py, pz = PATCH_SIZE

    prediction = np.zeros(
        (NUM_CLASSES, sx, sy, sz),
        dtype=np.float32
    )

    count = np.zeros(
        (sx, sy, sz),
        dtype=np.float32
    )

    step = [
        max(1, int(p * (1 - overlap)))
        for p in PATCH_SIZE
    ]

    for x in range(
        0,
        max(1, sx - px + 1),
        step[0]
    ):

        for y in range(
            0,
            max(1, sy - py + 1),
            step[1]
        ):

            for z in range(
                0,
                max(1, sz - pz + 1),
                step[2]
            ):

                x2 = min(
                    x + px,
                    sx
                )

                y2 = min(
                    y + py,
                    sy
                )

                z2 = min(
                    z + pz,
                    sz
                )

                patch = image[
                    x:x2,
                    y:y2,
                    z:z2
                ]

                # Pad
                pad = [

                    (0, px - patch.shape[0]),
                    (0, py - patch.shape[1]),
                    (0, pz - patch.shape[2])

                ]

                patch = np.pad(
                    patch,
                    pad,
                    mode="constant"
                )

                patch_tensor = torch.from_numpy(
                    patch
                ).float()

                patch_tensor = (
                    patch_tensor
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )

                logits = model(
                    patch_tensor
                )

                probs = torch.softmax(
                    logits,
                    dim=1
                )

                probs = probs[
                    0
                ].cpu().numpy()

                actual_shape = (
                    x2 - x,
                    y2 - y,
                    z2 - z
                )

                probs = probs[
                    :,
                    :actual_shape[0],
                    :actual_shape[1],
                    :actual_shape[2]
                ]

                prediction[
                    :,
                    x:x2,
                    y:y2,
                    z:z2
                ] += probs

                count[
                    x:x2,
                    y:y2,
                    z:z2
                ] += 1

    prediction /= np.maximum(
        count[None],
        1
    )

    return np.argmax(
        prediction,
        axis=0
    ).astype(
        np.uint8
    )


# ============================================================
# METRICS
# ============================================================

def calculate_dice(
    prediction,
    target,
    class_id
):

    pred = prediction == class_id
    true = target == class_id

    intersection = np.logical_and(
        pred,
        true
    ).sum()

    denominator = (
        pred.sum()
        +
        true.sum()
    )

    if denominator == 0:

        return np.nan

    return (
        2 * intersection
        / denominator
    )


@torch.no_grad()
def validate(
    model,
    cases,
    device
):

    pancreas_scores = []
    lesion_scores = []

    for case in cases:

        image = nib.load(
            case["image"]
        ).get_fdata().astype(
            np.float32
        )

        target = nib.load(
            case["label"]
        ).get_fdata().astype(
            np.uint8
        )

        image = np.clip(
            image,
            -150,
            250
        )

        image = (
            image + 150
        ) / 400

        prediction = predict_volume(
            model,
            image,
            device
        )

        pancreas_dice = calculate_dice(
            prediction,
            target,
            1
        )

        lesion_dice = calculate_dice(
            prediction,
            target,
            2
        )

        pancreas_scores.append(
            pancreas_dice
        )

        if np.any(target == 2):

            lesion_scores.append(
                lesion_dice
            )

    return {
        "pancreas_dice": np.nanmean(
            pancreas_scores
        ),
        "lesion_dice": np.nanmean(
            lesion_scores
        ) if lesion_scores else np.nan
    }


# ============================================================
# FOLDS
# ============================================================

def make_folds(cases):

    # Determine whether each case has a lesion
    strata = []

    for case in cases:

        label = nib.load(
            case["label"]
        ).get_fdata()

        strata.append(
            int(np.any(label == 2))
        )

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    folds = []

    for train_idx, val_idx in splitter.split(
        cases,
        strata
    ):

        folds.append(
            {
                "train": [
                    cases[i]
                    for i in train_idx
                ],
                "val": [
                    cases[i]
                    for i in val_idx
                ]
            }
        )

    return folds


# ============================================================
# TRAIN FOLD
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
    print(f"MEDNEXT — FOLD {fold}")
    print("=" * 70)

    train_dataset = PancreasDataset(
        train_cases,
        training=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = create_model(
        device
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

    fold_dir = (
        Path(output_dir)
        / f"fold_{fold}"
    )

    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_pancreas_dice = -1

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": loss
        }

        if epoch % VAL_EVERY == 0:

            metrics = validate(
                model,
                val_cases,
                device
            )

            record.update(
                metrics
            )

            print(
                f"Fold {fold} | "
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | "
                f"Pancreas Dice "
                f"{metrics['pancreas_dice']:.4f} | "
                f"Lesion Dice "
                f"{metrics['lesion_dice']:.4f}"
            )

            # Use pancreas Dice as checkpoint criterion
            if (
                metrics["pancreas_dice"]
                > best_pancreas_dice
            ):

                best_pancreas_dice = (
                    metrics["pancreas_dice"]
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "metrics": metrics
                    },
                    fold_dir
                    / "best_model.pt"
                )

        else:

            print(
                f"Fold {fold} | "
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f}"
            )

        history.append(
            record
        )

    with open(
        fold_dir / "history.json",
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    return history


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="./results_mednext"
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None
    )

    args = parser.parse_args()

    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Cases
    # --------------------------------------------------------

    images_dir = (
        Path(args.data_dir)
        / "imagesTr"
    )

    labels_dir = (
        Path(args.data_dir)
        / "labelsTr"
    )

    cases = []

    for image_file in sorted(
        images_dir.glob("*_0000.nii.gz")
    ):

        case_id = image_file.name.replace(
            "_0000.nii.gz",
            ""
        )

        label_file = (
            labels_dir
            / f"{case_id}.nii.gz"
        )

        if not label_file.exists():

            raise FileNotFoundError(
                label_file
            )

        cases.append(
            {
                "case_id": case_id,
                "image": str(
                    image_file
                ),
                "label": str(
                    label_file
                )
            }
        )

    print(
        f"Cases: {len(cases)}"
    )

    # --------------------------------------------------------
    # Create folds
    # --------------------------------------------------------

    folds = make_folds(
        cases
    )

    # Save fold definitions
    split_dir = (
        Path(args.data_dir)
        / "splits_mednext"
    )

    split_dir.mkdir(
        exist_ok=True
    )

    for i, fold in enumerate(
        folds
    ):

        with open(
            split_dir
            / f"fold_{i}.json",
            "w"
        ) as f:

            json.dump(
                {
                    "train": [
                        x["case_id"]
                        for x
                        in fold["train"]
                    ],
                    "val": [
                        x["case_id"]
                        for x
                        in fold["val"]
                    ]
                },
                f,
                indent=2
            )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    if args.fold is not None:

        train_fold(
            folds[args.fold]["train"],
            folds[args.fold]["val"],
            args.fold,
            args.output_dir,
            device
        )

    else:

        all_results = []

        for fold in range(5):

            history = train_fold(
                folds[fold]["train"],
                folds[fold]["val"],
                fold,
                args.output_dir,
                device
            )

            all_results.append(
                history
            )

        with open(
            Path(args.output_dir)
            / "all_results.json",
            "w"
        ) as f:

            json.dump(
                all_results,
                f,
                indent=2
            )


if __name__ == "__main__":

    main()