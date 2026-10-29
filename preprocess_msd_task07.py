from pathlib import Path
import os
import json
import shutil

import numpy as np
import SimpleITK as sitk


# ============================================================
# SETTINGS
# ============================================================

# Original extracted MSD Task07 folder
MSD_DIR = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\MSD\Task07_Pancreas")

# nnU-Net raw data location
NNUNET_RAW = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\MSD\nnUNet_raw")

DATASET_ID = 2
DATASET_NAME = "MSDPancreas"

OUTPUT_DIR = (
    NNUNET_RAW
    / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
)

IMAGES_OUT = OUTPUT_DIR / "imagesTr"
LABELS_OUT = OUTPUT_DIR / "labelsTr"

IMAGES_OUT.mkdir(parents=True, exist_ok=True)
LABELS_OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND CASES
# ============================================================

images = sorted(
    (MSD_DIR / "imagesTr").glob("*.nii.gz")
)

labels = sorted(
    (MSD_DIR / "labelsTr").glob("*.nii.gz")
)

if not images:
    raise RuntimeError(
        f"No training images found in {MSD_DIR / 'imagesTr'}"
    )

print(f"Found {len(images)} images")
print(f"Found {len(labels)} labels")


# ============================================================
# COPY TO nnU-Net FORMAT
# ============================================================

processed = []

for image_file in images:

    # pancreas_001.nii.gz -> pancreas_001
    case_id = image_file.name.replace(
        ".nii.gz", ""
    )

    label_file = (
        MSD_DIR
        / "labelsTr"
        / f"{case_id}.nii.gz"
    )

    if not label_file.exists():
        raise RuntimeError(
            f"Missing label for {case_id}"
        )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    label_img = sitk.ReadImage(
        str(label_file)
    )

    label_array = sitk.GetArrayFromImage(
        label_img
    )

    unique_labels = np.unique(
        label_array
    )

    if not set(unique_labels).issubset(
        {0, 1, 2}
    ):
        raise RuntimeError(
            f"{case_id}: unexpected labels "
            f"{unique_labels}"
        )

    # --------------------------------------------------------
    # nnU-Net naming
    # --------------------------------------------------------

    image_target = (
        IMAGES_OUT
        / f"{case_id}_0000.nii.gz"
    )

    label_target = (
        LABELS_OUT
        / f"{case_id}.nii.gz"
    )

    # IMPORTANT:
    # Copy original files unchanged.
    # No resampling, normalization or conversion.
    shutil.copy2(
        image_file,
        image_target
    )

    shutil.copy2(
        label_file,
        label_target
    )

    processed.append(case_id)

    print(
        f"{case_id}: labels={unique_labels.tolist()}"
    )


# ============================================================
# dataset.json
# ============================================================

dataset_json = {

    "channel_names": {
        "0": "CT"
    },

    "labels": {
        "background": 0,
        "pancreas": 1,
        "tumor": 2
    },

    "numTraining": len(processed),

    "file_ending": ".nii.gz"
}


with open(
    OUTPUT_DIR / "dataset.json",
    "w"
) as f:

    json.dump(
        dataset_json,
        f,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Cases:   {len(processed)}")
print(f"Output:  {OUTPUT_DIR}")

print()
print("Images:")
print(IMAGES_OUT)

print()
print("Labels:")
print(LABELS_OUT)