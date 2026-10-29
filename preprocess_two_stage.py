from pathlib import Path
import os
import json
import shutil

import numpy as np
import SimpleITK as sitk


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = Path(r"C:/Users/lucp13173/Documents/Data/CAPS/Jessa/Preprocessed")
OUTPUT_DIR = Path("C:/Users/lucp13173/Documents/Data/CAPS/Jessa/TwoStage")

MARGIN_MM = 25.0


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

STAGE1 = OUTPUT_DIR / "stage1_pancreas"
STAGE2 = OUTPUT_DIR / "stage2_tumour"

for root in [STAGE1, STAGE2]:
    (root / "imagesTr").mkdir(
        parents=True,
        exist_ok=True
    )

    (root / "labelsTr").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# HELPERS
# ============================================================

def bbox_from_mask(mask):
    """
    mask is NumPy Z,Y,X.
    Returns:
        z0,z1,y0,y1,x0,x1
    with upper bounds exclusive.
    """

    coords = np.argwhere(mask)

    if len(coords) == 0:
        raise RuntimeError(
            "Pancreas mask is empty."
        )

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    return z0, z1, y0, y1, x0, x1


def expand_bbox(
    bbox,
    spacing_xyz,
    shape_zyx,
    margin_mm
):
    """
    Expand bbox by a physical margin.
    """

    z0, z1, y0, y1, x0, x1 = bbox

    sx, sy, sz = spacing_xyz

    mz = int(np.ceil(margin_mm / sz))
    my = int(np.ceil(margin_mm / sy))
    mx = int(np.ceil(margin_mm / sx))

    z0 = max(0, z0 - mz)
    z1 = min(shape_zyx[0], z1 + mz)

    y0 = max(0, y0 - my)
    y1 = min(shape_zyx[1], y1 + my)

    x0 = max(0, x0 - mx)
    x1 = min(shape_zyx[2], x1 + mx)

    return z0, z1, y0, y1, x0, x1


def crop_sitk_image(image, bbox):
    """
    Crop a SimpleITK image while preserving the correct
    physical origin.
    """

    z0, z1, y0, y1, x0, x1 = bbox

    size = [
        int(x1 - x0),
        int(y1 - y0),
        int(z1 - z0)
    ]

    index = [
        int(x0),
        int(y0),
        int(z0)
    ]

    return sitk.RegionOfInterest(
        image,
        size=size,
        index=index
    )


# ============================================================
# PROCESS CASES
# ============================================================

label_files = sorted(
    (INPUT_DIR / "labelsTr").glob(
        "*.nii.gz"
    )
)

for label_file in label_files:

    case_id = label_file.name.replace(
        ".nii.gz",
        ""
    )

    image_file = (
        INPUT_DIR
        / "imagesTr"
        / f"{case_id}_0000.nii.gz"
    )

    print(f"Processing {case_id}")

    image = sitk.ReadImage(
        str(image_file)
    )

    label = sitk.ReadImage(
        str(label_file)
    )

    label_np = sitk.GetArrayFromImage(
        label
    )

    # ========================================================
    # STAGE 1
    #
    # whole pancreas:
    # original pancreas + tumour
    # ========================================================

    whole_pancreas_np = (
        (label_np == 1)
        | (label_np == 2)
    ).astype(np.uint8)

    pancreas_label = sitk.GetImageFromArray(
        whole_pancreas_np
    )

    pancreas_label.CopyInformation(
        label
    )

    # CT can simply be copied
    shutil.copy2(
        image_file,
        STAGE1
        / "imagesTr"
        / f"{case_id}_0000.nii.gz"
    )

    sitk.WriteImage(
        pancreas_label,
        str(
            STAGE1
            / "labelsTr"
            / f"{case_id}.nii.gz"
        ),
        useCompression=True
    )

    # ========================================================
    # STAGE 2
    #
    # Crop around whole pancreas
    # ========================================================

    bbox = bbox_from_mask(
        whole_pancreas_np
    )

    bbox = expand_bbox(
        bbox=bbox,
        spacing_xyz=image.GetSpacing(),
        shape_zyx=whole_pancreas_np.shape,
        margin_mm=MARGIN_MM
    )

    cropped_image = crop_sitk_image(
        image,
        bbox
    )

    # Binary tumour target
    tumour_np = (
        label_np == 2
    ).astype(np.uint8)

    tumour_label = sitk.GetImageFromArray(
        tumour_np
    )

    tumour_label.CopyInformation(
        label
    )

    cropped_tumour = crop_sitk_image(
        tumour_label,
        bbox
    )

    sitk.WriteImage(
        cropped_image,
        str(
            STAGE2
            / "imagesTr"
            / f"{case_id}_0000.nii.gz"
        ),
        useCompression=True
    )

    sitk.WriteImage(
        cropped_tumour,
        str(
            STAGE2
            / "labelsTr"
            / f"{case_id}.nii.gz"
        ),
        useCompression=True
    )

    print(
        f"  crop size: "
        f"{cropped_image.GetSize()}"
    )

    print(
        f"  tumour voxels: "
        f"{np.sum(tumour_np):,}"
    )


# ============================================================
# COPY IDENTICAL CV SPLITS
# ============================================================

for stage in [STAGE1, STAGE2]:

    shutil.copy2(
        INPUT_DIR / "folds.json",
        stage / "folds.json"
    )


# ============================================================
# nnU-Net dataset.json FILES
# ============================================================

stage1_json = {
    "channel_names": {
        "0": "CT"
    },
    "labels": {
        "background": 0,
        "pancreas": 1
    },
    "numTraining": len(label_files),
    "file_ending": ".nii.gz"
}


stage2_json = {
    "channel_names": {
        "0": "CT"
    },
    "labels": {
        "background": 0,
        "tumor": 1
    },
    "numTraining": len(label_files),
    "file_ending": ".nii.gz"
}


with open(
    STAGE1 / "dataset.json",
    "w"
) as f:
    json.dump(
        stage1_json,
        f,
        indent=2
    )


with open(
    STAGE2 / "dataset.json",
    "w"
) as f:
    json.dump(
        stage2_json,
        f,
        indent=2
    )


print()
print("Done.")
print(f"Stage 1: {STAGE1}")
print(f"Stage 2: {STAGE2}")