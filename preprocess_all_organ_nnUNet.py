from pathlib import Path
import os
import json
import tempfile
import zipfile

import numpy as np
import nrrd
import SimpleITK as sitk
import slicerio


# ============================================================
# SETTINGS
# ============================================================

MRB_DIR = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\Segmented")

NNUNET_RAW = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\nnUNet\nnUNet_raw")

DATASET_ID = 3
DATASET_NAME = "JessaMultiOrgan"

OUTPUT_DIR = (
    NNUNET_RAW
    / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
)

IMAGES_OUT = OUTPUT_DIR / "imagesTr"
LABELS_OUT = OUTPUT_DIR / "labelsTr"

IMAGES_OUT.mkdir(parents=True, exist_ok=True)
LABELS_OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET SEGMENTS
# ============================================================

SEGMENTS = {
    "003_aorta": 1,
    "005_celiac_artery": 2,
    "007_common_bile_duct": 3,
    "008_duodenum": 4,
    "017_pancreas_whole": 5,
    "021_pancreatic_duct": 6,
    "022_postcava_IVC": 7,
    "024_spleen": 8,
    "026_SMA": 9,
    "028_pancreatic_lesion_tumor": 10,
    "035_portal_vein": 11,
    "036_SMV": 12,
    "037_splenic_vein": 13,
    "038_IMV": 14,
}


# ============================================================
# HELPERS
# ============================================================

def find_segmentation_file(root):
    files = [
        p for p in Path(root).rglob("*")
        if p.is_file()
        and p.name.lower().endswith(".seg.nrrd")
    ]

    if len(files) != 1:
        raise RuntimeError(
            f"Expected 1 segmentation, found {len(files)}"
        )

    return files[0]


def find_ct_file(root):
    files = [
        p for p in Path(root).rglob("*")
        if p.is_file()
        and p.name.lower().endswith(".nrrd")
        and not p.name.lower().endswith(".seg.nrrd")
    ]

    if len(files) != 1:
        raise RuntimeError(
            f"Expected 1 CT NRRD, found {len(files)}"
        )

    return files[0]


def case_id_from_mrb(mrb):
    name = mrb.stem

    if name.startswith("Scene."):
        name = name[6:]

    return name


def get_layer_axis(raw, header):
    """
    Find the segmentation layer axis for 4D .seg.nrrd files.
    """

    if raw.ndim == 3:
        return None

    if raw.ndim != 4:
        raise RuntimeError(
            f"Unexpected segmentation shape: {raw.shape}"
        )

    kinds = header.get("kinds", None)

    if kinds is not None:
        for i, kind in enumerate(kinds):
            if str(kind).lower() in {"list", "vector"}:
                return i

    # Typical Slicer layout
    return 0


def mask_to_sitk(mask, header):
    """
    Convert raw NRRD X,Y,Z mask to SimpleITK image,
    preserving physical geometry.
    """

    dirs = np.asarray(
        header["space directions"],
        dtype=float
    )

    spatial_dirs = [
        d for d in dirs
        if np.all(np.isfinite(d))
    ][:3]

    spatial_dirs = np.asarray(spatial_dirs)

    spacing = np.linalg.norm(
        spatial_dirs,
        axis=1
    )

    direction = np.column_stack([
        spatial_dirs[i] / spacing[i]
        for i in range(3)
    ])

    origin = np.asarray(
        header["space origin"],
        dtype=float
    )

    # NRRD numpy indexing: X,Y,Z
    # SimpleITK array indexing: Z,Y,X
    mask_zyx = np.transpose(
        mask,
        (2, 1, 0)
    )

    img = sitk.GetImageFromArray(
        mask_zyx.astype(np.uint8)
    )

    img.SetSpacing(
        tuple(float(x) for x in spacing)
    )

    img.SetOrigin(
        tuple(float(x) for x in origin)
    )

    img.SetDirection(
        tuple(
            float(x)
            for x in direction.flatten()
        )
    )

    return img


def extract_segment_mask(
    raw,
    layer_axis,
    segment_info
):
    """
    Extract one binary segment mask.
    """

    label_value = int(
        segment_info["labelValue"]
    )

    layer = int(
        segment_info.get("layer", 0)
    )

    if raw.ndim == 3:
        layer_data = raw

    else:
        layer_data = np.take(
            raw,
            layer,
            axis=layer_axis
        )

    return (
        layer_data == label_value
    ).astype(np.uint8)


def resample_mask_to_ct(mask_img, ct_img):
    return sitk.Resample(
        mask_img,
        ct_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_mrb(mrb):

    case_id = case_id_from_mrb(mrb)

    print()
    print("=" * 70)
    print(f"Processing {case_id}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:

        # ----------------------------------------------------
        # Extract MRB
        # ----------------------------------------------------

        with zipfile.ZipFile(mrb, "r") as z:
            z.extractall(tmp)

        tmp = Path(tmp)

        seg_file = find_segmentation_file(tmp)
        ct_file = find_ct_file(tmp)

        # ----------------------------------------------------
        # Read CT
        # ----------------------------------------------------

        ct = sitk.ReadImage(
            str(ct_file),
            sitk.sitkFloat32
        )

        # ----------------------------------------------------
        # Read segmentation metadata + voxels
        # ----------------------------------------------------

        seg_meta = slicerio.read_segmentation(
            str(seg_file)
        )

        segments_by_name = {
            s["name"]: s
            for s in seg_meta["segments"]
        }

        raw, header = nrrd.read(
            str(seg_file)
        )

        layer_axis = get_layer_axis(
            raw,
            header
        )

        # ----------------------------------------------------
        # Final nnU-Net labelmap
        # ----------------------------------------------------

        shape_zyx = tuple(
            reversed(
                ct.GetSize()
            )
        )

        final_labels = np.zeros(
            shape_zyx,
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # Extract each target segment
        # ----------------------------------------------------

        for segment_name, target_label in SEGMENTS.items():

            if segment_name not in segments_by_name:
                print(
                    f"  MISSING: {segment_name}"
                )
                continue

            segment_info = segments_by_name[
                segment_name
            ]

            raw_mask = extract_segment_mask(
                raw,
                layer_axis,
                segment_info
            )

            n_raw = int(raw_mask.sum())

            if n_raw == 0:
                print(
                    f"  EMPTY:   {segment_name}"
                )
                continue

            mask_img = mask_to_sitk(
                raw_mask,
                header
            )

            mask_img = resample_mask_to_ct(
                mask_img,
                ct
            )

            mask_np = sitk.GetArrayFromImage(
                mask_img
            )

            n_resampled = int(
                np.count_nonzero(mask_np)
            )

            if n_resampled == 0:
                print(
                    f"  WARNING: {segment_name} "
                    f"became empty after resampling"
                )
                continue

            # Later labels overwrite earlier labels
            # if there is overlap.
            final_labels[
                mask_np > 0
            ] = target_label

            print(
                f"  {target_label:2d} "
                f"{segment_name:35s} "
                f"{n_resampled:,} voxels"
            )

        # ----------------------------------------------------
        # Convert final labelmap to SITK
        # ----------------------------------------------------

        label_img = sitk.GetImageFromArray(
            final_labels
        )

        label_img.CopyInformation(
            ct
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        image_out = (
            IMAGES_OUT
            / f"{case_id}_0000.nii.gz"
        )

        label_out = (
            LABELS_OUT
            / f"{case_id}.nii.gz"
        )

        sitk.WriteImage(
            ct,
            str(image_out),
            useCompression=True
        )

        sitk.WriteImage(
            label_img,
            str(label_out),
            useCompression=True
        )

        print(
            f"Saved {case_id}"
        )


# ============================================================
# MAIN
# ============================================================

mrb_files = sorted(
    list(MRB_DIR.glob("*.mrb"))
    + list(MRB_DIR.glob("*.MRB"))
)

if not mrb_files:
    raise RuntimeError(
        f"No MRB files found in {MRB_DIR}"
    )

print(
    f"Found {len(mrb_files)} MRB files"
)

for mrb in mrb_files:
    process_mrb(mrb)


# ============================================================
# dataset.json
# ============================================================

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

    "numTraining": len(mrb_files),

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


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Dataset created at:\n{OUTPUT_DIR}"
)