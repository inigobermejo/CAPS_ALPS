#!/usr/bin/env python3

"""
Prepare pancreatic CT segmentation data from 3D Slicer .mrb files
and optionally train nnU-Net v2 with 5-fold cross-validation.

Labels:
    0 = background
    1 = pancreas
    2 = pancreatic lesion / tumor

Expected Slicer segment names:
    017_pancreas_whole
    028_pancreatic_lesion_tumor

Input:
    *.mrb files containing:
        Data/*_CT_veneus.nrrd
        Data/*_SEG.seg.nrrd

Output:
    nnUNet_raw/
        Dataset501_Pancreas/
            dataset.json
            imagesTr/
            labelsTr/

Requirements:
    pip install slicerio SimpleITK numpy

nnU-Net:
    pip install nnunetv2
"""

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np
import SimpleITK as sitk
import slicerio


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATASET_ID = 501
DATASET_NAME = "Pancreas"

PANCREAS_NAME = "017_pancreas_whole"
LESION_NAME = "028_pancreatic_lesion_tumor"

PANCREAS_LABEL = 1
LESION_LABEL = 2


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def extract_mrb_contents(mrb_file, temporary_dir):
    """
    Extract CT and segmentation files from an MRB archive.
    """

    with zipfile.ZipFile(mrb_file, "r") as z:

        members = z.namelist()

        ct_candidates = [
            x for x in members
            if x.lower().endswith(".nrrd")
            and "seg.nrrd" not in x.lower()
            and "_ct_" in x.lower()
        ]

        seg_candidates = [
            x for x in members
            if x.lower().endswith(".seg.nrrd")
        ]

        if not ct_candidates:
            raise RuntimeError(
                f"No CT NRRD found in {mrb_file.name}"
            )

        if not seg_candidates:
            raise RuntimeError(
                f"No segmentation found in {mrb_file.name}"
            )

        ct_member = ct_candidates[0]
        seg_member = seg_candidates[0]

        ct_path = Path(temporary_dir) / Path(ct_member).name
        seg_path = Path(temporary_dir) / Path(seg_member).name

        with z.open(ct_member) as src, open(ct_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        with z.open(seg_member) as src, open(seg_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    return ct_path, seg_path


def get_segment(segmentation, segment_name):
    """
    Find a Slicer segment by exact name.
    """

    matches = [
        s for s in segmentation["segments"]
        if s.get("name") == segment_name
    ]

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple segments named '{segment_name}'"
        )

    return matches[0]


def create_labelmap(segmentation):
    """
    Create a 3-class labelmap from the Slicer segmentation.

    0 = background
    1 = pancreas
    2 = lesion

    Handles segments stored on different Slicer labelmap layers.
    """

    pancreas = get_segment(segmentation, PANCREAS_NAME)
    lesion = get_segment(segmentation, LESION_NAME)

    if pancreas is None:
        raise RuntimeError(
            f"Missing segment: {PANCREAS_NAME}"
        )

    if lesion is None:
        print(
            f"WARNING: {LESION_NAME} not found. "
            "This case will contain pancreas only."
        )

    voxels = segmentation["voxels"]

    # slicerio normally returns:
    #
    #   3D array       -> one labelmap layer
    #   4D array       -> multiple labelmap layers
    #
    # Convert 3D to 4D for uniform handling.

    if voxels.ndim == 3:
        voxels = voxels[np.newaxis, ...]

    elif voxels.ndim != 4:
        raise RuntimeError(
            f"Unexpected segmentation voxel dimensions: "
            f"{voxels.shape}"
        )

    output = np.zeros(voxels.shape[1:], dtype=np.uint8)

    # -------------------------------------------------------------
    # Pancreas
    # -------------------------------------------------------------

    pancreas_layer = pancreas["layer"]
    pancreas_value = pancreas["labelValue"]

    pancreas_mask = (
        voxels[pancreas_layer] == pancreas_value
    )

    output[pancreas_mask] = PANCREAS_LABEL

    # -------------------------------------------------------------
    # Lesion
    # -------------------------------------------------------------

    if lesion is not None:

        lesion_layer = lesion["layer"]
        lesion_value = lesion["labelValue"]

        lesion_mask = (
            voxels[lesion_layer] == lesion_value
        )

        # Lesion should normally be inside pancreas.
        # Give lesion priority in case of overlap.
        output[lesion_mask] = LESION_LABEL

    return output


def segmentation_to_sitk(segmentation, label_array):
    """
    Convert a Slicer segmentation array to a SimpleITK image,
    preserving its physical geometry.

    slicerio uses an IJK -> LPS affine.
    """

    affine = np.asarray(
        segmentation["ijkToLPS"],
        dtype=float
    )

    # numpy array is [k, j, i]
    # SimpleITK expects [x, y, z] indexing internally,
    # and GetImageFromArray handles this conversion.

    image = sitk.GetImageFromArray(
        label_array.astype(np.uint8)
    )

    # Columns of the affine correspond to I, J, K directions.
    direction_matrix = affine[:3, :3]

    spacing = np.linalg.norm(
        direction_matrix,
        axis=0
    )

    direction = direction_matrix / spacing

    origin = affine[:3, 3]

    image.SetSpacing(tuple(spacing))
    image.SetDirection(
        tuple(direction.flatten())
    )
    image.SetOrigin(tuple(origin))

    return image


def resample_label_to_ct(label_image, ct_image):
    """
    Resample segmentation onto the CT grid using nearest-neighbor.
    """

    resampler = sitk.ResampleImageFilter()

    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor
    )

    resampler.SetDefaultPixelValue(0)

    resampler.SetTransform(
        sitk.Transform()
    )

    output = resampler.Execute(label_image)

    return output


def sanitize_case_id(mrb_file):
    """
    Generate a unique nnU-Net case identifier.

    Example:
        Scene.CAPS4480.v-validated.mrb
        ->
        CAPS4480
    """

    name = mrb_file.stem

    # Remove Scene. prefix
    if name.startswith("Scene."):
        name = name[len("Scene."):]

    # Remove common validation suffixes
    for suffix in [
        ".v-validated",
        ".v-FINAL-v-validated",
        ".v-FINAL",
        "-validated",
    ]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Replace characters that could be inconvenient
    name = (
        name.replace(".", "_")
            .replace("-", "_")
            .replace(" ", "_")
    )

    return name


# -------------------------------------------------------------------------
# Process one MRB
# -------------------------------------------------------------------------

def process_case(mrb_file, images_dir, labels_dir):
    """

    Process one MRB and create:

        case_0001_0000.nii.gz
        case_0001.nii.gz
    """

    case_id = sanitize_case_id(mrb_file)

    print()
    print("=" * 80)
    print(f"Processing: {mrb_file.name}")
    print(f"Case ID:    {case_id}")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp:

        # -------------------------------------------------------------
        # Extract data
        # -------------------------------------------------------------

        ct_path, seg_path = extract_mrb_contents(
            mrb_file,
            tmp
        )

        print(f"CT:          {ct_path.name}")
        print(f"Segmentation: {seg_path.name}")

        # -------------------------------------------------------------
        # Read CT
        # -------------------------------------------------------------

        ct = sitk.ReadImage(str(ct_path))

        print(
            "CT size:     ",
            ct.GetSize()
        )

        print(
            "CT spacing:  ",
            ct.GetSpacing()
        )

        # -------------------------------------------------------------
        # Read Slicer segmentation
        # -------------------------------------------------------------

        segmentation = slicerio.read_segmentation(
            str(seg_path)
        )

        print(
            "Number of segments:",
            len(segmentation["segments"])
        )

        # -------------------------------------------------------------
        # Find required segments
        # -------------------------------------------------------------

        pancreas = get_segment(
            segmentation,
            PANCREAS_NAME
        )

        lesion = get_segment(
            segmentation,
            LESION_NAME
        )

        if pancreas is None:
            raise RuntimeError(
                f"{PANCREAS_NAME} missing"
            )

        print(
            f"Pancreas: layer={pancreas['layer']} "
            f"labelValue={pancreas['labelValue']} "
            f"extent={pancreas['extent']}"
        )

        if lesion is not None:
            print(
                f"Lesion:   layer={lesion['layer']} "
                f"labelValue={lesion['labelValue']} "
                f"extent={lesion['extent']}"
            )
        else:
            print("Lesion:   ABSENT")

        # -------------------------------------------------------------
        # Make labelmap
        # -------------------------------------------------------------

        label_array = create_labelmap(
            segmentation
        )

        print(
            "Segmentation array shape:",
            label_array.shape
        )

        unique, counts = np.unique(
            label_array,
            return_counts=True
        )

        print("Labels:")
        for u, c in zip(unique, counts):
            print(f"    {u}: {c:,} voxels")

        # -------------------------------------------------------------
        # Convert to SimpleITK
        # -------------------------------------------------------------

        label_image = segmentation_to_sitk(
            segmentation,
            label_array
        )

        # -------------------------------------------------------------
        # Resample to CT geometry
        # -------------------------------------------------------------

        label_image_ct = resample_label_to_ct(
            label_image,
            ct
        )

        # -------------------------------------------------------------
        # Verify resulting labels
        # -------------------------------------------------------------

        label_array_ct = sitk.GetArrayFromImage(
            label_image_ct
        )

        unique, counts = np.unique(
            label_array_ct,
            return_counts=True
        )

        print("Labels after CT resampling:")

        for u, c in zip(unique, counts):
            print(f"    {u}: {c:,} voxels")

        # -------------------------------------------------------------
        # Verify geometry
        # -------------------------------------------------------------

        assert (
            label_image_ct.GetSize()
            == ct.GetSize()
        )

        assert (
            label_image_ct.GetSpacing()
            == ct.GetSpacing()
        )

        # -------------------------------------------------------------
        # Write files
        # -------------------------------------------------------------

        image_output = (
            images_dir /
            f"{case_id}_0000.nii.gz"
        )

        label_output = (
            labels_dir /
            f"{case_id}.nii.gz"
        )

        sitk.WriteImage(
            ct,
            str(image_output)
        )

        sitk.WriteImage(
            label_image_ct,
            str(label_output)
        )

        print()
        print("Written:")
        print(f"  Image: {image_output}")
        print(f"  Label: {label_output}")

    return case_id


# -------------------------------------------------------------------------
# Create dataset.json
# -------------------------------------------------------------------------

def create_dataset_json(dataset_dir, case_ids):

    dataset = {
        "channel_names": {
            "0": "CT"
        },

        "labels": {
            "background": 0,
            "pancreas": 1,
            "lesion": 2
        },

        "numTraining": len(case_ids),

        "file_ending": ".nii.gz"
    }

    output = dataset_dir / "dataset.json"

    with open(output, "w") as f:
        json.dump(
            dataset,
            f,
            indent=4
        )

    print()
    print(f"Created: {output}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mrb_dir",
        required=True,
        help="Directory containing .mrb files"
    )

    parser.add_argument(
        "--nnunet_raw",
        required=True,
        help="nnUNet_raw directory"
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare dataset; do not preprocess/train"
    )

    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU ID"
    )

    args = parser.parse_args()

    mrb_dir = Path(args.mrb_dir)
    nnunet_raw = Path(args.nnunet_raw)

    dataset_dir = (
        nnunet_raw /
        f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    )

    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    images_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    labels_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mrb_files = sorted(
        mrb_dir.glob("*.mrb")
    )

    if len(mrb_files) == 0:
        raise RuntimeError(
            f"No .mrb files found in {mrb_dir}"
        )

    print()
    print("=" * 80)
    print("PANCREATIC SEGMENTATION DATASET")
    print("=" * 80)
    print(f"MRB files:  {len(mrb_files)}")
    print(f"Dataset:    Dataset{DATASET_ID:03d}_{DATASET_NAME}")
    print("=" * 80)

    case_ids = []

    # -------------------------------------------------------------
    # Process cases
    # -------------------------------------------------------------

    for mrb_file in mrb_files:

        try:

            case_id = process_case(
                mrb_file,
                images_dir,
                labels_dir
            )

            case_ids.append(case_id)

        except Exception as e:

            print()
            print("ERROR")
            print(f"File: {mrb_file}")
            print(f"Reason: {e}")
            print()

            raise

    # -------------------------------------------------------------
    # Create dataset.json
    # -------------------------------------------------------------

    create_dataset_json(
        dataset_dir,
        case_ids
    )

    # -------------------------------------------------------------
    # Preparation only?
    # -------------------------------------------------------------

    if args.prepare_only:

        print()
        print("=" * 80)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 80)

        print()
        print("Next step:")
        print()

        print(
            f"nnUNetv2_plan_and_preprocess "
            f"-d {DATASET_ID} "
            f"--verify_dataset_integrity"
        )

        return

    # -------------------------------------------------------------
    # nnU-Net preprocessing
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING NNUNET PREPROCESSING")
    print("=" * 80)

    subprocess.run(
        [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(DATASET_ID),
            "--verify_dataset_integrity"
        ],
        check=True
    )

    # -------------------------------------------------------------
    # Train folds 0-4
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING 5-FOLD CROSS VALIDATION")
    print("=" * 80)

    for fold in range(5):

        print()
        print("=" * 80)
        print(f"TRAINING FOLD {fold}")
        print("=" * 80)

        command = [
            "nnUNetv2_train",
            str(DATASET_ID),
            "3d_fullres",
            str(fold),
            "--npz",
            "-device",
            "cuda"
        ]

        subprocess.run(
            command,
            check=True
        )

    print()
    print("=" * 80)
    print("5-FOLD TRAINING COMPLETE")
    print("=" * 80)

    print()
    print(
        "nnU-Net has trained folds 0-4. "
        "The five models can subsequently be ensembled."
    )


if __name__ == "__main__":
    main()