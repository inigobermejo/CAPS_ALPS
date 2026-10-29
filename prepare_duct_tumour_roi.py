#!/usr/bin/env python3

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import nrrd
import SimpleITK as sitk
import slicerio


DUCT_NAME = "021_pancreatic_duct"
TUMOUR_NAME = "028_pancreatic_lesion_tumor"

DATASET_ID = 4

MRB_DIR = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\Segmented")
STAGE2_DIR = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\TwoStage\stage2_tumour")
OUTPUT_DIR = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\nnUNet\nnUNet_raw")

def case_id_from_mrb(path):
    name = path.stem
    if name.startswith("Scene."):
        name = name[len("Scene."):]
    return name


def read_segment_mask(seg_file, segment_name):
    """Read one named Slicer segment, retaining its physical geometry."""

    metadata = slicerio.read_segmentation(str(seg_file))
    segments = {
        s["name"]: s for s in metadata["segments"]
    }

    if segment_name not in segments:
        raise ValueError(f"Segment not found: {segment_name}")

    info = segments[segment_name]
    raw, header = nrrd.read(str(seg_file))

    label_value = int(info["labelValue"])
    layer = int(info.get("layer", 0))

    if raw.ndim == 4:
        kinds = [str(k).lower() for k in header["kinds"]]
        layer_axes = [
            i for i, k in enumerate(kinds)
            if k in ("list", "vector")
        ]
        if len(layer_axes) != 1:
            raise ValueError(
                f"Cannot identify layer axis: {kinds}"
            )
        layer_axis = layer_axes[0]
        raw = np.take(raw, layer, axis=layer_axis)

        directions = [
            d for i, d in enumerate(header["space directions"])
            if i != layer_axis
        ]
    elif raw.ndim == 3:
        directions = header["space directions"]
    else:
        raise ValueError(f"Unexpected NRRD shape: {raw.shape}")

    mask = (raw == label_value).astype(np.uint8)

    # pynrrd uses X,Y,Z indexing; SimpleITK arrays use Z,Y,X.
    image = sitk.GetImageFromArray(mask.transpose(2, 1, 0))

    vectors = np.asarray(directions, dtype=float)
    spacing = np.linalg.norm(vectors, axis=1)

    if np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError("Invalid NRRD spatial directions")

    direction_matrix = np.column_stack(
        [vectors[i] / spacing[i] for i in range(3)]
    )

    origin = np.asarray(header["space origin"], dtype=float)

    # SimpleITK expects physical coordinates in LPS.
    space = str(header.get("space", "")).lower()
    if space in ("right-anterior-superior", "ras"):
        conversion = np.diag([-1.0, -1.0, 1.0])
        direction_matrix = conversion @ direction_matrix
        origin = conversion @ origin
    elif space not in ("left-posterior-superior", "lps"):
        raise ValueError(f"Unsupported NRRD space: {space}")

    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetOrigin(tuple(float(v) for v in origin))
    image.SetDirection(
        tuple(float(v) for v in direction_matrix.ravel())
    )

    return image


def find_segmentation(mrb_file, extract_dir):
    with zipfile.ZipFile(mrb_file) as archive:
        archive.extractall(extract_dir)

    files = list(Path(extract_dir).rglob("*.seg.nrrd"))
    if len(files) != 1:
        raise RuntimeError(
            f"{mrb_file.name}: expected one .seg.nrrd, "
            f"found {len(files)}"
        )
    return files[0]


def process_case(case_id, mrb_file, roi_image, tumour_label, out_dir):
    roi = sitk.ReadImage(str(roi_image))
    tumour = sitk.ReadImage(str(tumour_label))

    # Existing stage-two labels must have the same geometry as ROI.
    if (
        roi.GetSize() != tumour.GetSize()
        or not np.allclose(roi.GetSpacing(), tumour.GetSpacing())
        or not np.allclose(roi.GetOrigin(), tumour.GetOrigin())
        or not np.allclose(roi.GetDirection(), tumour.GetDirection())
    ):
        raise ValueError(f"{case_id}: ROI/label geometry mismatch")

    tumour_np = sitk.GetArrayFromImage(tumour)

    if not np.isin(tumour_np, [0, 1]).all():
        raise ValueError(
            f"{case_id}: expected existing tumour-only "
            f"labels 0/1, found {np.unique(tumour_np)}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        seg_file = find_segmentation(mrb_file, tmp)
        duct_img = read_segment_mask(seg_file, DUCT_NAME)

        duct_roi = sitk.Resample(
            duct_img,
            roi,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )

        duct_np = sitk.GetArrayFromImage(duct_roi) > 0

    tumour_np = tumour_np == 1

    # Tumour takes priority if original annotations overlap.
    overlap = int(np.count_nonzero(duct_np & tumour_np))

    combined = np.zeros(tumour_np.shape, dtype=np.uint8)
    combined[duct_np] = 1
    combined[tumour_np] = 2

    combined_img = sitk.GetImageFromArray(combined)
    combined_img.CopyInformation(roi)

    shutil.copy2(
        roi_image,
        out_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
    )
    sitk.WriteImage(
        combined_img,
        str(out_dir / "labelsTr" / f"{case_id}.nii.gz"),
        useCompression=True,
    )

    counts = np.bincount(combined.ravel(), minlength=3)
    print(
        f"{case_id}: duct={counts[1]:,}, "
        f"tumour={counts[2]:,}, overlap={overlap:,}"
    )

    return overlap, int(counts[1]), int(counts[2])


def main():
   
    source = Path(STAGE2_DIR)
    destination = (
        Path(OUTPUT_DIR)
        / f"Dataset{DATASET_ID:03d}_DuctTumourROI"
    )

    (destination / "imagesTr").mkdir(parents=True, exist_ok=True)
    (destination / "labelsTr").mkdir(parents=True, exist_ok=True)

    mrb_files = sorted(
        p for p in Path(MRB_DIR).iterdir()
        if p.suffix.lower() == ".mrb"
    )
    mrbs = {case_id_from_mrb(p): p for p in mrb_files}

    roi_images = sorted((source / "imagesTr").glob("*_0000.nii.gz"))

    if not roi_images:
        raise RuntimeError("No cropped images found")

    results = []

    for roi_image in roi_images:
        case_id = roi_image.name.removesuffix("_0000.nii.gz")
        tumour_label = source / "labelsTr" / f"{case_id}.nii.gz"

        if case_id not in mrbs:
            raise FileNotFoundError(
                f"No MRB matching '{case_id}'. "
                "Check the case naming convention."
            )

        if not tumour_label.is_file():
            raise FileNotFoundError(tumour_label)

        overlap, n_duct, n_tumour = process_case(
            case_id,
            mrbs[case_id],
            roi_image,
            tumour_label,
            destination,
        )
        results.append((case_id, overlap, n_duct, n_tumour))

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "pancreatic_duct": 1,
            "pancreatic_lesion_tumor": 2,
        },
        "numTraining": len(results),
        "file_ending": ".nii.gz",
    }

    (destination / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2)
    )

    # Preserve the exact cross-validation definition if supplied.
    folds_file = source / "folds.json"
    if folds_file.is_file():
        shutil.copy2(folds_file, destination / "folds.json")

    print(f"\nDataset written to {destination}")
    print(f"Processed {len(results)} scans")
    print(
        f"Total tumour/duct overlap: "
        f"{sum(r[1] for r in results):,} voxels"
    )
    print(
        f"Scans with no duct voxels in ROI: "
        f"{sum(r[2] == 0 for r in results)}"
    )
    print(
        f"Scans with no tumour: "
        f"{sum(r[3] == 0 for r in results)}"
    )


if __name__ == "__main__":
    main()