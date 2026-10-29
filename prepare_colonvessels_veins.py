#!/usr/bin/env python3
"""Prepare ColonVessels venous-phase CTs for multiclass nnU-Net v2.

Uses slicerio metadata (segment name -> layer/label value and IJK-to-LPS
geometry), pynrrd voxels, and SimpleITK nearest-neighbor resampling.

Output labels:
  0 background
  1 portal vein
  2 superior mesenteric vein (SMV)
  3 splenic vein
  4 inferior mesenteric vein (IMV)

Default crop: axial (superior-inferior) only, around union of annotated
veins, targeting at least 440 mm coverage with 50 mm margins on each side.
This *source pretraining* crop uses GT vessel masks; it is not suitable
for standalone held-out evaluation without a similarly available ROI.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import nrrd
import SimpleITK as sitk
import slicerio


INPUT_DIR = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\ColonVessel\data")
OUTPUT_DIR = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\nnUNet\nnUNet_raw")

DATASET_ID = 6
DATASET_NAME = "ColonVessels"
CROP = "axial" 
MIN_AXIAL_MM = 440 
AXIAL_MARGIN_MM = 50
MISSING_POLICY= "skip" # choices=["skip", "error", "background"]

# Order determines tie-breaking at overlaps: portal vein has final priority.
TARGET_SEGMENTS = {
    "IMV": (4, [
        "infeirormesentericvein(v.mesentericainferior)",
        "inferiormesentericvein(v.mesentericainferior)",
        "v.mesentericainferior(inferiormesentericvein)",
        "inferiormesenterricvein(v.mesentericainferior)",
        "inferiormesetnericvein(v.mesentericainferior)",
        "inferiromesentericvein(v.mesentericainferior)"
    ]),
    "splenic_vein": (3, [
        "splenicvein(v.lienalis)",
    ]),
    "SMV": (2, [
        "superiormeentericvein(v.mesentericasuperior)",
        "superiormesentericartery(v.mesentericasuperior)",
        "superiormesentericvein(v.mesentericasuperior)",
        "superiormesentericveins(v.mesentericasuperior)",
        "inferiormesentericvein(v.mesentericasuperior)",
        "v.mesentericasuperior",
    ]),
    "portal_vein": (1, [
        "portalvein(v.portae)",
        "portalveins(v.portae)",
        "venaportae(v.portae)"
    ]),
}

LABELS = {
    "background": 0,
    "portal_vein": 1,
    "SMV": 2,
    "splenic_vein": 3,
    "IMV": 4,
}


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).casefold()


def read_multiclass_segmentation(path: Path):
    info = slicerio.read_segmentation(str(path), skip_voxels=True)
    raw, header = nrrd.read(str(path), index_order="F")

    if raw.ndim == 3:
        spatial_shape = raw.shape
        layer_axis = None
    elif raw.ndim == 4:
        kinds = [str(k).lower() for k in header.get("kinds", [])]
        candidates = [i for i, k in enumerate(kinds) if k in {"list", "vector"}]
        if len(candidates) != 1:
            raise ValueError(f"Unrecognized layer axis, kinds={kinds}")
        layer_axis = candidates[0]
        spatial_shape = tuple(dim for i, dim in enumerate(raw.shape) if i != layer_axis)
    else:
        raise ValueError(f"Unsupported segmentation dimensions: {raw.shape}")

    indexed = {}
    for segment in info["segments"]:
        key = normalize_name(segment["name"])
        indexed[key] = segment

    merged = np.zeros(spatial_shape, dtype=np.uint8)
    missing = []
    empty = []
    overlaps = 0
    original_counts = {}

    for short_name, (label_value, variants) in TARGET_SEGMENTS.items():
        segment = next(
            (
                indexed[normalize_name(name)]
                for name in variants
                if normalize_name(name) in indexed
            ),
            None
        )
        if segment is None:
            missing.append(short_name)
            continue

        layer = int(segment.get("layer", 0))
        value = int(segment["labelValue"])
        voxels = raw if layer_axis is None else np.take(raw, layer, axis=layer_axis)
        mask = (voxels == value)
        original_counts[short_name] = int(np.count_nonzero(mask))
        if not original_counts[short_name]:
            empty.append(short_name)
            continue
        overlaps += int(np.count_nonzero(mask & (merged != 0)))
        merged[mask] = label_value

    # Use SlicerIO's IJK->LPS matrix; it already handles RAS/LPS conversion.
    affine = np.asarray(info["ijkToLPS"], dtype=np.float64)
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Invalid segmentation voxel spacing")
    directions = basis / spacing[None, :]

    seg_img = sitk.GetImageFromArray(np.transpose(merged, (2, 1, 0)))
    seg_img.SetSpacing(tuple(map(float, spacing)))
    seg_img.SetOrigin(tuple(map(float, affine[:3, 3])))
    seg_img.SetDirection(tuple(map(float, directions.ravel())))
    return seg_img, missing, empty, overlaps, original_counts


def crop_axial(ct: sitk.Image, label: sitk.Image,
               minimum_mm: float, margin_mm: float):
    """Crop only CT image axis 2, assumed to be superior-inferior."""
    direction = np.asarray(ct.GetDirection(), dtype=float).reshape((3, 3))
    if abs(direction[2, 2]) < 0.90:
        raise ValueError(
            "CT image axis 2 is not approximately superior-inferior; "
            "refusing automatic axial crop"
        )

    arr = sitk.GetArrayViewFromImage(label)  # Z,Y,X
    where_z = np.flatnonzero(np.any(arr != 0, axis=(1, 2)))
    if not where_z.size:
        raise ValueError("No selected vessel voxels found in CT geometry")

    lo, hi = int(where_z[0]), int(where_z[-1] + 1)
    spacing_z = float(ct.GetSpacing()[2])
    margin = int(math.ceil(margin_mm / spacing_z))
    want = max(int(math.ceil(minimum_mm / spacing_z)), hi - lo + 2 * margin)
    depth = int(ct.GetSize()[2])
    size_z = min(want, depth)

    # Centre on labelled vessels; clamp to CT boundary.
    mid = (lo + hi) / 2.0
    start_z = max(0, min(int(round(mid - size_z / 2.0)), depth - size_z))

    # Guarantee that all labels, and requested margins when possible, survive.
    if start_z > lo or start_z + size_z < hi:
        raise RuntimeError("Computed ROI would truncate a vessel")

    index = [0, 0, int(start_z)]
    size = [int(ct.GetSize()[0]), int(ct.GetSize()[1]), int(size_z)]
    ct_crop = sitk.RegionOfInterest(ct, size, index)
    label_crop = sitk.RegionOfInterest(label, size, index)
    return ct_crop, label_crop, start_z, size_z


def process_case(folder: Path, images_dir: Path, labels_dir: Path):
    name = folder.name
    ct_path = folder / f"{name}_Venous_Phase_CT.nrrd"
    seg_path = folder / f"{name}_Venous_Phase_Veins.seg.nrrd"
    if not ct_path.is_file() or not seg_path.is_file():
        raise FileNotFoundError(f"Missing CT or vein segmentation for {name}")

    ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    if ct.GetDimension() != 3:
        raise ValueError(f"CT is not 3D: {ct.GetDimension()}D")

    segmentation, missing, empty, overlaps, raw_counts = read_multiclass_segmentation(seg_path)
    missing_or_empty = missing + empty
    if missing_or_empty:
        if MISSING_POLICY == "error":
            raise ValueError(f"Missing/empty required annotations: {missing_or_empty}")
        if MISSING_POLICY == "skip":
            return {"patient": name, "status": "skipped_missing", "missing": ";".join(missing),
                    "empty": ";".join(empty), "overlap_voxels": overlaps}
        # background policy: only for cases verified truly negative/absent.

    # Nearest-neighbour: no fractional class IDs; preserve original CT geometry.
    label = sitk.Resample(
        segmentation, ct, sitk.Transform(), sitk.sitkNearestNeighbor,
        0, sitk.sitkUInt8
    )
    counts_before = np.bincount(sitk.GetArrayViewFromImage(label).ravel(), minlength=5)
    for short_name, count in raw_counts.items():
        target_id = LABELS[short_name]
        if count > 0 and counts_before[target_id] == 0:
            raise ValueError(
                f"{short_name}: nonempty in original NRRD but empty after CT resampling. "
                "Check geometry/registration."
            )

    original_depth = ct.GetSize()[2]
    start_z = 0
    size_z = original_depth
    if CROP == "axial":
        ct, label, start_z, size_z = crop_axial(
            ct, label, MIN_AXIAL_MM, AXIAL_MARGIN_MM
        )

    counts_after = np.bincount(sitk.GetArrayViewFromImage(label).ravel(), minlength=5)
    if not np.array_equal(counts_before[1:], counts_after[1:]):
        raise RuntimeError(f"{name}: crop removed labelled vessel voxels")

    # Safe to write only after validating all the selected labels and geometry.
    sitk.WriteImage(ct, str(images_dir / f"{name}_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(label, str(labels_dir / f"{name}.nii.gz"), useCompression=True)

    record = {
        "patient": name, "status": "written", "missing": ";".join(missing),
        "empty": ";".join(empty), "overlap_voxels": overlaps,
        "ct_original_slices": original_depth,
        "ct_saved_slices": size_z,
        "ct_start_slice": start_z,
        "axial_extent_mm": round(size_z * ct.GetSpacing()[2], 1),
    }
    for value, structure in enumerate(["background", "portal_vein", "SMV", "splenic_vein", "IMV"]):
        if value != 0:
            record[f"voxels_{structure}"] = int(counts_after[value])
    return record


def main():

    if MIN_AXIAL_MM <= 0 or AXIAL_MARGIN_MM < 0:
        Exception("Crop dimensions must be positive (margin nonnegative)")
    dataset_dir = OUTPUT_DIR / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted(p for p in INPUT_DIR.iterdir()
                     if p.is_dir() and re.fullmatch(r"pat_\d{3}", p.name))
    if not folders:
        Exception(f"No pat_XXX folders found in {INPUT_DIR}")

    # Fail before starting if the destination already has data; avoid stale cases
    # and inconsistent dataset.json when rerunning with different options.
    existing = list(images_dir.glob("*.nii.gz")) + list(labels_dir.glob("*.nii.gz"))
    if existing:
        Exception(f"Output already contains {len(existing)} NIfTI files: {dataset_dir}. "
                 "Use a fresh Dataset ID or manually clear the old output.")

    records = []
    print(f"Found {len(folders)} candidate patient folders")
    for i, folder in enumerate(folders, 1):
        print(f"[{i:02d}/{len(folders):02d}] {folder.name}...", flush=True)
        try:
            result = process_case(folder, images_dir, labels_dir)
        except Exception as exc:
            result = {"patient": folder.name, "status": "ERROR", "error": str(exc)}
        records.append(result)
        detail = result.get("missing") or result.get("empty") or result.get("error", "")
        print(f"  {result['status']} {detail}", flush=True)

    fields = ["patient", "status", "error", "missing", "empty", "overlap_voxels",
              "ct_original_slices", "ct_saved_slices", "ct_start_slice", "axial_extent_mm",
              "voxels_portal_vein", "voxels_SMV", "voxels_splenic_vein", "voxels_IMV"]
    with (dataset_dir / "conversion_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    successful = [r for r in records if r["status"] == "written"]
    if successful:
        dataset = {
            "channel_names": {"0": "CT"},
            "labels": LABELS,
            "numTraining": len(successful),
            "file_ending": ".nii.gz",
        }
        (dataset_dir / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")

    print(f"\nSaved {len(successful)} / {len(folders)} scans to {dataset_dir}")
    print(f"Conversion report: {dataset_dir / 'conversion_report.csv'}")
    if any(r["status"] == "ERROR" for r in records):
        sys.exit("Some scans failed: review conversion_report.csv before training")
    if not successful:
        sys.exit("No cases written; check missing-policy and segment names")
    print("Run nnUNetv2_plan_and_preprocess after auditing the labels and report.")


if __name__ == "__main__":
    main()
