
from pathlib import Path
import csv
import json

import numpy as np

DATASET_FOLDER = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\nnUNet\nnUNet_preprocessed\Dataset004_PancreaticLesionCropped")
LABEL = 1
PLANS = "nnUNetResEncUNetLPlans"
CONFIG = "3d_fullres"

def read_segmentation(path):
    """Read a preprocessed nnU-Net segmentation."""

    if path.suffix == ".b2nd":
        import blosc2
        arr = blosc2.open(urlpath=str(path), mode="r")[:]

    elif path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")

    elif path.suffix == ".npz":
        with np.load(path) as f:
            arr = f["seg"]

    else:
        raise ValueError(f"Unsupported file: {path}")

    # nnU-Net segmentations normally have shape (1, Z, Y, X).
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Unexpected shape: {arr.shape}")
        arr = arr[0]

    return arr


def main():

     # Find the actual preprocessed data directory
    plans_path = DATASET_FOLDER / f"{PLANS}.json"

    with open(plans_path) as f:
        plans = json.load(f)

    data_identifier = (
        plans["configurations"][CONFIG]["data_identifier"]
    )

    data_dir = DATASET_FOLDER / data_identifier

    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    print(f"Reading: {data_dir}")
    print(f"Tumour label: {LABEL}\n")

    # Find segmentations, prioritizing the newer .b2nd format
    files = {}

    for path in sorted(data_dir.glob("*_seg.b2nd")):
        files[path.name.removesuffix("_seg.b2nd")] = path

    for path in sorted(data_dir.glob("*_seg.npy")):
        case = path.name.removesuffix("_seg.npy")
        files.setdefault(case, path)

    for path in sorted(data_dir.glob("*.npz")):
        case = path.stem
        files.setdefault(case, path)

    if not files:
        raise RuntimeError("No preprocessed segmentations found")

    results = []

    print(f"{'Case':42s} {'Tumour voxels':>15s}  Labels")
    print("-" * 85)

    for case, path in sorted(files.items()):
        seg = read_segmentation(path)

        count = int(np.count_nonzero(seg == LABEL))
        labels = np.unique(seg).tolist()

        results.append({
            "case": case,
            "tumour_voxels": count,
            "labels_present": str(labels)
        })

        print(f"{case:42s} {count:15,d}  {labels}")

    # Save CSV
    output = DATASET_FOLDER / "tumour_voxel_counts.csv"

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case", "tumour_voxels", "labels_present"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    counts = np.array(
        [r["tumour_voxels"] for r in results]
    )

    positive = counts[counts > 0]

    print("\nSummary")
    print("-" * 50)
    print(f"Total scans:                {len(counts)}")
    print(f"Scans with tumour voxels:   {len(positive)}")
    print(f"Scans without tumour:       {np.sum(counts == 0)}")

    if len(positive):
        print(f"Minimum positive voxels:    {positive.min():,}")
        print(f"Median positive voxels:     {np.median(positive):,.0f}")
        print(f"Maximum positive voxels:    {positive.max():,}")

    print(f"\nSaved CSV: {output}")


if __name__ == "__main__":
    main()
