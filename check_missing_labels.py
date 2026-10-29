from pathlib import Path
import os
import tempfile
import zipfile
import csv

import numpy as np
import nrrd
import slicerio


# ============================================================
# SETTINGS
# ============================================================

MRB_DIR = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\Segmented")



OUTPUT_DIR = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\missing_labels")


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TARGET_LABELS = [
    "003_aorta",
    "005_celiac_artery",
    "007_common_bile_duct",
    "008_duodenum",
    "017_pancreas_whole",
    "021_pancreatic_duct",
    "022_postcava_IVC",
    "024_spleen",
    "026_SMA",
    "028_pancreatic_lesion_tumor",
    "035_portal_vein",
    "036_SMV",
    "037_splenic_vein",
    "038_IMV",
]


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
            f"Expected exactly one .seg.nrrd, found {len(files)}"
        )

    return files[0]


def get_layer_axis(raw, header):

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

    # Default for Slicer segmentation NRRDs
    return 0


def segment_has_voxels(
    raw,
    layer_axis,
    segment_info
):

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

    return np.any(
        layer_data == label_value
    )


def case_id_from_mrb(mrb):

    name = mrb.stem

    if name.startswith("Scene."):
        name = name[6:]

    return name


# ============================================================
# PROCESS ONE MRB
# ============================================================

def get_missing_labels(mrb):

    with tempfile.TemporaryDirectory() as tmp:

        with zipfile.ZipFile(mrb, "r") as z:
            z.extractall(tmp)

        seg_file = find_segmentation_file(tmp)

        # Metadata
        seg = slicerio.read_segmentation(
            str(seg_file)
        )

        segments = {
            s["name"]: s
            for s in seg["segments"]
        }

        # Actual voxel data
        raw, header = nrrd.read(
            str(seg_file)
        )

        layer_axis = get_layer_axis(
            raw,
            header
        )

        missing = []

        for label_name in TARGET_LABELS:

            # Segment not defined
            if label_name not in segments:

                missing.append(label_name)
                continue

            # Segment exists but contains no voxels
            if not segment_has_voxels(
                raw,
                layer_axis,
                segments[label_name]
            ):

                missing.append(label_name)

        return missing


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


# Number of scans in which each label is missing
missing_counts = {
    label: 0
    for label in TARGET_LABELS
}

# Missing labels per scan
missing_by_scan = {}


for i, mrb in enumerate(mrb_files, start=1):

    case_id = case_id_from_mrb(mrb)

    print(
        f"[{i}/{len(mrb_files)}] "
        f"{case_id}"
    )

    try:

        missing = get_missing_labels(
            mrb
        )

    except Exception as e:

        print(
            f"  ERROR: {e}"
        )
        continue

    if missing:

        missing_by_scan[case_id] = missing

        for label in missing:
            missing_counts[label] += 1

        print(
            "  Missing:",
            ", ".join(missing)
        )

    else:

        print("  All labels present")


# ============================================================
# REPORT 1:
# LABELS MISSING FROM AT LEAST ONE SCAN
# ============================================================

summary_file = (
    OUTPUT_DIR
    / "missing_labels_summary.csv"
)

with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "label",
        "missing_scans",
        "total_scans",
        "present_scans",
        "present_percent"
    ])

    for label, count in sorted(
        missing_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):

        if count == 0:
            continue

        present = (
            len(mrb_files) - count
        )

        writer.writerow([
            label,
            count,
            len(mrb_files),
            present,
            round(
                100 * present / len(mrb_files),
                1
            )
        ])


# ============================================================
# REPORT 2:
# SCANS WITH AT LEAST ONE MISSING LABEL
# ============================================================

scan_file = (
    OUTPUT_DIR
    / "scans_with_missing_labels.csv"
)

with open(
    scan_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "scan",
        "number_missing",
        "missing_labels"
    ])

    for case_id, missing in sorted(
        missing_by_scan.items()
    ):

        writer.writerow([
            case_id,
            len(missing),
            "; ".join(missing)
        ])


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("LABELS MISSING FROM AT LEAST ONE SCAN")
print("=" * 70)

for label, count in sorted(
    missing_counts.items(),
    key=lambda x: (-x[1], x[0])
):

    if count > 0:

        print(
            f"{label:35s} "
            f"{count:3d} / {len(mrb_files)} scans missing"
        )


print()
print("=" * 70)
print("SCANS WITH MISSING LABELS")
print("=" * 70)

for case_id, missing in sorted(
    missing_by_scan.items()
):

    print()
    print(case_id)

    for label in missing:
        print(f"    {label}")


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Total scans:                    {len(mrb_files)}"
)

print(
    f"Scans with >=1 missing label:   "
    f"{len(missing_by_scan)}"
)

print(
    f"Scans with all labels present:  "
    f"{len(mrb_files) - len(missing_by_scan)}"
)

print()
print(f"Saved: {summary_file}")
print(f"Saved: {scan_file}")