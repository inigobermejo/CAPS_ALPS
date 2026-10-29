from pathlib import Path
import tempfile
import shutil
import json
import random

import nrrd
import numpy as np
import SimpleITK as sitk
import slicerio
import zipfile


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = Path(r"C:/Users/lucp13173/Documents/Data/CAPS/Jessa/Segmented")
OUTPUT_DIR = Path("C:/Users/lucp13173/Documents/Data/CAPS/Jessa/Preprocessed")

PANCREAS_NAME = "017_pancreas_whole"
TUMOR_NAME = "028_pancreatic_lesion_tumor"
DUCT_NAME = "021_pancreatic_duct"

# Output labels
PANCREAS_LABEL = 1
TUMOR_LABEL = 2

N_FOLDS = 5
SEED = 42


# ============================================================
# HELPERS
# ============================================================

def extract_segment(seg_file, segment):
    """Extract one segment from a .seg.nrrd as a SITK image."""

    info = segment
    raw, header = nrrd.read(seg_file)

    label = info["labelValue"]
    layer = info["layer"]

    if raw.ndim == 4:
        raw = raw[layer]

    mask = (raw == label).astype(np.uint8)

    # NRRD directions/origin
    dirs = np.asarray(header["space directions"], dtype=float)

    # Keep the three spatial directions
    dirs = np.asarray([d for d in dirs if np.all(np.isfinite(d))][:3])

    spacing = np.linalg.norm(dirs, axis=1)
    direction = np.column_stack(
        [dirs[i] / spacing[i] for i in range(3)]
    )

    image = sitk.GetImageFromArray(
        np.transpose(mask, (2, 1, 0))
    )
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(header["space origin"]))
    image.SetDirection(tuple(direction.flatten()))

    return image


def resample_to_ct(mask, ct):
    return sitk.Resample(
        mask,
        ct,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


# ============================================================
# MAIN
# ============================================================

images_dir = OUTPUT_DIR / "imagesTr"
labels_dir = OUTPUT_DIR / "labelsTr"

images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

cases = []

for mrb in sorted(INPUT_DIR.glob("*.mrb")):

    case_id = mrb.stem
    if case_id.startswith("Scene."):
        case_id = case_id[6:]

    with tempfile.TemporaryDirectory() as tmp:

        # ----------------------------------------------------
        # Extract MRB
        # ----------------------------------------------------

        with zipfile.ZipFile(mrb, "r") as z:
            z.extractall(tmp)

        tmp = Path(tmp)

        # One CT and one segmentation
        seg_file = next(tmp.rglob("*.seg.nrrd"))
        nrrds = [
            p for p in tmp.rglob("*.nrrd")
            if not p.name.endswith(".seg.nrrd")
        ]

        if len(nrrds) != 1:
            raise RuntimeError(
                f"{mrb.name}: expected one CT NRRD, found {len(nrrds)}"
            )

        ct_file = nrrds[0]

        # ----------------------------------------------------
        # CT
        # ----------------------------------------------------

        ct = sitk.ReadImage(
            str(ct_file),
            sitk.sitkFloat32
        )

        # ----------------------------------------------------
        # Segment metadata
        # ----------------------------------------------------

        seg = slicerio.read_segmentation(
            str(seg_file)
        )

        segments = {
            s["name"]: s
            for s in seg["segments"]
        }

        # ----------------------------------------------------
        # Extract the two relevant segments
        # ----------------------------------------------------

        pancreas = extract_segment(
            str(seg_file),
            segments[PANCREAS_NAME]
        )

        tumor = extract_segment(
            str(seg_file),
            segments[TUMOR_NAME]
        )

        duct = extract_segment(
            str(seg_file),
            segments[DUCT_NAME]
        )

        # ----------------------------------------------------
        # Resample to CT geometry
        # ----------------------------------------------------

        pancreas = resample_to_ct(
            pancreas, ct
        )

        tumor = resample_to_ct(
            tumor, ct
        )

        duct = resample_to_ct(
            duct, ct
        )

        # ----------------------------------------------------
        # Create nnU-Net-style labelmap
        # ----------------------------------------------------

        p = sitk.GetArrayFromImage(pancreas)
        t = sitk.GetArrayFromImage(tumor)
        d = sitk.GetArrayFromImage(duct)

        labels = np.zeros_like(
            p,
            dtype=np.uint8
        )

        # Whole pancreas includes the pancreatic duct
        whole_pancreas = (p > 0) | (d > 0)
        labels[whole_pancreas] = PANCREAS_LABEL
        labels[t > 0] = TUMOR_LABEL

        label = sitk.GetImageFromArray(labels)
        label.CopyInformation(ct)

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        sitk.WriteImage(
            ct,
            str(images_dir / f"{case_id}_0000.nii.gz"),
            useCompression=True
        )

        sitk.WriteImage(
            label,
            str(labels_dir / f"{case_id}.nii.gz"),
            useCompression=True
        )

        cases.append(case_id)

        print(f"Processed {case_id}")


# ============================================================
# NON-STRATIFIED 5-FOLD SPLIT
# ============================================================

random.seed(SEED)
random.shuffle(cases)

folds = [
    cases[i::N_FOLDS]
    for i in range(N_FOLDS)
]

with open(
    OUTPUT_DIR / "folds.json",
    "w"
) as f:

    json.dump(
        {
            "n_folds": N_FOLDS,
            "seed": SEED,
            "folds": {
                str(i): fold
                for i, fold in enumerate(folds)
            }
        },
        f,
        indent=2
    )

print(f"\nProcessed {len(cases)} cases.")
print(f"Dataset: {OUTPUT_DIR}")
print(f"Folds:   {OUTPUT_DIR / 'folds.json'}")