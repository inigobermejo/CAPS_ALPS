import os
import zipfile
import tempfile
import shutil
import json
from pathlib import Path

import numpy as np
import nrrd
import slicerio
import SimpleITK as sitk


# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing your .mrb files
MRB_INPUT_DIR = "C:/Users/lucp13173/Documents/Data/CAPS/Jessa/Segmented"

# nnU-Net dataset output directory
# Example:
# C:\nnUNet_raw\Dataset001_Pancreas
OUTPUT_DIR = "C:/Users/lucp13173/Documents/Data/CAPS/Jessa/nnUNet_raw"

# Keep temporary extracted MRB folders for debugging?
# Normally False.
KEEP_TEMP = False

# Segment names in the Slicer segmentation
PANCREAS_NAME = "017_pancreas_whole"
TUMOR_NAME = "028_pancreatic_lesion_tumor"

# Our fixed nnU-Net labels
PANCREAS_LABEL = 1
TUMOR_LABEL = 2


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

imagesTr = Path(OUTPUT_DIR) / "imagesTr"
labelsTr = Path(OUTPUT_DIR) / "labelsTr"

imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND FILES INSIDE MRB
# ============================================================

def find_files(extract_dir):

    ct_files = []
    seg_files = []

    for root, dirs, files in os.walk(extract_dir):

        for filename in files:

            path = Path(root) / filename
            lower = filename.lower()

            if lower.endswith(".seg.nrrd"):
                seg_files.append(path)

            elif lower.endswith(".nrrd"):
                # Candidate CT files.
                # We deliberately do NOT require "_ct_".
                if not lower.endswith(".seg.nrrd"):
                    ct_files.append(path)

    return ct_files, seg_files


# ============================================================
# IDENTIFY CT
# ============================================================

def choose_ct(ct_candidates, seg_file):

    if not ct_candidates:
        raise RuntimeError("No non-segmentation NRRD files found.")

    # Prefer filenames containing CT-related terms.
    scored = []

    for path in ct_candidates:

        name = path.name.lower()

        score = 0

        if "ct" in name:
            score += 10

        if "veneus" in name or "venous" in name:
            score += 5

        if "abdomen" in name:
            score += 2

        scored.append((score, path))

    scored.sort(
        key=lambda x: (-x[0], str(x[1]))
    )

    return scored[0][1]


# ============================================================
# GET SEGMENT INFORMATION
# ============================================================

def get_target_segments(seg_file):

    segmentation = slicerio.read_segmentation(str(seg_file))

    found = {}

    for segment in segmentation["segments"]:

        name = segment["name"]

        if name == PANCREAS_NAME:

            found["pancreas"] = {
                "name": name,
                "labelValue": int(segment["labelValue"]),
                "layer": int(segment["layer"]),
                "extent": segment.get("extent"),
            }

        elif name == TUMOR_NAME:

            found["tumor"] = {
                "name": name,
                "labelValue": int(segment["labelValue"]),
                "layer": int(segment["layer"]),
                "extent": segment.get("extent"),
            }

    return found


# ============================================================
# EXTRACT A SEGMENT MASK FROM RAW SEG.NRRD
# ============================================================

def extract_segment_mask(seg_file, label_value, layer):

    # Read raw NRRD data.
    raw, header = nrrd.read(str(seg_file))

    print(f"    Raw segmentation shape: {raw.shape}")

    if raw.ndim == 3:

        # Single labelmap layer
        #
        # labelValue is directly encoded in the 3D array.
        mask = raw == label_value

    elif raw.ndim == 4:

        # Multiple labelmap layers.
        #
        # First dimension is the layer dimension.
        if layer >= raw.shape[0]:
            raise RuntimeError(
                f"Layer {layer} does not exist in "
                f"segmentation with shape {raw.shape}"
            )

        layer_data = raw[layer]

        mask = layer_data == label_value

    else:

        raise RuntimeError(
            f"Unexpected segmentation dimensionality: "
            f"{raw.ndim}, shape={raw.shape}"
        )

    return mask.astype(np.uint8), header


# ============================================================
# CREATE SIMPLEITK IMAGE FROM NRRD MASK
# ============================================================

def mask_to_sitk(mask, header):

    """
    Convert a numpy mask whose axes are NRRD axes (X,Y,Z)
    into a SimpleITK image while preserving NRRD geometry.
    """

    # SimpleITK GetImageFromArray expects numpy in Z,Y,X order.
    mask_zyx = np.transpose(mask, (2, 1, 0))

    image = sitk.GetImageFromArray(mask_zyx)

    # NRRD geometry
    space_directions = np.asarray(
        header["space directions"],
        dtype=float
    )

    space_origin = np.asarray(
        header["space origin"],
        dtype=float
    )

    # Columns/vectors correspond to X/Y/Z directions.
    spacing = np.linalg.norm(
        space_directions,
        axis=1
    )

    direction = (
        space_directions / spacing[:, None]
    )

    image.SetSpacing(
        tuple(spacing)
    )

    # SimpleITK direction is row-major 3x3.
    image.SetDirection(
        tuple(direction.flatten())
    )

    image.SetOrigin(
        tuple(space_origin)
    )

    return image


# ============================================================
# RESAMPLE MASK TO CT GEOMETRY
# ============================================================

def resample_to_ct(mask_image, ct_image):

    return sitk.Resample(
        mask_image,
        ct_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )


# ============================================================
# PROCESS ONE MRB
# ============================================================

def process_mrb(mrb_file):

    mrb_file = Path(mrb_file)

    case_id = mrb_file.stem

    # Remove possible Scene. prefix
    # You can change this if you want the exact filename preserved.
    if case_id.startswith("Scene."):
        case_id = case_id[6:]

    print("\n")
    print("=" * 70)
    print(f"PROCESSING: {mrb_file.name}")
    print(f"CASE ID:    {case_id}")
    print("=" * 70)

    temp_dir = tempfile.mkdtemp(
        prefix="mrb_extract_"
    )

    try:

        # ----------------------------------------------------
        # Extract MRB
        # ----------------------------------------------------

        print("\nExtracting MRB...")

        with zipfile.ZipFile(
            mrb_file,
            "r"
        ) as z:

            z.extractall(temp_dir)

        # ----------------------------------------------------
        # Find CT and segmentation
        # ----------------------------------------------------

        ct_candidates, seg_candidates = find_files(
            temp_dir
        )

        print(
            f"Found {len(ct_candidates)} CT candidate(s)"
        )

        print(
            f"Found {len(seg_candidates)} segmentation file(s)"
        )

        if len(seg_candidates) == 0:
            raise RuntimeError(
                "No .seg.nrrd found."
            )

        if len(seg_candidates) > 1:

            print(
                "WARNING: multiple segmentation files found:"
            )

            for f in seg_candidates:
                print("   ", f)

            raise RuntimeError(
                "Multiple .seg.nrrd files; "
                "please add a selection rule."
            )

        seg_file = seg_candidates[0]

        ct_file = choose_ct(
            ct_candidates,
            seg_file
        )

        print("\nCT:")
        print("   ", ct_file)

        print("\nSEG:")
        print("   ", seg_file)

        # ----------------------------------------------------
        # Read CT
        # ----------------------------------------------------

        print("\nReading CT...")

        ct_image = sitk.ReadImage(
            str(ct_file)
        )

        print(
            "    CT size:",
            ct_image.GetSize()
        )

        print(
            "    CT spacing:",
            ct_image.GetSpacing()
        )

        print(
            "    CT origin:",
            ct_image.GetOrigin()
        )

        print(
            "    CT direction:",
            ct_image.GetDirection()
        )

        # ----------------------------------------------------
        # Read segment metadata
        # ----------------------------------------------------

        print("\nReading segmentation metadata...")

        segments = get_target_segments(
            seg_file
        )

        print("\nTarget segments found:")

        for key in ["pancreas", "tumor"]:

            if key in segments:

                s = segments[key]

                print(
                    f"    {key}: "
                    f"labelValue={s['labelValue']}, "
                    f"layer={s['layer']}"
                )

            else:

                print(
                    f"    {key}: NOT FOUND"
                )

        if "pancreas" not in segments:

            raise RuntimeError(
                f"Required segment not found: "
                f"{PANCREAS_NAME}"
            )

        # Tumour is allowed to be absent from a particular case.
        # But the segment itself should normally exist.
        if "tumor" not in segments:

            print(
                "    WARNING: tumour segment metadata "
                "not found."
            )

        # ----------------------------------------------------
        # Extract pancreas
        # ----------------------------------------------------

        print("\nExtracting pancreas...")

        pancreas = segments["pancreas"]

        pancreas_mask, seg_header = extract_segment_mask(
            seg_file,
            pancreas["labelValue"],
            pancreas["layer"]
        )

        print(
            "    Raw pancreas voxels:",
            int(pancreas_mask.sum())
        )

        # ----------------------------------------------------
        # Convert pancreas mask to physical image
        # ----------------------------------------------------

        pancreas_image = mask_to_sitk(
            pancreas_mask,
            seg_header
        )

        print(
            "    Segmentation geometry:",
            pancreas_image.GetSize(),
            pancreas_image.GetSpacing(),
            pancreas_image.GetOrigin()
        )

        # ----------------------------------------------------
        # Resample pancreas onto CT
        # ----------------------------------------------------

        pancreas_ct = resample_to_ct(
            pancreas_image,
            ct_image
        )

        pancreas_ct_array = sitk.GetArrayFromImage(
            pancreas_ct
        )

        print(
            "    CT-grid pancreas voxels:",
            int(np.sum(pancreas_ct_array > 0))
        )

        # ----------------------------------------------------
        # Extract tumour
        # ----------------------------------------------------

        if "tumor" in segments:

            tumor = segments["tumor"]

            tumor_mask, tumor_header = (
                extract_segment_mask(
                    seg_file,
                    tumor["labelValue"],
                    tumor["layer"]
                )
            )

            print(
                "    Raw tumour voxels:",
                int(tumor_mask.sum())
            )

            tumor_image = mask_to_sitk(
                tumor_mask,
                tumor_header
            )

            tumor_ct = resample_to_ct(
                tumor_image,
                ct_image
            )

            tumor_ct_array = (
                sitk.GetArrayFromImage(
                    tumor_ct
                )
            )

            print(
                "    CT-grid tumour voxels:",
                int(np.sum(tumor_ct_array > 0))
            )

        else:

            tumor_ct_array = np.zeros(
                pancreas_ct_array.shape,
                dtype=np.uint8
            )

        # ----------------------------------------------------
        # Construct final nnU-Net labelmap
        # ----------------------------------------------------

        labels_array = np.zeros(
            pancreas_ct_array.shape,
            dtype=np.uint8
        )

        # Pancreas = 1
        labels_array[
            pancreas_ct_array > 0
        ] = PANCREAS_LABEL

        # Tumour = 2
        #
        # If tumour overlaps pancreas, tumour takes priority.
        labels_array[
            tumor_ct_array > 0
        ] = TUMOR_LABEL

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        unique, counts = np.unique(
            labels_array,
            return_counts=True
        )

        print("\nFinal labelmap:")

        for value, count in zip(
            unique,
            counts
        ):

            print(
                f"    label {value}: "
                f"{count:,} voxels"
            )

        if PANCREAS_LABEL not in unique:

            raise RuntimeError(
                "ERROR: final labelmap contains "
                "no pancreas voxels."
            )

        # ----------------------------------------------------
        # Convert array back to SimpleITK
        # ----------------------------------------------------

        labels_image = sitk.GetImageFromArray(
            labels_array
        )

        # Copy exactly the CT geometry.
        labels_image.CopyInformation(
            ct_image
        )

        # ----------------------------------------------------
        # Save CT
        # ----------------------------------------------------

        image_output = (
            imagesTr /
            f"{case_id}_0000.nii.gz"
        )

        label_output = (
            labelsTr /
            f"{case_id}.nii.gz"
        )

        print("\nSaving:")

        print(
            "    image:",
            image_output
        )

        print(
            "    label:",
            label_output
        )

        sitk.WriteImage(
            ct_image,
            str(image_output),
            useCompression=True
        )

        sitk.WriteImage(
            sitk.Cast(
                labels_image,
                sitk.sitkUInt8
            ),
            str(label_output),
            useCompression=True
        )

        # ----------------------------------------------------
        # Verify files after writing
        # ----------------------------------------------------

        check_image = sitk.ReadImage(
            str(image_output)
        )

        check_label = sitk.ReadImage(
            str(label_output)
        )

        check_array = sitk.GetArrayFromImage(
            check_label
        )

        print("\nVerification:")

        print(
            "    CT size:",
            check_image.GetSize()
        )

        print(
            "    label size:",
            check_label.GetSize()
        )

        print(
            "    label spacing:",
            check_label.GetSpacing()
        )

        print(
            "    label direction:",
            check_label.GetDirection()
        )

        print(
            "    label unique values:",
            np.unique(check_array)
        )

        # Geometry check
        if (
            check_image.GetSize()
            != check_label.GetSize()
        ):
            raise RuntimeError(
                "CT and labelmap sizes differ!"
            )

        if not np.allclose(
            check_image.GetSpacing(),
            check_label.GetSpacing()
        ):
            raise RuntimeError(
                "CT and labelmap spacing differ!"
            )

        print(
            "\nSUCCESS:",
            case_id
        )

        return True

    except Exception as e:

        print("\n!!! ERROR !!!")
        print(
            f"{case_id}: {type(e).__name__}: {e}"
        )

        return False

    finally:

        if KEEP_TEMP:

            print(
                "\nTemporary files retained at:"
            )
            print(temp_dir)

        else:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )


# ============================================================
# PROCESS ALL MRBs
# ============================================================

def main():

    mrb_files = sorted(
        Path(MRB_INPUT_DIR).glob("*.mrb")
    )

    print(
        f"Found {len(mrb_files)} MRB files."
    )

    if not mrb_files:

        raise RuntimeError(
            "No .mrb files found."
        )

    successful = []
    failed = []

    for mrb_file in mrb_files:

        success = process_mrb(
            mrb_file
        )

        if success:
            successful.append(
                mrb_file.name
            )
        else:
            failed.append(
                mrb_file.name
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Failed:     {len(failed)}"
    )

    if failed:

        print("\nFailed cases:")

        for case in failed:
            print(
                "   ",
                case
            )

    # ========================================================
    # CREATE nnU-Net dataset.json
    # ========================================================

    dataset_json = {

        "channel_names": {
            "0": "CT"
        },

        "labels": {
            "background": 0,
            "pancreas": 1,
            "tumor": 2
        },

        "numTraining": len(successful),

        "file_ending": ".nii.gz"
    }

    dataset_json_path = (
        Path(OUTPUT_DIR) /
        "dataset.json"
    )

    with open(
        dataset_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset_json,
            f,
            indent=4
        )

    print(
        "\nCreated:",
        dataset_json_path
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()