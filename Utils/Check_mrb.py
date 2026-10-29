import os
import zipfile
import tempfile
import numpy as np
import slicerio
import nrrd


# ---------------------------------------------------------
# CHANGE THIS TO YOUR .MRB FILE
# ---------------------------------------------------------
mrb_file = r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\Segmented\Scene.CAPS1730.v-validated.mrb"


# ---------------------------------------------------------
# Extract MRB
# ---------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:

    print("Extracting MRB to:")
    print(tmpdir)

    with zipfile.ZipFile(mrb_file, "r") as z:
        z.extractall(tmpdir)

    print("\n========== SEGMENTATION FILES ==========")

    for root, dirs, files in os.walk(tmpdir):
        for file in files:
            print(os.path.relpath(os.path.join(root, file), tmpdir))
    # -----------------------------------------------------
    # Find all SEG.NRRD files
    # -----------------------------------------------------
    seg_files = []

    for root, dirs, files in os.walk(tmpdir):
        for file in files:
            if file.lower().endswith(".seg.nrrd"):
                seg_files.append(os.path.join(root, file))

    print("\nSEG.NRRD files found:")
    for f in seg_files:
        print(" ", f)

    if not seg_files:
        raise RuntimeError("No .seg.nrrd file found in the MRB")

    if len(seg_files) > 1:
        print("\nWARNING: More than one .seg.nrrd found.")
        print("Using the first one.")

    seg_file = seg_files[0]

    print("\nUsing segmentation:")
    print(seg_file)

    raw, header = nrrd.read(seg_file)

    print("\n========== RAW NRRD ==========")
    print("shape:", raw.shape)
    print("dtype:", raw.dtype)

    unique, counts = np.unique(raw, return_counts=True)

    print("\nUnique values:")
    for value, count in zip(unique, counts):
        print(f"  {value}: {count:,}")
        
    print("\n========== HEADER ==========")

    for key in [
        "dimension",
        "sizes",
        "type",
        "kinds",
        "space",
        "space directions",
        "space origin",
        "encoding"
    ]:
        print(f"{key}: {header.get(key)}")

    print("RAW shape:", raw.shape)
    print("RAW dtype:", raw.dtype)

    unique, counts = np.unique(raw, return_counts=True)

    print("\nRAW UNIQUE VALUES:")
    for value, count in zip(unique, counts):
        print(f"  {value}: {count:,}")

    print("\n========== SEG.NRRD HEADER ==========")

    for key, value in header.items():
        print(f"{key}: {value}")

    # -----------------------------------------------------
    # Read segmentation
    # -----------------------------------------------------
    seg = slicerio.read_segmentation(seg_file)

    print("\n========== SEGMENT METADATA ==========")

    for i, s in enumerate(seg["segments"]):
        print(f"\nSegment {i}")
        for key, value in s.items():
            print(f"  {key}: {value}")

    arr = np.asarray(seg["voxels"])

    pancreas_mask = arr[0] == 33

    # Tumour: label 12, but this case has zero tumour voxels
    tumour_mask = arr[0] == 12

    print("Pancreas voxels:", np.sum(pancreas_mask))
    print("Tumour voxels:", np.sum(tumour_mask))

    print("\n========== VOXEL INFORMATION ==========")
    print("voxels type:", type(seg["voxels"]))
    print("voxels shape:", arr.shape)
    print("voxels dtype:", arr.dtype)

    print("\nNumber of segments:", len(seg["segments"]))


    # -----------------------------------------------------
    # Inspect the two segments we care about
    # -----------------------------------------------------
    target_names = [
        "017_pancreas_whole",
        "028_pancreatic_lesion_tumor"
    ]

    for s in seg["segments"]:

        if s["name"] not in target_names:
            continue

        print("\n======================================")
        print("SEGMENT:", s["name"])
        print("======================================")

        print("layer:", s["layer"])
        print("labelValue:", s["labelValue"])
        print("extent:", s["extent"])

        layer = s["layer"]
        label = s["labelValue"]

        # -------------------------------------------------
        # Inspect the layer
        # -------------------------------------------------
        layer_arr = np.asarray(seg["voxels"][layer])

        print("\nLayer information:")
        print(" shape:", layer_arr.shape)
        print(" dtype:", layer_arr.dtype)

        unique, counts = np.unique(
            layer_arr,
            return_counts=True
        )

        print("\nUnique values in layer:")

        # Don't print an enormous list if there are many
        # values, although normally there should be very few.
        for value, count in zip(unique, counts):
            print(f"  {value}: {count:,} voxels")

        print("\nlabelValue:", label)
        print(
            "number equal to labelValue:",
            np.sum(layer_arr == label)
        )

        print(
            "number of non-zero voxels:",
            np.sum(layer_arr != 0)
        )

    print("\n========== ALL SEGMENTS ==========")

    arr = np.asarray(seg["voxels"])

    print("voxels shape:", arr.shape)

    # There is one layer dimension followed by X/Y/Z
    for s in seg["segments"]:

        layer = s["layer"]
        label = s["labelValue"]

        layer_arr = arr[layer]

        n = np.sum(layer_arr == label)

        print(
            f"{s['name']:<40} "
            f"layer={layer:<2} "
            f"label={label:<3} "
            f"matching_voxels={n:,}"
        )


    print("\n========== ACTUAL LABEL COUNTS ==========")

    # Since there are only a few dozen labels, this is manageable
    unique, counts = np.unique(arr[0], return_counts=True)

    for value, count in zip(unique, counts):
        print(f"label {value:<3}: {count:,} voxels")

    print("\n")
    print("Temporary directory will now be deleted.")