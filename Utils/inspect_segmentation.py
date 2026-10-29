from pathlib import Path
import zipfile
import tempfile
import slicerio

# ---------------------------------------------------------------------
# CHANGE THIS
# ---------------------------------------------------------------------
mrb_file = Path("C:/Users/lucp13173/Documents/Data/CAPS/Jessa/Segmented/Scene.CAPS4480.v-validated.mrb")
# ---------------------------------------------------------------------


with tempfile.TemporaryDirectory() as tmp:

    tmp = Path(tmp)

    # Extract only the segmentation from the MRB
    with zipfile.ZipFile(mrb_file, "r") as z:
        seg_files = [
            f for f in z.namelist()
            if f.lower().endswith(".seg.nrrd")
        ]

        if not seg_files:
            raise RuntimeError("No .seg.nrrd file found in MRB")

        print("Segmentation file:")
        print(seg_files[0])

        seg_path = tmp / Path(seg_files[0]).name

        with z.open(seg_files[0]) as src, open(seg_path, "wb") as dst:
            dst.write(src.read())

    # Read segmentation
    segmentation = slicerio.read_segmentation(str(seg_path))

    print("\nSegmentation contents:")
    print("=" * 80)

    print(segmentation.keys())

    print("\nSegments:")
    print("=" * 80)

    # Print the structure without dumping voxel data
    for key, value in segmentation.items():

        if key == "segments":
            print(f"\nNumber of segments: {len(value)}")

            for i, segment in enumerate(value):

                print(f"\nSegment {i}:")
                print("-" * 40)

                if isinstance(segment, dict):
                    for k, v in segment.items():

                        # Don't print potentially huge voxel arrays
                        if hasattr(v, "shape"):
                            print(f"{k}: array with shape {v.shape}")
                        else:
                            print(f"{k}: {v}")

                else:
                    print(type(segment))
                    print(segment)

        else:
            print(f"\n{key}: {value}")