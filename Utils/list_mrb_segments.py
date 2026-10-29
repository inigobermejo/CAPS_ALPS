from pathlib import Path
import argparse
import tempfile
import zipfile

import numpy as np
import nrrd
import slicerio


MRB_FILE = r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\Segmented\Scene.CAPS1730.v-validated.mrb"


def find_segmentation_file(root):
    """Find the .seg.nrrd inside the extracted MRB."""
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
    """
    For a 4D Slicer segmentation, identify the axis that
    represents segmentation layers.
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
            if str(kind).lower() in {
                "list",
                "vector"
            }:
                return i

    # Slicer normally stores layers in the first dimension.
    print(
        "WARNING: could not identify layer axis from NRRD "
        "header; assuming axis 0."
    )

    return 0


def list_nonempty_segments(mrb_file):

    mrb_file = Path(mrb_file)

    with tempfile.TemporaryDirectory() as tmp:

        # ----------------------------------------------------
        # Extract MRB
        # ----------------------------------------------------

        with zipfile.ZipFile(mrb_file, "r") as z:
            z.extractall(tmp)

        seg_file = find_segmentation_file(tmp)

        print(f"MRB:          {mrb_file.name}")
        print(f"Segmentation: {seg_file.name}")

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        segmentation = slicerio.read_segmentation(
            str(seg_file)
        )

        segments = segmentation["segments"]

        # ----------------------------------------------------
        # Raw segmentation voxels
        # ----------------------------------------------------

        raw, header = nrrd.read(
            str(seg_file)
        )

        layer_axis = get_layer_axis(
            raw,
            header
        )

        print(f"Raw shape:    {raw.shape}")
        print()
        print("Non-empty segments:")
        print("-" * 75)

        nonempty = []

        for segment in segments:

            name = segment["name"]
            label_value = segment["labelValue"]
            layer = segment.get("layer", 0)

            # ----------------------------------------------
            # Select correct segmentation layer
            # ----------------------------------------------

            if raw.ndim == 3:

                layer_data = raw

            else:

                layer_data = np.take(
                    raw,
                    int(layer),
                    axis=layer_axis
                )

            # ----------------------------------------------
            # Count voxels for this segment
            # ----------------------------------------------

            voxel_count = int(
                np.count_nonzero(
                    layer_data == label_value
                )
            )

            if voxel_count > 0:

                nonempty.append({
                    "name": name,
                    "layer": layer,
                    "label": label_value,
                    "voxels": voxel_count
                })

                print(
                    f"{name:40s} "
                    f"layer={layer:<3} "
                    f"label={label_value:<3} "
                    f"voxels={voxel_count:,}"
                )

        print("-" * 75)

        print(
            f"{len(nonempty)} non-empty segments "
            f"out of {len(segments)} defined segments."
        )

        return nonempty


if __name__ == "__main__":

    list_nonempty_segments(MRB_FILE)