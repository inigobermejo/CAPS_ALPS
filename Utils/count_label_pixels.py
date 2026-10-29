from pathlib import Path
import numpy as np
import SimpleITK as sitk

PREDICTIONS_DIR = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\nnUNet\nnUNet_raw\Dataset001_PancreasTumor\labelsTr")
CASE = "CAPS1320.v-validated"

pred = sitk.GetArrayFromImage(
    sitk.ReadImage(str(PREDICTIONS_DIR / f"{CASE}.nii.gz"))
)

labels, counts = np.unique(pred, return_counts=True)

for label, count in zip(labels, counts):
    print(f"Class {label}: {count:,} voxels")