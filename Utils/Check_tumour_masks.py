from pathlib import Path
import SimpleITK as sitk
import numpy as np

labels_dir = Path(r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\Preprocessed\labelsTr")

for f in sorted(labels_dir.glob("*.nii.gz")):

    x = sitk.GetArrayFromImage(
        sitk.ReadImage(str(f))
    )

    n = np.sum(x == 2)

    print(
        f"{f.stem:25s} "
        f"tumour voxels = {n:,}"
    )