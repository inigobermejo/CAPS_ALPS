import SimpleITK as sitk
import numpy as np

filename = r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\nnUNet_raw\Dataset501_Pancreas\labelsTr\CAPS1090.nii.gz"

img = sitk.ReadImage(filename)
arr = sitk.GetArrayFromImage(img)

print("SIZE:")
print(img.GetSize())

print("\nSPACING:")
print(img.GetSpacing())

print("\nORIGIN:")
print(img.GetOrigin())

print("\nDIRECTION:")
print(img.GetDirection())

print("\nARRAY SHAPE:")
print(arr.shape)

print("\nUNIQUE VALUES:")
values, counts = np.unique(arr, return_counts=True)

for v, c in zip(values, counts):
    print(f"  {v}: {c:,} voxels")

print("\nNONZERO:")
print("  total:", np.count_nonzero(arr))

for label in [1, 2]:
    coords = np.argwhere(arr == label)

    if len(coords) == 0:
        print(f"  label {label}: ABSENT")
    else:
        print(
            f"  label {label}: "
            f"{len(coords):,} voxels, "
            f"index range={coords.min(axis=0)} -> "
            f"{coords.max(axis=0)}"
        )