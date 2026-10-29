from pathlib import Path
import json
import csv

import numpy as np
import SimpleITK as sitk


# ============================================================
# SETTINGS — EDIT THESE
# ============================================================

FOLD = 0

# Predictions obtained by applying the full-image model
# to the cropped CT images.
PREDICTIONS_DIR = Path(
    r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\TwoStage\diagnostics\imagesTr_fold0_predictions"
)

# Ground truth of the cropped images (Dataset004).
LABELS_DIR = Path(
    r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\TwoStage\stage2_tumour\labelsTr"
)

# Original cross-validation splits (Dataset001).
SPLITS_FILE = Path(
    r"C:\Users\lucp13173\Documents\Data\CAPS\Jessa\TwoStage\diagnostics\splits_final.json"
)

# Tumour label in the full-image model's predictions.
PRED_LABEL = 2

# Tumour label in the cropped ground truth.
GT_LABEL = 1

# ------------------------------------------------------------
# OPTIONAL: compare with predictions on original full CTs
# Set both to None to disable.
# ------------------------------------------------------------

FULL_PREDICTIONS_DIR = None
FULL_LABELS_DIR = None

# Example:
# FULL_PREDICTIONS_DIR = Path(
#     r"C:\path\to\full_predictions_fold0"
# )
# FULL_LABELS_DIR = Path(
#     r"C:\path\to\nnUNet_raw\Dataset001_PancreasTumor\labelsTr"
# )

FULL_PRED_LABEL = 2
FULL_GT_LABEL = 2

# Output CSV.
OUTPUT_FILE = Path("cropped_evaluation_fold0.csv")

# ============================================================
# FUNCTIONS
# ============================================================

def load_validation_cases(split_file, fold):
    """Read nnU-Net splits_final.json or our original folds.json."""

    with open(split_file) as f:
        splits = json.load(f)

    if isinstance(splits, list):
        return splits[fold]["val"]

    return splits["folds"][str(fold)]


def check_geometry(pred_img, gt_img, case):

    if (
        pred_img.GetSize() != gt_img.GetSize()
        or not np.allclose(
            pred_img.GetSpacing(), gt_img.GetSpacing()
        )
        or not np.allclose(
            pred_img.GetOrigin(), gt_img.GetOrigin(), atol=1e-3
        )
        or not np.allclose(
            pred_img.GetDirection(), gt_img.GetDirection()
        )
    ):
        raise ValueError(
            f"Image geometry mismatch for {case}"
        )


def calculate_metrics(gt, pred):

    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))

    n_gt = int(np.count_nonzero(gt))
    n_pred = int(np.count_nonzero(pred))

    # Exclude tumour-negative cases from the mean Dice.
    if n_gt == 0:
        dice = float("nan")
        recall = float("nan")
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        recall = tp / (tp + fn)

    precision = (
        tp / (tp + fp)
        if n_pred > 0
        else float("nan")
    )

    return {
        "gt_voxels": n_gt,
        "pred_voxels": n_pred,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def evaluate_case(
    case,
    predictions_dir,
    labels_dir,
    pred_label,
    gt_label,
):

    pred_path = predictions_dir / f"{case}.nii.gz"
    gt_path = labels_dir / f"{case}.nii.gz"

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    pred_img = sitk.ReadImage(str(pred_path))
    gt_img = sitk.ReadImage(str(gt_path))

    check_geometry(pred_img, gt_img, case)

    pred = sitk.GetArrayFromImage(pred_img) == pred_label
    gt = sitk.GetArrayFromImage(gt_img) == gt_label

    return calculate_metrics(gt, pred)


# ============================================================
# MAIN
# ============================================================

def main():

    cases = load_validation_cases(
        SPLITS_FILE, FOLD
    )

    compare_full = (
        FULL_PREDICTIONS_DIR is not None
        and FULL_LABELS_DIR is not None
    )

    if (FULL_PREDICTIONS_DIR is None) != (FULL_LABELS_DIR is None):
        raise ValueError(
            "Set both FULL_PREDICTIONS_DIR and FULL_LABELS_DIR, "
            "or leave both as None."
        )

    results = []

    print(f"\nEvaluating fold {FOLD}")
    print(f"Number of validation cases: {len(cases)}\n")

    for case in cases:

        cropped = evaluate_case(
            case,
            PREDICTIONS_DIR,
            LABELS_DIR,
            PRED_LABEL,
            GT_LABEL,
        )

        row = {
            "case": case,
            **cropped,
        }

        if compare_full:

            full = evaluate_case(
                case,
                FULL_PREDICTIONS_DIR,
                FULL_LABELS_DIR,
                FULL_PRED_LABEL,
                FULL_GT_LABEL,
            )

            row["full_dice"] = full["dice"]
            row["full_pred_voxels"] = full["pred_voxels"]
            row["dice_change"] = (
                cropped["dice"] - full["dice"]
            )

        results.append(row)

        message = (
            f"{case:35s} "
            f"GT={cropped['gt_voxels']:7d} "
            f"Pred={cropped['pred_voxels']:7d} "
            f"Dice={cropped['dice']:.3f}"
        )

        if compare_full:
            message += (
                f"  Full={row['full_dice']:.3f}"
                f"  Change={row['dice_change']:+.3f}"
            )

        print(message)

    if not results:
        raise RuntimeError("No validation cases found")

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(OUTPUT_FILE, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    positives = [
        r for r in results
        if r["gt_voxels"] > 0
    ]

    negatives = [
        r for r in results
        if r["gt_voxels"] == 0
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Total cases:                  {len(results)}")
    print(f"Tumour-positive cases:        {len(positives)}")
    print(f"Tumour-negative cases:        {len(negatives)}")

    print(
        "Positive cases predicted empty: ",
        sum(r["pred_voxels"] == 0 for r in positives)
    )

    print(
        "Negative cases with false positives:",
        sum(r["pred_voxels"] > 0 for r in negatives)
    )

    if positives:

        cropped_dice = np.array([
            r["dice"] for r in positives
        ])

        print(
            f"\nMean cropped Dice:           "
            f"{cropped_dice.mean():.3f}"
        )

        print(
            f"Median cropped Dice:         "
            f"{np.median(cropped_dice):.3f}"
        )

        if compare_full:

            full_dice = np.array([
                r["full_dice"] for r in positives
            ])

            print(
                f"Mean full-image Dice:        "
                f"{full_dice.mean():.3f}"
            )

            print(
                f"Mean Dice change:            "
                f"{(cropped_dice - full_dice).mean():+.3f}"
            )

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()