from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import SimpleITK as sitk

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)


# ============================================================
# BASIC METRICS
# ============================================================

def dice(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    ng = gt.sum()
    npred = pred.sum()

    if ng == 0 and npred == 0:
        return np.nan

    return (
        2.0 * np.logical_and(gt, pred).sum()
        / (ng + npred)
    )


def iou(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return np.nan

    return np.logical_and(gt, pred).sum() / union


def precision(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()

    if tp + fp == 0:
        return np.nan

    return tp / (tp + fp)


def recall(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    if tp + fn == 0:
        return np.nan

    return tp / (tp + fn)


# ============================================================
# SURFACE METRICS
# ============================================================

def surface_distances(gt, pred, spacing_zyx):
    """
    Symmetric surface distances in mm.

    spacing_zyx must correspond to NumPy array order:
        Z, Y, X
    """

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    # HD95 / ASSD are undefined if either object is empty
    if not gt.any() or not pred.any():
        return None

    structure = generate_binary_structure(3, 1)

    gt_surface = gt & ~binary_erosion(
        gt,
        structure=structure,
        border_value=0
    )

    pred_surface = pred & ~binary_erosion(
        pred,
        structure=structure,
        border_value=0
    )

    distance_to_gt = distance_transform_edt(
        ~gt_surface,
        sampling=spacing_zyx
    )

    distance_to_pred = distance_transform_edt(
        ~pred_surface,
        sampling=spacing_zyx
    )

    pred_to_gt = distance_to_gt[pred_surface]
    gt_to_pred = distance_to_pred[gt_surface]

    return np.concatenate([
        pred_to_gt,
        gt_to_pred
    ])


def hd95(gt, pred, spacing_zyx):
    distances = surface_distances(
        gt,
        pred,
        spacing_zyx
    )

    if distances is None:
        return np.nan

    return float(
        np.percentile(distances, 95)
    )


def assd(gt, pred, spacing_zyx):
    distances = surface_distances(
        gt,
        pred,
        spacing_zyx
    )

    if distances is None:
        return np.nan

    return float(
        np.mean(distances)
    )


# ============================================================
# GEOMETRY CHECK
# ============================================================

def check_geometry(gt, pred, case_id):
    if gt.GetSize() != pred.GetSize():
        raise RuntimeError(
            f"{case_id}: different image sizes:\n"
            f"GT:   {gt.GetSize()}\n"
            f"Pred: {pred.GetSize()}"
        )

    if not np.allclose(
        gt.GetSpacing(),
        pred.GetSpacing(),
        atol=1e-4
    ):
        raise RuntimeError(
            f"{case_id}: spacing differs."
        )

    if not np.allclose(
        gt.GetOrigin(),
        pred.GetOrigin(),
        atol=1e-3
    ):
        raise RuntimeError(
            f"{case_id}: origin differs."
        )

    if not np.allclose(
        gt.GetDirection(),
        pred.GetDirection(),
        atol=1e-4
    ):
        raise RuntimeError(
            f"{case_id}: direction differs."
        )


# ============================================================
# EVALUATE ONE STRUCTURE
# ============================================================

def evaluate_structure(
    gt,
    pred,
    spacing,
    voxel_volume_ml
):

    gt_count = int(gt.sum())
    pred_count = int(pred.sum())

    return {
        "gt_present": gt_count > 0,
        "pred_present": pred_count > 0,

        "gt_volume_ml":
            gt_count * voxel_volume_ml,

        "pred_volume_ml":
            pred_count * voxel_volume_ml,

        "dice":
            dice(gt, pred),

        "iou":
            iou(gt, pred),

        "precision":
            precision(gt, pred),

        "recall":
            recall(gt, pred),

        "hd95_mm":
            hd95(gt, pred, spacing),

        "assd_mm":
            assd(gt, pred, spacing),
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(
    dataset_dir,
    predictions_dir,
    output_dir
):

    dataset_dir = Path(dataset_dir)
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load folds
    # --------------------------------------------------------

    with open(
        dataset_dir / "folds.json"
    ) as f:
        folds_json = json.load(f)

    folds = folds_json["folds"]

    rows = []

    # --------------------------------------------------------
    # Evaluate folds
    # --------------------------------------------------------

    for fold in range(5):

        print(f"\nEvaluating fold {fold}")

        validation_cases = folds[str(fold)]

        prediction_folder = (
            predictions_dir /
            f"fold_{fold}"
        )

        for case_id in validation_cases:

            gt_file = (
                dataset_dir /
                "labelsTr" /
                f"{case_id}.nii.gz"
            )

            pred_file = (
                prediction_folder /
                f"{case_id}.nii.gz"
            )

            if not pred_file.exists():
                print(
                    f"WARNING: prediction missing: "
                    f"{pred_file}"
                )
                continue

            gt_img = sitk.ReadImage(
                str(gt_file)
            )

            pred_img = sitk.ReadImage(
                str(pred_file)
            )

            check_geometry(
                gt_img,
                pred_img,
                case_id
            )

            gt = sitk.GetArrayFromImage(
                gt_img
            )

            pred = sitk.GetArrayFromImage(
                pred_img
            )

            # SimpleITK = X,Y,Z
            # NumPy      = Z,Y,X
            spacing = tuple(
                reversed(
                    gt_img.GetSpacing()
                )
            )

            voxel_volume_ml = (
                np.prod(gt_img.GetSpacing())
                / 1000.0
            )

            # ------------------------------------------------
            # Structures
            # ------------------------------------------------

            structures = {

                # anatomical pancreas:
                # pancreas + duct + tumour
                "whole_pancreas": (
                    np.isin(gt, [1, 2]),
                    np.isin(pred, [1, 2])
                ),

                "tumor": (
                    gt == 2,
                    pred == 2
                )
            }

            for name, (g, p) in structures.items():

                metrics = evaluate_structure(
                    g,
                    p,
                    spacing,
                    voxel_volume_ml
                )

                rows.append({
                    "fold": fold,
                    "case": case_id,
                    "structure": name,
                    **metrics
                })

            print(f"  {case_id}")

    # ========================================================
    # SAVE PER-CASE RESULTS
    # ========================================================

    df = pd.DataFrame(rows)

    per_case_file = (
        output_dir /
        "metrics_per_case.csv"
    )

    df.to_csv(
        per_case_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    metric_columns = [
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95_mm",
        "assd_mm"
    ]

    summary = (
        df
        .groupby("structure")[metric_columns]
        .agg([
            "mean",
            "std",
            "median",
            "count"
        ])
    )

    summary.to_csv(
        output_dir /
        "metrics_summary.csv"
    )

    # ========================================================
    # TUMOUR-POSITIVE CASES ONLY
    #
    # Important because some of your scans contain no tumour.
    # ========================================================

    positive = df[
        df["gt_present"]
    ]

    positive_summary = (
        positive
        .groupby("structure")[metric_columns]
        .agg([
            "mean",
            "std",
            "median",
            "count"
        ])
    )

    positive_summary.to_csv(
        output_dir /
        "metrics_summary_gt_positive.csv"
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        df.groupby("structure")[
            ["dice", "hd95_mm", "assd_mm"]
        ].mean()
    )

    print()
    print(f"Per-case results: {per_case_file}")
    print(
        f"Summary: "
        f"{output_dir / 'metrics_summary.csv'}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="Common dataset directory"
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction directory containing fold_0 ... fold_4"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory for evaluation results"
    )

    args = parser.parse_args()

    evaluate(
        args.dataset,
        args.predictions,
        args.output
    )