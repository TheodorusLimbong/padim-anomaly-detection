import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def compute_auroc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def compute_precision_recall_f1(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return precision, recall, f1


def compute_pixel_auroc(anomaly_maps, ground_truth_masks):
    """
    Compute pixel-level AUROC.

    anomaly_maps: list of [H, W] tensor/array per test image
    ground_truth_masks: list of [H, W] tensor/array per test image
    returns: pixel_auroc (float)
    """
    all_scores = []
    all_labels = []

    for a_map, gt_mask in zip(anomaly_maps, ground_truth_masks):
        a_map_np = (
            a_map.cpu().numpy().flatten()
            if torch.is_tensor(a_map)
            else np.array(a_map).flatten()
        )
        gt_np = (
            gt_mask.cpu().numpy().flatten()
            if torch.is_tensor(gt_mask)
            else np.array(gt_mask).flatten()
        )
        all_scores.extend(a_map_np)
        all_labels.extend(gt_np)

    return roc_auc_score(all_labels, all_scores)


def find_optimal_threshold(y_true, y_scores, method="percentile", percentile=95):
    """
    Find optimal threshold for anomaly detection.

    method:
      "percentile" — threshold = 95th percentile of normal scores (proposal 3.6.7)
      "f1" — threshold that maximizes F1-score on validation set
    """
    if method == "percentile":
        normal_scores = [s for s, l in zip(y_scores, y_true) if l == 0]
        if len(normal_scores) == 0:
            return float(np.percentile(y_scores, percentile))
        return float(np.percentile(normal_scores, percentile))

    elif method == "f1":
        best_thresh = 0.0
        best_f1 = 0.0
        for thresh in sorted(set(y_scores)):
            y_pred = [1 if s >= thresh else 0 for s in y_scores]
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return float(best_thresh)

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_image_level_metrics(y_true, y_scores, threshold):
    """
    Compute all image-level metrics: AUROC + Precision + Recall + F1.

    returns: dict with auroc, precision, recall, f1, threshold
    """
    auroc = compute_auroc(y_true, y_scores)
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred)

    return {
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }