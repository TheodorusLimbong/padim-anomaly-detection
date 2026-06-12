import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from skimage.measure import label


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


def compute_pro_score(anomaly_maps, ground_truth_masks, max_fpr=0.3, num_thresholds=500):
    """
    Compute PRO-score (Per-Region Overlap) as described in the PaDiM paper.

    PRO is the average per-region overlap (recall per connected component),
    integrated over FPR from 0 to max_fpr. Higher is better.

    anomaly_maps: list of [H, W] tensor/array per test image
    ground_truth_masks: list of [H, W] tensor/array per test image (0=normal, 1=anomaly)
    max_fpr: maximum FPR for integration (default 0.3, per PaDiM paper)
    num_thresholds: number of thresholds for evaluation (default 500)
    returns: pro_score (float)
    """
    maps = []
    masks = []
    for am, gt in zip(anomaly_maps, ground_truth_masks):
        if torch.is_tensor(am):
            maps.append(am.cpu().numpy())
        else:
            maps.append(np.array(am))
        if torch.is_tensor(gt):
            masks.append((gt.cpu().numpy() > 0).astype(np.int32))
        else:
            masks.append((np.array(gt) > 0).astype(np.int32))

    # Extract all connected components from ground truth masks
    comps = []
    for img_idx, mask in enumerate(masks):
        labeled, n = label(mask, return_num=True, connectivity=2)
        for cid in range(1, n + 1):
            comp_mask = (labeled == cid)
            comps.append((img_idx, comp_mask, comp_mask.sum()))

    if len(comps) == 0:
        return 1.0  # No ground-truth anomalies means perfect score

    # Threshold range (descending: strict -> relaxed)
    all_scores = np.concatenate([m.ravel() for m in maps])
    thresholds = np.linspace(all_scores.max(), all_scores.min(), num_thresholds)

    fprs = np.zeros(num_thresholds)
    pros = np.zeros(num_thresholds)

    for i, thresh in enumerate(thresholds):
        binary_maps = [(m >= thresh) for m in maps]

        # Per-component recall
        recalls = np.array([
            binary_maps[img_idx][comp_mask].sum() / area
            for img_idx, comp_mask, area in comps
        ])
        pros[i] = recalls.mean()

        # Pixel-level false positive rate
        fp_total = 0
        tn_total = 0
        for img_idx in range(len(maps)):
            normal = (masks[img_idx] == 0)
            fp_total += binary_maps[img_idx][normal].sum()
            tn_total += normal.sum() - binary_maps[img_idx][normal].sum()
        fprs[i] = fp_total / (fp_total + tn_total) if (fp_total + tn_total) > 0 else 0.0

    # Sort by FPR
    sort_idx = np.argsort(fprs)
    fprs = fprs[sort_idx]
    pros = pros[sort_idx]

    # Interpolate PRO at evenly spaced FPR targets from 0 to max_fpr
    target_fprs = np.linspace(0, max_fpr, 100)
    pro_interp = np.interp(target_fprs, fprs, pros, left=0.0, right=pros[-1])

    return float(pro_interp.mean())


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