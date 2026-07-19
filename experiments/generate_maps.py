# Cara jalankan: python experiments/generate_maps.py
# Generate anomaly maps + GT masks untuk experiment run (subset)

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for d in ["prepocessing", "feature extractor", "anomaly detection", "evaluation"]:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d))

import torch
from torch.utils.data import DataLoader
from src.config import IMAGE_SIZE, BATCH_SIZE, KNN_K, DATASET_PATH, GAUSS_SIGMA
from dataset_wrapper import MVTecDataset
from inference import compute_padim_scores, compute_knn_scores

BASE = "output/experiments/subset"


def main():
    SUBSET_DIRS = sorted(os.listdir(BASE))

    dataset_root = os.path.dirname(DATASET_PATH)
    category = os.path.basename(DATASET_PATH)
    test_dataset = MVTecDataset(root_dir=dataset_root, category=category, phase="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_labels = []
    test_gt_masks = []
    for batch in test_loader:
        test_labels.extend(batch["label"].tolist())
        for mask_tensor in batch.get("mask", []):
            test_gt_masks.append(mask_tensor.squeeze(0))

    for subdir in SUBSET_DIRS:
        exp_dir = os.path.join(BASE, subdir)
        mpath = os.path.join(exp_dir, "metrics.json")
        if not os.path.exists(mpath):
            continue
        if not os.path.exists(os.path.join(exp_dir, "padim_stats.pt")):
            print(f"[SKIP] {subdir} — no padim_stats.pt")
            continue

        has_padim = os.path.exists(os.path.join(exp_dir, "padim_maps.pt"))
        has_knn = os.path.exists(os.path.join(exp_dir, "knn_maps.pt"))
        has_gt = os.path.exists(os.path.join(exp_dir, "gt_masks.pt"))
        if has_padim and has_knn and has_gt:
            print(f"[SKIP] {subdir} — all maps exist")
            continue

        print(f"[INFO] Generating maps for {subdir}...")
        test_patches_padim = torch.load(os.path.join(exp_dir, "test_features_padim.pt"), map_location="cpu")
        test_patches_knn = torch.load(os.path.join(exp_dir, "test_features_knn.pt"), map_location="cpu")
        stats = torch.load(os.path.join(exp_dir, "padim_stats.pt"), map_location="cpu")

        if not has_padim:
            print(f"  Generating PaDiM maps...")
            _, padim_maps = compute_padim_scores(
                test_patches_padim, stats["mean"], stats["cov_inv"], img_size=IMAGE_SIZE, sigma=GAUSS_SIGMA,
            )
            torch.save(padim_maps, os.path.join(exp_dir, "padim_maps.pt"))
            print(f"  PaDiM done: {len(padim_maps)} maps")
        else:
            print(f"  PaDiM maps exist, skipping")

        if not has_knn:
            feature_bank = torch.load(os.path.join(exp_dir, "feature_bank.pt"), map_location="cpu")
            print(f"  Generating KNN maps (feature bank: {feature_bank.shape})...")
            torch.cuda.empty_cache()
            _, knn_maps = compute_knn_scores(
                test_patches_knn, feature_bank, k=KNN_K, img_size=IMAGE_SIZE,
            )
            torch.save(knn_maps, os.path.join(exp_dir, "knn_maps.pt"))
            print(f"  KNN done: {len(knn_maps)} maps")
        else:
            print(f"  KNN maps exist, skipping")

        if not has_gt:
            torch.save(test_gt_masks, os.path.join(exp_dir, "gt_masks.pt"))
            print(f"  GT masks saved")
        else:
            print(f"  GT masks exist, skipping")

    print("[INFO] Done.")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
