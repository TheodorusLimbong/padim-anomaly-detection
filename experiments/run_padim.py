# Cara jalankan: python experiments/run_padim.py
# Pipeline end-to-end: preprocessing -> fitur -> dim reduction -> PaDiM statistik -> KNN bank -> inference -> metrik

import sys
import os
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# Setup sys.path
# Directory names contain spaces (e.g. "feature extractor"), jadi pakai sys.path manual
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
for d in ["prepocessing", "feature extractor", "anomaly detection", "evaluation"]:
    sys.path.insert(0, os.path.join(repo_root, d))

# Config: IMAGE_SIZE=224, PADIM_N_DIMS=550, KNN_K=5, GAUSS_SIGMA=4, SEED=1024
from src.config import (
    DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    SELECTED_LAYERS, DEVICE, SEED, PADIM_N_DIMS, KNN_K, GAUSS_SIGMA,
)
from dataset_wrapper import MVTecDataset         # Loader dataset MVTec AD
from models.backbone import ResNet50Backbone     # ResNet-50 + MoCo v2 weights
from models.hook_feature import FeatureExtractor # Forward hooks di layer1, layer2, layer3
from utils.feature_utils import reshape_embedding # [B,C,H,W] -> [B,H*W,C]
from padim import reduce_dim, compute_statistics, compute_cov_inv, save_statistics  # PaDiM training
from knn_baseline import build_feature_bank       # KNN: flatten patch ke feature bank
from inference import compute_padim_scores, compute_knn_scores  # Test inference
from metrics import (
    find_optimal_threshold,
    compute_image_level_metrics,
    compute_pixel_auroc,
    compute_pro_score,
)


def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def print_table(padim_metrics, knn_metrics):
    sep = "-" * 60
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'PaDiM':<15} {'KNN (K=' + str(KNN_K) + ')':<15}")
    print(sep)
    for key in padim_metrics:
        padim_val = padim_metrics[key]
        knn_val = knn_metrics[key]
        if isinstance(padim_val, (int, float)):
            print(f"{key:<20} {padim_val:<15.4f} {knn_val:<15.4f}")
        else:
            print(f"{key:<20} {str(padim_val):<15} {str(knn_val):<15}")
    print("=" * 60)


def run():
    # =============================
    # BAGIAN 0: SETUP
    # =============================
    # Bikin folder output dengan timestamp, simpan config snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(repo_root, "output", "experiments", f"run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"[INFO] Experiment directory: {exp_dir}")

    config_dict = {
        "dataset_path": DATASET_PATH,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "selected_layers": SELECTED_LAYERS,
        "padim_n_dims": PADIM_N_DIMS,
        "knn_k": KNN_K,
        "gauss_sigma": GAUSS_SIGMA,
        "device": DEVICE,
        "seed": SEED,
    }
    save_json(config_dict, os.path.join(exp_dir, "config.json"))

    device = torch.device(DEVICE)
    torch.manual_seed(SEED)
    dataset_root = os.path.dirname(DATASET_PATH)
    category = os.path.basename(DATASET_PATH)  # "bottle"

    # =============================
    # BAGIAN 1: FEATURE EXTRACTION
    # =============================
    # 209 gambar train -> backbone ResNet-50 + MoCo v2 -> [209, 1792, 56, 56]
    # 1792 = 256(layer1) + 512(layer2) + 1024(layer3) setelah concatenation
    # 56x56 = grid patch (224/4 = 56)

    print("[INFO] Loading ResNet-50 + MoCo v2 backbone (frozen)...")
    model = ResNet50Backbone(pretrained=True).to(device)
    model.eval()
    extractor = FeatureExtractor(model=model, selected_layers=SELECTED_LAYERS)

    # --- TRAIN FEATURES ---
    train_dataset = MVTecDataset(root_dir=dataset_root, category=category, phase="train")
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    print("[INFO] Extracting train features...")
    train_raw_list = []
    for batch in train_loader:
        images = batch["image"].to(device)
        embedding = extractor.extract(images)
        train_raw_list.append(embedding.cpu())

    train_raw = torch.cat(train_raw_list, dim=0)  # [209, 1792, 56, 56]
    torch.save(train_raw, os.path.join(exp_dir, "train_raw.pt"))
    print(f"  Shape: {train_raw.shape}")

    # --- DIMENSION REDUCTION (1792 -> 550) ---
    # PaDiM paper: random select ~550 channel dari 1792
    print(f"[INFO] Reducing dimensions: {train_raw.shape[1]} -> {PADIM_N_DIMS}")
    train_reduced, dim_indices = reduce_dim(train_raw, n_dims=PADIM_N_DIMS, seed=SEED)
    torch.save(dim_indices, os.path.join(exp_dir, "dim_indices.pt"))  # index channel terpilih
    print(f"  Reduced shape: {train_reduced.shape}")  # [209, 550, 56, 56]

    # --- PADIM STATISTICS ---
    # 3.136 posisi (56x56). Tiap posisi: 209 patch x 550 channel
    # -> hitung mean[550] + cov_inv[550x550] per posisi
    print("[INFO] Computing PaDiM statistics (mean + cov_inv)...")
    mean, cov = compute_statistics(train_reduced)
    cov_inv = compute_cov_inv(cov)
    save_statistics(mean, cov_inv, exp_dir, "padim_stats.pt", dim_indices)
    print(f"  Mean: {mean.shape}, Cov_inv: {cov_inv.shape}")  # [550,3136] dan [550,550,3136]

    # --- FEATURE BANK (KNN) ---
    # KNN perlu feature bank: tumpuk semua patch jadi [655424, 550]
    # 209 gambar x 3136 patch = 655.424 patch
    train_reduced_knn = train_raw[:, dim_indices, :, :].contiguous()
    train_patches_knn = reshape_embedding(train_reduced_knn)  # [209, 3136, 550]
    feature_bank = build_feature_bank(train_patches_knn)      # [655424, 550]
    torch.save(feature_bank, os.path.join(exp_dir, "feature_bank.pt"))
    print(f"  Feature bank: {feature_bank.shape}")

    # =============================
    # BAGIAN 1b: TEST FEATURES
    # =============================
    # 83 gambar test (9 normal + 74 anomali)
    test_dataset = MVTecDataset(root_dir=dataset_root, category=category, phase="test")
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    print("[INFO] Extracting test features...")
    test_raw_list = []
    test_labels = []          # 0=normal, 1=anomali (dari dataset)
    test_gt_masks = []        # Ground truth pixel-level untuk segmentasi

    for batch in test_loader:
        images = batch["image"].to(device)
        embedding = extractor.extract(images)
        test_raw_list.append(embedding.cpu())
        test_labels.extend(batch["label"].tolist())
        for mask_tensor in batch.get("mask", []):
            test_gt_masks.append(mask_tensor.squeeze(0))

    test_raw = torch.cat(test_raw_list, dim=0)  # [83, 1792, 56, 56]
    print(f"  Test raw shape: {test_raw.shape}")

    # Test features untuk PaDiM: reduksi 1792->550, reshape jadi [83, 3136, 550]
    test_reduced = test_raw[:, dim_indices, :, :].contiguous()
    test_patches_padim = reshape_embedding(test_reduced)
    torch.save(test_patches_padim, os.path.join(exp_dir, "test_features_padim.pt"))

    # Test features untuk KNN: reduksi 1792->550 (SAMA persis, biar fair)
    test_reduced_knn = test_raw[:, dim_indices, :, :].contiguous()
    test_patches_knn = reshape_embedding(test_reduced_knn).to(device)
    torch.save(test_patches_knn.cpu(), os.path.join(exp_dir, "test_features_knn.pt"))

    n_anom = sum(test_labels)
    print(f"  Labels: {n_anom} anomalies / {len(test_labels)} total")

    # =============================
    # BAGIAN 2: INFERENCE
    # =============================
    # PaDiM: Mahalanobis per patch -> upsample 56->224 -> Gaussian blur sigma=4 -> min-max norm
    print("[INFO] Running PaDiM inference...")
    padim_scores, padim_maps = compute_padim_scores(
        test_patches_padim, mean, cov_inv, img_size=IMAGE_SIZE, sigma=GAUSS_SIGMA,
    )
    torch.save(padim_scores, os.path.join(exp_dir, "padim_scores.pt"))
    print(f"  Scores shape: {padim_scores.shape}")  # [83]

    # KNN: Euclidean chunked -> ambil K=5 terkecil -> rata-rata
    print(f"[INFO] Running KNN (K={KNN_K}) inference...")
    knn_scores, knn_maps = compute_knn_scores(
        test_patches_knn, feature_bank, k=KNN_K, img_size=IMAGE_SIZE,
    )
    torch.save(knn_scores, os.path.join(exp_dir, "knn_scores.pt"))
    print(f"  Scores shape: {knn_scores.shape}")  # [83]

    # =============================
    # BAGIAN 3: EVALUASI
    # =============================
    # Threshold P95: ambil 95th percentile dari skor gambar normal
    # Bandingkan skor test ke threshold -> hitung AUROC, F1, Pixel AUROC, PRO-score

    print("[INFO] Computing metrics...")

    padim_thresh = find_optimal_threshold(
        test_labels, padim_scores.tolist(), method="percentile", percentile=95,
    )
    knn_thresh = find_optimal_threshold(
        test_labels, knn_scores.tolist(), method="percentile", percentile=95,
    )

    # Image-level: AUROC, Precision, Recall, F1
    padim_metrics = compute_image_level_metrics(
        test_labels, padim_scores.tolist(), padim_thresh,
    )
    knn_metrics = compute_image_level_metrics(
        test_labels, knn_scores.tolist(), knn_thresh,
    )

    # Pixel-level: Pixel AUROC + PRO-score (segmentasi)
    padim_metrics["pixel_auroc"] = compute_pixel_auroc(padim_maps, test_gt_masks)
    padim_metrics["pro_score"] = compute_pro_score(padim_maps, test_gt_masks)
    knn_metrics["pixel_auroc"] = compute_pixel_auroc(knn_maps, test_gt_masks)
    knn_metrics["pro_score"] = compute_pro_score(knn_maps, test_gt_masks)

    for d in (padim_metrics, knn_metrics):
        d["n_train"] = train_raw.shape[0]
        d["n_test"] = test_raw.shape[0]
        d["n_dims_padim"] = PADIM_N_DIMS
        d["n_dims_knn"] = PADIM_N_DIMS
        d["knn_k"] = KNN_K
        d["n_anomalies"] = n_anom
        d["category"] = category

    print_table(padim_metrics, knn_metrics)

    results = {
        "timestamp": timestamp,
        "exp_dir": exp_dir,
        "config": config_dict,
        "padim": padim_metrics,
        "knn": knn_metrics,
    }
    save_json(results, os.path.join(exp_dir, "metrics.json"))
    print(f"\n[INFO] Results saved to: {exp_dir}")
    print("[INFO] Done.")


if __name__ == "__main__":
    run()
