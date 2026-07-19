"""
Generate heatmap crop comparison figure (Gambar 4.gamma) for BAB 4.6.1.
Loads 4 sample images from test set, computes anomaly maps via PaDiM and KNN,
then creates a side-by-side comparison with crop detail.
"""
import sys, os
sys.path.insert(0, r"D:\skripsi\self supervised\code")
for d in ["prepocessing", "feature extractor", "anomaly detection", "evaluation"]:
    sys.path.insert(0, os.path.join(r"D:\skripsi\self supervised\code", d))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

OUTPUT_DIR = r"D:\skripsi\self supervised\code\output\figures"
EXP_DIR = r"D:\skripsi\self supervised\code\output\experiments\run_20260612_152608"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load statistics
print("Loading statistics...")
stats = torch.load(os.path.join(EXP_DIR, "padim_stats.pt"), map_location="cpu")
mean = stats["mean"]
cov_inv = stats["cov_inv"]
dim_indices = torch.load(os.path.join(EXP_DIR, "dim_indices.pt"), map_location="cpu")
feature_bank = torch.load(os.path.join(EXP_DIR, "feature_bank.pt"), map_location="cpu")

# Load 4 test samples
# Indices: 0=good, 16=broken_large, 42=broken_small, 62=contamination
sample_indices = [0, 16, 42, 62]
class_names = ["Good", "Broken Large", "Broken Small", "Contamination"]

# Load test features
print("Loading test features...")
test_feat = torch.load(os.path.join(EXP_DIR, "test_features_padim.pt"), map_location="cpu")
selected_feat = test_feat[sample_indices]

# Load test image paths
from src.config import DATASET_PATH
from load_dataset import load_mvtec_paths
root_dir = os.path.dirname(DATASET_PATH)
category = os.path.basename(DATASET_PATH)
img_paths, labels, _ = load_mvtec_paths(root_dir, category, "test")

# Compute PaDiM scores for 4 samples
print("Computing PaDiM scores...")
from inference import compute_padim_scores, compute_knn_scores
padim_scores, padim_maps = compute_padim_scores(
    selected_feat, mean, cov_inv, img_size=224, sigma=4
)

# Compute KNN scores for 4 samples
print("Computing KNN scores...")
knn_scores, knn_maps = compute_knn_scores(
    selected_feat, feature_bank, k=5, img_size=224
)

# Load GT masks
gt_masks = []
for idx in sample_indices:
    path = img_paths[idx]
    parts = path.replace("\\", "/").split("/")
    defect_type = parts[-2]
    filename = parts[-1].replace(".png", "_mask.png")
    mask_dir = os.path.join(root_dir, category, "ground_truth", defect_type)
    mask_path = os.path.join(mask_dir, filename)
    if os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = np.array(Image.fromarray(mask).resize((224, 224), Image.NEAREST))
        gt_masks.append(mask)
    else:
        gt_masks.append(np.zeros((224, 224), dtype=np.uint8))

# ===== MAIN FIGURE: 4x4 grid =====
print("Creating main figure...")
fig, axes = plt.subplots(4, 4, figsize=(14, 14))

for row in range(4):
    # Original image
    img = np.array(Image.open(img_paths[sample_indices[row]]).convert("RGB").resize((224, 224)))
    axes[row, 0].imshow(img)
    axes[row, 0].set_title(f"{class_names[row]}", fontsize=11, fontweight="bold")
    axes[row, 0].axis("off")

    # PaDiM
    axes[row, 1].imshow(padim_maps[row].numpy(), cmap="jet", vmin=0, vmax=1)
    axes[row, 1].set_title(f"PaDiM\nScore: {padim_scores[row]:.4f}", fontsize=10)
    axes[row, 1].axis("off")

    # KNN (per-image normalize for display)
    knm = knn_maps[row].numpy()
    vmin_k, vmax_k = knm.min(), knm.max()
    knm_norm = (knm - vmin_k) / (vmax_k - vmin_k + 1e-8) if vmax_k > vmin_k else knm
    axes[row, 2].imshow(knm_norm, cmap="jet", vmin=0, vmax=1)
    axes[row, 2].set_title(f"KNN\nScore: {knn_scores[row]:.4f}", fontsize=10)
    axes[row, 2].axis("off")

    # GT
    axes[row, 3].imshow(gt_masks[row], cmap="gray")
    axes[row, 3].set_title("Ground Truth", fontsize=11, fontweight="bold")
    axes[row, 3].axis("off")
    
    # Add crop box on anomaly maps for rows 1-3
    if row > 0:
        cx, cy, cs = 70, 70, 50
        for col in [1, 2]:
            rect = mpatches.Rectangle((cx, cy), cs, cs,
                linewidth=2, edgecolor="white", facecolor="none", linestyle="--")
            axes[row, col].add_patch(rect)

# Column labels
for col, label in enumerate(["Original", "PaDiM", "KNN", "Ground Truth"]):
    axes[0, col].set_title(label, fontsize=12, fontweight="bold", pad=15)

plt.suptitle("Perbandingan Heatmap PaDiM vs KNN",
             fontsize=14, fontweight="bold", y=0.98)
fig.text(0.5, 0.01,
    "Gambar 4.\u03b3 Perbandingan heatmap pada 4 sampel uji. Kolom 1: citra asli, "
    "Kolom 2: heatmap PaDiM (Mahalanobis + Gauss \u03c3=4), Kolom 3: heatmap KNN (Euclidean, tanpa smoothing), "
    "Kolom 4: ground truth mask. Kotak putih putus-putus = region crop untuk detail smoothness.",
    ha="center", va="bottom", fontsize=9, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
path1 = os.path.join(OUTPUT_DIR, "heatmap_detail_comparison.png")
plt.savefig(path1, dpi=200, bbox_inches="tight")
plt.close()
print(f"[OK] {path1}")

# ===== CROP DETAIL FIGURE =====
print("Creating crop detail figure...")
from scipy.ndimage import zoom

crop_fig, crop_axes = plt.subplots(2, 3, figsize=(10, 7))
cx, cy, cs = 70, 70, 50
anomaly_row = 1  # broken large

# Row 1: Full maps with crop region
crop_axes[0, 0].imshow(padim_maps[anomaly_row].numpy(), cmap="jet", vmin=0, vmax=1)
rect = mpatches.Rectangle((cx, cy), cs, cs, linewidth=2, edgecolor="white", facecolor="none", linestyle="--")
crop_axes[0, 0].add_patch(rect)
crop_axes[0, 0].set_title("PaDiM Full Map", fontsize=10)
crop_axes[0, 0].axis("off")

kn_full = knn_maps[anomaly_row].numpy()
vmin_k, vmax_k = kn_full.min(), kn_full.max()
kn_full_n = (kn_full - vmin_k) / (vmax_k - vmin_k + 1e-8) if vmax_k > vmin_k else kn_full
crop_axes[0, 1].imshow(kn_full_n, cmap="jet", vmin=0, vmax=1)
rect = mpatches.Rectangle((cx, cy), cs, cs, linewidth=2, edgecolor="white", facecolor="none", linestyle="--")
crop_axes[0, 1].add_patch(rect)
crop_axes[0, 1].set_title("KNN Full Map", fontsize=10)
crop_axes[0, 1].axis("off")

crop_axes[0, 2].imshow(gt_masks[anomaly_row], cmap="gray")
crop_axes[0, 2].set_title("Ground Truth", fontsize=10)
crop_axes[0, 2].axis("off")

# Row 2: 4x zoom crops
zf = 4
padim_crop = padim_maps[anomaly_row].numpy()[cy:cy+cs, cx:cx+cs]
kn_crop = kn_full_n[cy:cy+cs, cx:cx+cs]
gt_crop = gt_masks[anomaly_row][cy:cy+cs, cx:cx+cs].astype(float)

# Normalize crop for display
pc_norm = (padim_crop - padim_crop.min()) / (padim_crop.max() - padim_crop.min() + 1e-8)
kc_norm = (kn_crop - kn_crop.min()) / (kn_crop.max() - kn_crop.min() + 1e-8)

pc_zoom = zoom(pc_norm, zf, order=1)
kc_zoom = zoom(kc_norm, zf, order=1)
gt_zoom = zoom(gt_crop, zf, order=0)

crop_axes[1, 0].imshow(pc_zoom, cmap="jet", vmin=0, vmax=1)
crop_axes[1, 0].set_title("PaDiM Crop (4x zoom)", fontsize=10)
crop_axes[1, 0].axis("off")

crop_axes[1, 1].imshow(kc_zoom, cmap="jet", vmin=0, vmax=1)
crop_axes[1, 1].set_title("KNN Crop (4x zoom)", fontsize=10)
crop_axes[1, 1].axis("off")

crop_axes[1, 2].imshow(gt_zoom, cmap="gray")
crop_axes[1, 2].set_title("GT Crop (4x zoom)", fontsize=10)
crop_axes[1, 2].axis("off")

plt.suptitle("Detail Smoothness Heatmap — Broken Large",
             fontsize=12, fontweight="bold", y=0.98)
crop_fig.text(0.5, 0.01,
    "Perbesaran 4x pada region defect: PaDiM (kiri) memiliki gradien halus dan kontinu akibat Gaussian smoothing, "
    "sedangkan KNN (tengah) memperlihatkan tekstur lebih kasar dan artefak grid 56\u00d756.",
    ha="center", va="bottom", fontsize=9, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
path2 = os.path.join(OUTPUT_DIR, "heatmap_smoothness_detail.png")
plt.savefig(path2, dpi=200, bbox_inches="tight")
plt.close()
print(f"[OK] {path2}")

print("\n=== DONE ===")
