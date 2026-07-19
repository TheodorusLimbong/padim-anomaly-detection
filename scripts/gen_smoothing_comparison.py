"""
Generate Gambar 4.beta: Gaussian smoothing comparison.
Before (sigma=0) vs After (sigma=4).
Simple per-image normalization for both.
"""
import sys, os
sys.path.insert(0, r"D:\skripsi\self supervised\code")
for d in ["prepocessing", "anomaly detection", "evaluation"]:
    sys.path.insert(0, os.path.join(r"D:\skripsi\self supervised\code", d))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUTPUT_DIR = r"D:\skripsi\self supervised\code\output\figures"
EXP_DIR = r"D:\skripsi\self supervised\code\output\experiments\run_20260612_152608"
ROOT_DIR = r"D:\skripsi\self supervised\code\dataset\mvtec_anomaly_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.config import DATASET_PATH
from load_dataset import load_mvtec_paths
root_dir = os.path.dirname(DATASET_PATH)
category = os.path.basename(DATASET_PATH)
img_paths, labels, _ = load_mvtec_paths(root_dir, category, "test")

def get_defect_type(path):
    parts = path.replace("\\", "/").split("/")
    return parts[-2]

# Cari sample pertama setiap kelas berdasarkan defect type dari path
target_types = ["good", "broken_small", "broken_large", "contamination"]
class_names = ["Normal", "Broken Small", "Broken Large", "Contamination"]
sample_indices = []
for tt in target_types:
    for i, p in enumerate(img_paths):
        if get_defect_type(p) == tt:
            sample_indices.append(i)
            break
print(f"Sample indices: {sample_indices}")
print(f"Defect types: {[get_defect_type(img_paths[i]) for i in sample_indices]}")

def compute_map(features, mean, cov_inv_p, sigma, normalize=True):
    device = "cpu"
    H = W = 56

    delta = features.to(device) - mean.T.unsqueeze(0)
    patch_scores = torch.einsum("bpc,pcd,bpd->bp", delta, cov_inv_p, delta)
    patch_scores = torch.sqrt(patch_scores.clamp(min=0))

    a_map = patch_scores.view(-1, 1, H, W)
    a_map = F.interpolate(a_map, size=(224, 224), mode="bilinear", align_corners=False)

    if sigma > 0:
        ksize = int(2 * round(3 * sigma) + 1)
        x = torch.arange(-(ksize // 2), ksize // 2 + 1, device=device, dtype=torch.float32)
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).expand(1, 1, ksize, ksize)
        a_map = F.conv2d(a_map, kernel, padding=ksize // 2)

    arr = a_map.squeeze().numpy()
    if normalize:
        m_min, m_max = arr.min(), arr.max()
        if m_max > m_min:
            arr = (arr - m_min) / (m_max - m_min)
    return arr

print("Loading statistics...")
stats = torch.load(os.path.join(EXP_DIR, "padim_stats.pt"), map_location="cpu")
mean, cov_inv = stats["mean"], stats["cov_inv"]
device = "cpu"
cov_inv_p = cov_inv.permute(2, 0, 1).contiguous().to(device)
mean_d = mean.to(device)

print("Loading sample features...")
test_feat = torch.load(os.path.join(EXP_DIR, "test_features_padim.pt"), map_location="cpu")

# Compute global min/max dari seluruh 83 test images untuk σ=4 (sesuai dashboard)
print("Computing all 83 σ=4 maps for global normalization...")
all_raw_maps = []
batch_size = 16
for i in range(0, len(test_feat), batch_size):
    batch = test_feat[i:i+batch_size]
    raw = compute_map(batch, mean_d, cov_inv_p, sigma=4, normalize=False)
    if raw.ndim == 2:
        all_raw_maps.append(raw)
    else:
        for r in raw:
            all_raw_maps.append(r)
global_min = min(m.min() for m in all_raw_maps)
global_max = max(m.max() for m in all_raw_maps)
print(f"Global range: [{global_min:.6f}, {global_max:.6f}]")

print(f"Processing {len(sample_indices)} samples...")
all_before = []
all_after = []
all_imgs = []
all_gts = []

for idx in sample_indices:
    feat = test_feat[idx:idx+1]
    before = compute_map(feat, mean_d, cov_inv_p, sigma=0)
    after_raw = compute_map(feat, mean_d, cov_inv_p, sigma=4, normalize=False)
    after = (after_raw - global_min) / (global_max - global_min)
    all_before.append(before)
    all_after.append(after)

    path = img_paths[idx]
    img = np.array(Image.open(path).convert("RGB").resize((224, 224)))
    all_imgs.append(img)

    parts = path.replace("\\", "/").split("/")
    defect_type = parts[-2]
    filename = parts[-1].replace(".png", "_mask.png")
    mask_dir = os.path.join(root_dir, category, "ground_truth", defect_type)
    mask_path = os.path.join(mask_dir, filename)
    if os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = np.array(Image.fromarray(mask).resize((224, 224), Image.NEAREST))
    else:
        mask = np.zeros((224, 224), dtype=np.uint8)
    all_gts.append(mask)

print("Creating figure...")
n_rows = len(sample_indices)
fig, axes = plt.subplots(n_rows, 4, figsize=(14, 4 * n_rows))

col_labels = ["Original", "Before Smoothing", "After Smoothing", "Ground Truth"]

for row in range(n_rows):
    axes[row, 0].imshow(all_imgs[row])
    axes[row, 0].axis("off")
    # Label di pojok kiri atas masing-masing gambar original
    axes[row, 0].text(5, 15, class_names[row], fontsize=11, fontweight="bold",
                      color="white", bbox=dict(facecolor="black", alpha=0.6, pad=3),
                      va="top", ha="left")

    before_display = all_before[row] ** 0.35
    axes[row, 1].imshow(before_display, cmap="jet", vmin=0, vmax=1)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(all_after[row], cmap="jet", vmin=0, vmax=1)
    axes[row, 2].axis("off")

    axes[row, 3].imshow(all_gts[row], cmap="gray")
    axes[row, 3].axis("off")

for col, label in enumerate(col_labels):
    axes[0, col].set_title(label, fontsize=12, fontweight="bold", pad=10)

plt.suptitle("Perbandingan Anomaly Map Sebelum dan Sesudah Gaussian Smoothing",
             fontsize=13, fontweight="bold", y=0.98)
fig.text(0.5, 0.01,
    "Gambar 4.\u03b2 Perbandingan anomaly map sebelum smoothing (kiri) dan sesudah Gaussian smoothing "
    "\u03c3=4 (kanan) pada empat kelas: normal, broken small, broken large, dan contamination.",
    ha="center", va="bottom", fontsize=9, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
path = os.path.join(OUTPUT_DIR, "gaussian_smoothing_comparison.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[OK] {path}")
print("DONE")
