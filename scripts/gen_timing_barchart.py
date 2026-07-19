# Cara jalankan: python scripts/gen_timing_barchart.py
# Generate bar chart perbandingan waktu inferensi PaDiM vs KNN (GPU + CPU)
# Output: output/figures/timing_barchart.png (untuk BAB 4.6.2)

import matplotlib.pyplot as plt
import os

REPO_ROOT = r"D:\skripsi\self supervised\code"
OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data — total inference time per image (83 test images)
# GPU (RTX 3060), CPU (AMD Ryzen 7 5800H)
padim_times = {
    "GPU (RTX 3060)": 0.050,
    "CPU (Ryzen 7 5800H)": 0.639,
}
knn_times = {
    "GPU (RTX 3060)": 4.872,
    "CPU (Ryzen 7 5800H)": 20.195,
}

colors = ["#4ECDC4", "#FF9F43"]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# --- Left panel: PaDiM ---
ax = axes[0]
categories = list(padim_times.keys())
vals = list(padim_times.values())
bars = ax.barh(categories, vals, color=colors, height=0.5, edgecolor="white", linewidth=1.2)

for bar, val in zip(bars, vals):
    if val < 0.15:
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s", va="center", fontsize=10, fontweight="bold")
    else:
        ax.text(bar.get_width() - 0.04, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s", va="center", ha="right", fontsize=10, fontweight="bold", color="white")

ax.set_xlim(0, 0.8)
ax.set_title("PaDiM", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Waktu (detik)", fontsize=11)
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Right panel: KNN ---
ax = axes[1]
categories_k = list(knn_times.keys())
vals_k = list(knn_times.values())
bars = ax.barh(categories_k, vals_k, color=colors, height=0.5, edgecolor="white", linewidth=1.2)

for bar, val in zip(bars, vals_k):
    if val < 6:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s", va="center", fontsize=10, fontweight="bold")
    else:
        ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s", va="center", ha="right", fontsize=10, fontweight="bold", color="white")

ax.set_xlim(0, 24)
ax.set_title("KNN", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Waktu (detik)", fontsize=11)
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Perbandingan Waktu Inferensi per Gambar", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "timing_barchart.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
