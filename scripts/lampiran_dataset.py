# Cara jalankan: python scripts/lampiran_dataset.py
# Generate gambar grid 2x5 sampel dataset MVTec AD bottle untuk lampiran skripsi
# Output: output/figures/lampiran_dataset.png

import os
import matplotlib.pyplot as plt
from PIL import Image

BASE = "dataset/mvtec_anomaly_detection/bottle"
OUT = "output/figures/lampiran_dataset.png"

categories = {
    "Train Good":   os.path.join(BASE, "train/good"),
    "Test Good":    os.path.join(BASE, "test/good"),
    "Broken Large": os.path.join(BASE, "test/broken_large"),
    "Broken Small": os.path.join(BASE, "test/broken_small"),
    "Contamination": os.path.join(BASE, "test/contamination"),
}

N_SAMPLES = 2
fig, axes = plt.subplots(N_SAMPLES, len(categories),
                         figsize=(len(categories) * 2.5, N_SAMPLES * 2.5))

for col, (label, path) in enumerate(categories.items()):
    files = sorted(os.listdir(path))[:N_SAMPLES]
    for row, fname in enumerate(files):
        img = Image.open(os.path.join(path, fname)).convert("RGB")
        axes[row][col].imshow(img)
        axes[row][col].axis("off")
        if row == 0:
            axes[row][col].set_title(label, fontsize=10, fontweight="bold")
        if col == 0:
            axes[row][col].set_ylabel(fname, fontsize=7, rotation=0, labelpad=10, va="center")

plt.tight_layout(pad=1.0)
os.makedirs("output/figures", exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
