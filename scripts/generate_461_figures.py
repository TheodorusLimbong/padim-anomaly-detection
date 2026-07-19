# Cara jalankan: python scripts/generate_461_figures.py
# Generate figures untuk subbab 4.6.1 Pengaruh Heatmap:
#   Gambar 4.beta  - Pipeline heatmap comparison
#   Gambar 4.gamma - Heatmap detail comparison (close-up crop)
#   Gambar 4.delta - Gap visualization (Pixel AUROC & PRO-score)

"""
import os, sys, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

OUTPUT_DIR = r"D:\skripsi\self supervised\code\output\figures"
EXP_DIR = r"D:\skripsi\self supervised\code\output\experiments\run_20260612_152608"
DEVICE = "cpu"

sys.path.insert(0, r"D:\skripsi\self supervised\code")
sys.path.insert(0, r"D:\skripsi\self supervised\code\anomaly detection")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Gambar 4.beta — Pipeline Heatmap Comparison Flowchart
# ==============================================================================
def draw_pipeline_fig():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors_padim = {"fill": "#4A90D9", "edge": "#2B6CB0", "text": "white"}
    colors_knn   = {"fill": "#E57373", "edge": "#C62828", "text": "white"}
    colors_shared = {"fill": "#E8E8E8", "edge": "#888888", "text": "#333333"}
    highlight = {"fill": "#FFD54F", "edge": "#F9A825", "text": "#333333"}

    def draw_pipeline(ax, colors, steps, title, highlight_idx=None):
        ax.set_xlim(0, 5.5)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        for i, (label, desc) in enumerate(steps):
            y = 7 - i * 1.3
            bw = 4.5 if i < 3 else 4.5
            bh = 0.7

            if highlight_idx is not None and i == highlight_idx:
                fc = highlight["fill"]
                ec = highlight["edge"]
                tc = highlight["text"]
                lw = 3
            else:
                fc = colors["fill"]
                ec = colors["edge"]
                tc = colors["text"]
                lw = 1.5

            box = mpatches.FancyBboxPatch(
                (0.5, y - bh / 2), bw, bh,
                boxstyle="round,pad=0.1",
                facecolor=fc, edgecolor=ec, linewidth=lw
            )
            ax.add_patch(box)

            # Arrow down (except last)
            if i < len(steps) - 1:
                ax.annotate("",
                    xy=(2.75, y - bh / 2), xytext=(2.75, y - bh / 2 - 0.6),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5)
                )

            # Text
            ax.text(2.75, y, f"{label}: {desc}",
                    ha="center", va="center", fontsize=10, color=tc,
                    fontweight="bold" if highlight_idx is not None and i == highlight_idx else "normal")

    steps_padim = [
        ("Input", "Patch Embedding [3136, 550]"),
        ("Step 1", "Mahalanobis Distance per patch"),
        ("Step 2", "Reshape ke [56, 56]"),
        ("Step 3", "Bilinear Upsample 56 → 224"),
        ("Step 4", "Gaussian Smoothing σ=4, kernel 25x25"),
        ("Step 5", "Global Min-Max Normalization"),
    ]
    steps_knn = [
        ("Input", "Patch Embedding [3136, 550]"),
        ("Step 1", "Euclidean Distance ke Feature Bank"),
        ("Step 2", "Reshape ke [56, 56]"),
        ("Step 3", "Bilinear Upsample 56 → 224"),
        ("Step 4", "Tanpa Gaussian Smoothing"),
        ("Step 5", "Tanpa Normalisasi"),
    ]

    draw_pipeline(axes[0], colors_padim, steps_padim, "PaDiM Pipeline", highlight_idx=4)
    draw_pipeline(axes[1], colors_knn,   steps_knn,   "KNN Pipeline",   highlight_idx=4)

    # Legend
    fig.text(0.5, 0.02,
        "Gambar 4.\u03b2 Perbandingan pipeline pembentukan heatmap antara PaDiM (kiri) dan KNN (kanan). "
        "Kotak kuning menandai perbedaan utama: Gaussian smoothing hanya diterapkan pada PaDiM.",
        ha="center", va="bottom", fontsize=9, fontstyle="italic")

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUTPUT_DIR, "pipeline_heatmap_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# ==============================================================================
# Gambar 4.gamma — Heatmap Detail Comparison (close-up crop)
# ==============================================================================
def generate_heatmap_crop():
    """Load test_features, run inference on 4 sample images, crop a detail area."""
    from inference import compute_padim_scores, compute_knn_scores
    from padim import load_statistics

    # Load data
    mean, cov_inv, _ = load_statistics(EXP_DIR)
    dim_indices = torch.load(os.path.join(EXP_DIR, "dim_indices.pt"), map_location="cpu")
    feature_bank = torch.load(os.path.join(EXP_DIR, "feature_bank.pt"), map_location="cpu")
    test_feat = torch.load(os.path.join(EXP_DIR, "test_features_padim.pt"), map_location="cpu")

    # Reduce KNN test features using same dim_indices
    knn_test_feat = test_feat.clone()

    # Load test images and labels to pick samples
    sys.path.insert(0, r"D:\skripsi\self supervised\code\prepocessing")
    from load_dataset import load_mvtec_paths
    from src.config import DATASET_PATH
    root_dir = os.path.dirname(DATASET_PATH)
    category = os.path.basename(DATASET_PATH)
    img_paths, labels, _ = load_mvtec_paths(root_dir, category, "test")

    # Pick 4 representative samples: good, broken_large, broken_small, contamination
    sample_indices = []
    for target_label in [0, 1, 2, 3]:
        for i, (p, l) in enumerate(zip(img_paths, labels)):
            if l == target_label and target_label not in [labels[si] for si in sample_indices]:
                sample_indices.append(i)
                break
    print(f"Sample indices: {sample_indices}")
    print(f"Labels: {[labels[i] for i in sample_indices]}")

    # Compute PaDiM scores for selected samples
    padim_scores, padim_maps = compute_padim_scores(
        test_feat[sample_indices], mean, cov_inv, img_size=224, sigma=4
    )

    # Compute KNN scores for selected samples
    knn_scores, knn_maps = compute_knn_scores(
        knn_test_feat[sample_indices], feature_bank, k=5, img_size=224
    )

    # Load ground truth masks
    gt_masks = []
    for idx in sample_indices:
        path = img_paths[idx]
        parts = path.replace("\\", "/").split("/")
        defect_type = parts[-2]
        filename = parts[-1].replace(".png", "_mask.png")
        mask_dir = os.path.join(root_dir, category, "ground_truth", defect_type)
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):
            from PIL import Image
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = np.array(Image.fromarray(mask).resize((224, 224), Image.NEAREST))
            gt_masks.append(mask)
        else:
            gt_masks.append(np.zeros((224, 224), dtype=np.uint8))

    # Create side-by-side comparison figure
    class_names = ["Good", "Broken Large", "Broken Small", "Contamination"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))

    for row in range(4):
        # Original image
        from PIL import Image
        img = np.array(Image.open(img_paths[sample_indices[row]]).convert("RGB").resize((224, 224)))
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{class_names[row]}\nOriginal", fontsize=10)
        axes[row, 0].axis("off")

        # PaDiM map with crop box indicator
        padim_map = padim_maps[row].numpy()
        axes[row, 1].imshow(padim_map, cmap="jet", vmin=0, vmax=1)
        axes[row, 1].set_title(f"PaDiM Heatmap\nScore: {padim_scores[row]:.4f}", fontsize=10)
        axes[row, 1].axis("off")

        # Draw crop box on PaDiM map
        if row > 0:  # anomaly samples
            crop_x, crop_y, crop_s = 70, 70, 50
            rect = mpatches.Rectangle((crop_x, crop_y), crop_s, crop_s,
                                       linewidth=2, edgecolor="white", facecolor="none", linestyle="--")
            axes[row, 1].add_patch(rect)

        # KNN map with crop box indicator
        knn_map = knn_maps[row].numpy()
        vmin_k, vmax_k = knn_map.min(), knn_map.max()
        knn_norm = (knn_map - vmin_k) / (vmax_k - vmin_k + 1e-8) if vmax_k > vmin_k else knn_map
        axes[row, 2].imshow(knn_norm, cmap="jet", vmin=0, vmax=1)
        axes[row, 2].set_title(f"KNN Heatmap\nScore: {knn_scores[row]:.4f}", fontsize=10)
        axes[row, 2].axis("off")

        if row > 0:
            rect2 = mpatches.Rectangle((crop_x, crop_y), crop_s, crop_s,
                                        linewidth=2, edgecolor="white", facecolor="none", linestyle="--")
            axes[row, 2].add_patch(rect2)

        # GT mask
        axes[row, 3].imshow(gt_masks[row], cmap="gray")
        axes[row, 3].set_title("Ground Truth Mask", fontsize=10)
        axes[row, 3].axis("off")

    plt.suptitle("Perbandingan Heatmap PaDiM vs KNN",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01,
        "Gambar 4.\u03b3 Perbandingan heatmap PaDiM (kolom 2) dan KNN (kolom 3) pada 4 sampel. "
        "Kotak putih putus-putus menandai region yang diperbesar untuk menunjukkan perbedaan smoothness.",
        ha="center", va="bottom", fontsize=9, fontstyle="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "heatmap_detail_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")

    # ====== CROP DETAIL FIGURE ======
    # Create a close-up comparison for one anomaly sample
    crop_fig, crop_axes = plt.subplots(2, 3, figsize=(10, 7))

    anomaly_row = 1  # broken large
    crop_x, crop_y, crop_s = 70, 70, 50

    # Create zoomed crops
    for row_offset, (map_data, method_name, cm) in enumerate([
        (padim_maps[anomaly_row].numpy(), "PaDiM", "jet"),
        (knn_norm if False else
         (lambda m: (m - m.min()) / (m.max() - m.min() + 1e-8))(knn_maps[anomaly_row].numpy()),
         "KNN", "jet")
    ]):
        pass

    # Actually create crops properly
    padim_crop = padim_maps[anomaly_row].numpy()[crop_y:crop_y+crop_s, crop_x:crop_x+crop_s]
    knn_raw = knn_maps[anomaly_row].numpy()
    knn_crop = knn_raw[crop_y:crop_y+crop_s, crop_x:crop_x+crop_s]
    # Normalize each crop for comparison
    padim_crop_norm = (padim_crop - padim_crop.min()) / (padim_crop.max() - padim_crop.min() + 1e-8)
    knn_crop_norm = (knn_crop - knn_crop.min()) / (knn_crop.max() - knn_crop.min() + 1e-8)
    gt_crop = gt_masks[anomaly_row][crop_y:crop_y+crop_s, crop_x:crop_x+crop_s]

    # Row 1: Full heatmaps with crop region
    crop_axes[0, 0].imshow(padim_maps[anomaly_row].numpy(), cmap="jet", vmin=0, vmax=1)
    rect_p = mpatches.Rectangle((crop_x, crop_y), crop_s, crop_s,
                                 linewidth=2, edgecolor="white", facecolor="none", linestyle="--")
    crop_axes[0, 0].add_patch(rect_p)
    crop_axes[0, 0].set_title("PaDiM Full Map", fontsize=10)
    crop_axes[0, 0].axis("off")

    knn_full = knn_maps[anomaly_row].numpy()
    knn_full_n = (knn_full - knn_full.min()) / (knn_full.max() - knn_full.min() + 1e-8)
    crop_axes[0, 1].imshow(knn_full_n, cmap="jet", vmin=0, vmax=1)
    rect_k = mpatches.Rectangle((crop_x, crop_y), crop_s, crop_s,
                                 linewidth=2, edgecolor="white", facecolor="none", linestyle="--")
    crop_axes[0, 1].add_patch(rect_k)
    crop_axes[0, 1].set_title("KNN Full Map", fontsize=10)
    crop_axes[0, 1].axis("off")

    crop_axes[0, 2].imshow(gt_masks[anomaly_row], cmap="gray")
    crop_axes[0, 2].set_title("Ground Truth", fontsize=10)
    crop_axes[0, 2].axis("off")

    # Row 2: Crops upsampled (detail comparison)
    from scipy.ndimage import zoom
    zoom_factor = 4
    padim_zoom = zoom(padim_crop_norm, zoom_factor, order=1)
    knn_zoom = zoom(knn_crop_norm, zoom_factor, order=1)
    gt_zoom = zoom(gt_crop.astype(float), zoom_factor, order=0)

    crop_axes[1, 0].imshow(padim_zoom, cmap="jet", vmin=0, vmax=1)
    crop_axes[1, 0].set_title("PaDiM Crop (4x zoom)", fontsize=10)
    crop_axes[1, 0].axis("off")

    crop_axes[1, 1].imshow(knn_zoom, cmap="jet", vmin=0, vmax=1)
    crop_axes[1, 1].set_title("KNN Crop (4x zoom)", fontsize=10)
    crop_axes[1, 1].axis("off")

    crop_axes[1, 2].imshow(gt_zoom, cmap="gray", vmin=0, vmax=1)
    crop_axes[1, 2].set_title("GT Crop (4x zoom)", fontsize=10)
    crop_axes[1, 2].axis("off")

    plt.suptitle("Detail Perbandingan Smoothness Heatmap — Broken Large",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01,
        "Perbesaran 4x pada region defect menunjukkan: PaDiM (kiri) memiliki gradien halus dan kontinu, "
        "sedangkan KNN (tengah) memperlihatkan artefak grid 56\u00d756 yang lebih jelas.",
        ha="center", va="bottom", fontsize=9, fontstyle="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    crop_path = os.path.join(OUTPUT_DIR, "heatmap_smoothness_detail.png")
    crop_fig.savefig(crop_path, dpi=200, bbox_inches="tight")
    crop_fig.close()
    print(f"[OK] {crop_path}")


# ==============================================================================
# Gambar 4.delta — Gap Visualization Bar Chart
# ==============================================================================
def draw_gap_chart():
    metrics = ["Pixel AUROC", "PRO-score"]
    padim_vals = [0.9838, 0.9525]
    knn_vals   = [0.9755, 0.9386]
    gaps       = [0.0083, 0.0139]

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))

    bars1 = ax.bar(x - width/2, padim_vals, width, label="PaDiM",
                   color="#4A90D9", edgecolor="#2B6CB0", linewidth=1)
    bars2 = ax.bar(x + width/2, knn_vals,   width, label="KNN",
                   color="#E57373", edgecolor="#C62828", linewidth=1)

    # Gap annotations
    for i, (p, k, g) in enumerate(zip(padim_vals, knn_vals, gaps)):
        y_max = max(p, k)
        ax.annotate(f"\u0394 = {g:.4f}",
                    xy=(i, y_max + 0.005),
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                    color="#D32F2F",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#FFEBEE", edgecolor="#D32F2F"))

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                f"{h:.4f}", ha="center", va="bottom", fontsize=9, color="#2B6CB0")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                f"{h:.4f}", ha="center", va="bottom", fontsize=9, color="#C62828")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0.9, 1.02)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add reference line at 1.0
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    fig.text(0.5, -0.02,
        "Gambar 4.\u03b4 Perbandingan metrik segmentasi PaDiM vs KNN. Gap PRO-score (0,0139) "
        "lebih besar dari gap Pixel AUROC (0,0083), mengindikasikan keunggulan PaDiM dalam deteksi region defect utuh.",
        ha="center", va="top", fontsize=9, fontstyle="italic")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gap_segmentation_metrics.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("=== Generating figures for 4.6.1 Pengaruh Heatmap ===\n")
    draw_pipeline_fig()
    draw_gap_chart()
    # try:
    #     generate_heatmap_crop()
    # except Exception as e:
    #     print(f"[WARN] Heatmap crop generation failed: {e}")
    #     print("[INFO] Falling back to generating only pipeline and gap chart")
    print("\n=== Done ===")
