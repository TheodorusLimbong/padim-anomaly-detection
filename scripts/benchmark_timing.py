# Cara jalankan: python scripts/benchmark_timing.py
# Benchmark waktu inferensi PaDiM vs KNN (GPU + CPU) untuk 83 test images

# Cara jalankan: python scripts/benchmark_timing.py
# Benchmark waktu inferensi PaDiM vs KNN untuk 83 test images
# Mengukur: GPU (RTX 3060) + CPU (Ryzen 7 5800H)
# Output: timing_barchart.png di output/figures/

import sys, os, time, json
import numpy as np
from PIL import Image
from torchvision import transforms

REPO_ROOT = r"D:\skripsi\self supervised\code"
sys.path.insert(0, REPO_ROOT)
for d in ["prepocessing", "feature extractor", "anomaly detection"]:
    p = os.path.join(REPO_ROOT, d)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
from prepocessing.resize import get_resize_transform
from prepocessing.normalization import get_normalize_transform
from prepocessing.load_dataset import load_mvtec_paths
from models.backbone import ResNet50Backbone
from models.hook_feature import FeatureExtractor
from knn_baseline import compute_knn_anomaly_score
from src.config import SELECTED_LAYERS

EXP_DIR = os.path.join(REPO_ROOT, "output/experiments/run_20260612_152608")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output/figures")
N_BENCH = 83
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset/mvtec_anomaly_detection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("BENCHMARK TIMING — Dashboard-Accurate")
print("=" * 60)
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# ===== 1. LOAD EXPERIMENT DATA =====
print("\n[1] Loading experiment data...")
stats = torch.load(os.path.join(EXP_DIR, "padim_stats.pt"), map_location="cpu")
mean, cov_inv = stats["mean"], stats["cov_inv"]  # [550,3136], [550,550,3136]
dim_indices = torch.load(os.path.join(EXP_DIR, "dim_indices.pt"), map_location="cpu")  # [550]
feature_bank = torch.load(os.path.join(EXP_DIR, "feature_bank.pt"), map_location="cpu")  # [655424,550]
with open(os.path.join(EXP_DIR, "metrics.json")) as f:
    metrics = json.load(f)

print(f"  mean: {mean.shape}, cov_inv: {cov_inv.shape}")
print(f"  dim_indices: {dim_indices.shape}, feature_bank: {feature_bank.shape}")

sigma = metrics.get("config", {}).get("gauss_sigma", 4)
knn_k = metrics.get("config", {}).get("knn_k", 5)
print(f"  gauss_sigma={sigma}, knn_k={knn_k}")

# Pre-compute for KNN (same as dashboard utils.py:264)
feature_bank_norm_sq = (feature_bank ** 2).sum(dim=1)

# Pre-permute cov_inv (same as dashboard utils.py:271-272)
cov_inv_p = cov_inv.permute(2, 0, 1).contiguous()  # [3136, 550, 550]

# ===== 2. STRATIFIED SAMPLING =====
print("\n[2] Loading test image paths...")
img_paths, labels, _ = load_mvtec_paths(DATASET_ROOT, "bottle", "test")
print(f"  Total test images: {len(img_paths)}")

# Parse defect types from paths
defect_types = []
for p in img_paths:
    parts = p.replace("\\", "/").split("/")
    defect_types.append(parts[-2])

selected_indices = list(range(N_BENCH))
selected_paths = [img_paths[i] for i in selected_indices]
selected_labels = [labels[i] for i in selected_indices]
selected_defects = [defect_types[i] for i in selected_indices]

n_anom = sum(selected_labels)
print(f"  Using all {N_BENCH} images ({n_anom} anomaly, {N_BENCH - n_anom} normal)")

# ===== 3. HELPER FUNCTIONS (identical to dashboard) =====
_gauss_kernel_cache = {}

def _gaussian_kernel(sigma, device="cpu", dtype=torch.float32):
    key = (sigma, str(device), str(dtype))
    cached = _gauss_kernel_cache.get(key)
    if cached is not None:
        return cached
    kernel_size = int(2 * round(3 * sigma) + 1)
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=dtype)
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
    kernel = kernel_2d.expand(1, 1, kernel_size, kernel_size)
    _gauss_kernel_cache[key] = kernel
    return kernel

def _batch_mahalanobis(delta, cov_inv_p):
    """B=1: bmm (identical to dashboard utils.py:209-213)"""
    B, P, C = delta.shape
    d = delta.squeeze(0).unsqueeze(1)  # [P, 1, C]
    step1 = torch.bmm(d, cov_inv_p)    # [P,1,C] @ [P,C,C] → [P,1,C]
    step2 = torch.bmm(step1, d.permute(0, 2, 1))  # [P,1,C] @ [P,C,1] → [P,1,1]
    return step2.view(1, P)

# ===== 4. MAIN BENCHMARK LOOP =====
all_results = {}

for device_name, device in [("GPU", "cuda"), ("CPU", "cpu")]:
    if device_name == "GPU" and not torch.cuda.is_available():
        print(f"\n[SKIP] GPU not available")
        continue

    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK: {device_name}")
    print(f"{'=' * 60}")

    # 4a. Load model
    print(f"  Loading ResNet-50 + MoCo v2 on {device_name}...")
    t0 = time.perf_counter()
    model = ResNet50Backbone(pretrained=True).to(device)
    model.eval()
    extractor = FeatureExtractor(model=model, selected_layers=SELECTED_LAYERS)
    model_load_time = time.perf_counter() - t0
    print(f"  Model loaded in {model_load_time:.2f}s")

    # 4b. Move data to device
    mean_d = mean.to(device)
    cov_inv_p_d = cov_inv_p.to(device)
    dim_indices_d = dim_indices.to(device)

    # 4c. Warm-up (1 image, throwaway)
    print(f"  Warm-up (1 image)...")
    warm_img = Image.open(selected_paths[0]).convert("RGB")
    warm_tensor = transforms.Compose([
        get_resize_transform(224),
        transforms.ToTensor(),
        get_normalize_transform()
    ])(warm_img).unsqueeze(0).to(device)
    with torch.no_grad():
        warm_emb = extractor.extract(warm_tensor)
        warm_red = warm_emb[:, dim_indices_d, :, :].contiguous()
        b, c, h, w = warm_red.shape
        warm_patches = warm_red.permute(0, 2, 3, 1).reshape(b, h * w, c)
        # PaDiM warm
        delta = warm_patches - mean_d.T.unsqueeze(0)
        _ = _batch_mahalanobis(delta, cov_inv_p_d)
        # KNN warm
        _ = compute_knn_anomaly_score(warm_patches.squeeze(0), feature_bank,
                                      k=knn_k, chunk_size=30000,
                                      bank_norm_sq=feature_bank_norm_sq)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  Warm-up done.\n")

    # 4d. Main loop
    feat_times = []
    padim_times = []
    knn_times = []

    for i in range(N_BENCH):
        img_path = selected_paths[i]
        pil_img = Image.open(img_path).convert("RGB")

        # --- FEATURE EXTRACTION (same as dashboard preprocess_and_extract) ---
        t_feat = time.perf_counter()

        transform = transforms.Compose([
            get_resize_transform(224),
            transforms.ToTensor(),
            get_normalize_transform()
        ])
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = extractor.extract(img_tensor)  # [1, 1792, 56, 56]

        reduced = embedding[:, dim_indices_d, :, :].contiguous()  # [1, 550, 56, 56]
        b, c, h, w = reduced.shape
        patches = reduced.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [1, 3136, 550]

        if device == "cuda":
            torch.cuda.synchronize()
        feat_time = time.perf_counter() - t_feat
        feat_times.append(feat_time)

        # --- PADIM INFERENCE (same internal timer as infer_padim) ---
        t_padim = time.perf_counter()

        with torch.no_grad():
            delta = patches - mean_d.T.unsqueeze(0)  # [1, 3136, 550]
            patch_scores = _batch_mahalanobis(delta, cov_inv_p_d)  # [1, 3136]
            patch_scores = torch.sqrt(patch_scores.clamp(min=0))

            padim_map = patch_scores.view(1, 1, 56, 56)
            padim_map = F.interpolate(padim_map, size=(224, 224),
                                      mode="bilinear", align_corners=False)
            kernel = _gaussian_kernel(sigma, device=device)
            padim_map = F.conv2d(padim_map, kernel, padding=kernel.shape[-1] // 2)
            _ = padim_map.max().item()

        if device == "cuda":
            torch.cuda.synchronize()
        padim_time = time.perf_counter() - t_padim
        padim_times.append(padim_time)

        # --- KNN INFERENCE (same internal timer as infer_knn) ---
        t_knn = time.perf_counter()

        with torch.no_grad():
            patch_scores_knn = compute_knn_anomaly_score(
                patches.squeeze(0),
                feature_bank,
                k=knn_k,
                chunk_size=30000,
                bank_norm_sq=feature_bank_norm_sq,
            )
            knn_map = patch_scores_knn.view(1, 1, 56, 56)
            knn_map = F.interpolate(knn_map, size=(224, 224),
                                    mode="bilinear", align_corners=False)
            _ = knn_map.max().item()

        if device == "cuda":
            torch.cuda.synchronize()
        knn_time = time.perf_counter() - t_knn
        knn_times.append(knn_time)

        print(f"  [{device_name}] img {i+1}/{N_BENCH} ({selected_defects[i]:15s}): "
              f"feat={feat_time:.3f}s  padim={padim_time:.3f}s  knn={knn_time:.3f}s")

    # 4e. Compute stats
    feat_arr = np.array(feat_times)
    padim_arr = np.array(padim_times)
    knn_arr = np.array(knn_times)

    print(f"\n  {'=' * 55}")
    print(f"  SUMMARY: {device_name}")
    print(f"  {'=' * 55}")
    print(f"  {'Component':<25} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    print(f"  {'-' * 57}")
    print(f"  {'Feature Extraction':<25} {feat_arr.mean():>8.3f}s {feat_arr.min():>8.3f}s "
          f"{feat_arr.max():>8.3f}s {feat_arr.std():>8.3f}s")
    print(f"  {'PaDiM Inference':<25} {padim_arr.mean():>8.3f}s {padim_arr.min():>8.3f}s "
          f"{padim_arr.max():>8.3f}s {padim_arr.std():>8.3f}s")
    print(f"  {'KNN Inference':<25} {knn_arr.mean():>8.3f}s {knn_arr.min():>8.3f}s "
          f"{knn_arr.max():>8.3f}s {knn_arr.std():>8.3f}s")
    print(f"  {'─' * 57}")

    total_padim = (feat_arr + padim_arr).mean()
    total_knn = (feat_arr + knn_arr).mean()
    print(f"  {'PaDiM Total (feat+padim)':<25} {total_padim:>8.3f}s")
    print(f"  {'KNN Total (feat+knn)':<25} {total_knn:>8.3f}s")
    if total_padim > 0:
        print(f"  {'Speedup (KNN/PaDiM)':<25} {total_knn/total_padim:>8.2f}x")

    all_results[device_name] = {
        "feat": {"mean": float(feat_arr.mean()), "min": float(feat_arr.min()),
                 "max": float(feat_arr.max()), "std": float(feat_arr.std()),
                 "raw": [float(x) for x in feat_arr]},
        "padim": {"mean": float(padim_arr.mean()), "min": float(padim_arr.min()),
                  "max": float(padim_arr.max()), "std": float(padim_arr.std()),
                  "raw": [float(x) for x in padim_arr]},
        "knn": {"mean": float(knn_arr.mean()), "min": float(knn_arr.min()),
                "max": float(knn_arr.max()), "std": float(knn_arr.std()),
                "raw": [float(x) for x in knn_arr]},
        "total_padim": float(total_padim),
        "total_knn": float(total_knn),
        "speedup": float(total_knn / total_padim) if total_padim > 0 else 0,
    }

    # Cleanup
    del model, extractor, mean_d, cov_inv_p_d, dim_indices_d
    if device == "cuda":
        torch.cuda.empty_cache()

# ===== 5. SAVE RESULTS =====
out_path = os.path.join(OUTPUT_DIR, "benchmark_timing_results.txt")
with open(out_path, "w") as f:
    f.write(f"Benchmark Timing Results (Dashboard-Accurate)\n")
    f.write(f"{'=' * 60}\n")
    f.write(f"Experiment: run_20260612_152608\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Samples: {N_BENCH} (stratified: 2 broken_large, 2 broken_small, "
            f"4 contamination, 2 good)\n")
    if torch.cuda.is_available():
        f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
    f.write(f"{'=' * 60}\n\n")

    for dev_name, res in all_results.items():
        f.write(f"{'=' * 55}\n")
        f.write(f"SUMMARY: {dev_name}\n")
        f.write(f"{'=' * 55}\n")
        f.write(f"{'Component':<25} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}\n")
        f.write(f"{'-' * 57}\n")
        for comp in ["feat", "padim", "knn"]:
            label = {"feat": "Feature Extraction",
                     "padim": "PaDiM Inference",
                     "knn": "KNN Inference"}[comp]
            r = res[comp]
            f.write(f"{label:<25} {r['mean']:>8.3f}s {r['min']:>8.3f}s "
                    f"{r['max']:>8.3f}s {r['std']:>8.3f}s\n")
        f.write(f"{'─' * 57}\n")
        f.write(f"{'PaDiM Total (feat+padim)':<25} {res['total_padim']:>8.3f}s\n")
        f.write(f"{'KNN Total (feat+knn)':<25} {res['total_knn']:>8.3f}s\n")
        if res['speedup']:
            f.write(f"{'Speedup (KNN/PaDiM)':<25} {res['speedup']:>8.2f}x\n")
        f.write("\n")

        f.write(f"Per-image detail ({dev_name}):\n")
        f.write(f"{'img':>4} {'defect':>16} {'feat':>8} {'padim':>8} {'knn':>8}\n")
        f.write(f"{'-' * 48}\n")
        for i in range(N_BENCH):
            f.write(f"{i+1:>4} {selected_defects[i]:>16} "
                    f"{res['feat']['raw'][i]:>8.3f} "
                    f"{res['padim']['raw'][i]:>8.3f} "
                    f"{res['knn']['raw'][i]:>8.3f}\n")
        f.write("\n")

    f.write(f"{'=' * 60}\n")
    f.write("End of report\n")

print(f"\nResults saved to: {out_path}")
print("Done.")
