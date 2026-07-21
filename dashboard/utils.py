# Cara jalankan: (di-import oleh dashboard/app.py)
# Utilitas dashboard: load_model(backbone), load_experiment_data(mean+cov+feature_bank),
# infer_single_image(PaDiM + KNN sequential), normalisasi map, kernel cache

import os, json, sys
import torch
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepocessing.load_dataset import load_mvtec_paths
from src.config import DATASET_PATH, IMAGE_SIZE
from torchvision import transforms

SUBSET_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output/experiments/subset"
)

METRIC_LABELS = {
    "auroc": "AUROC",
    "f1": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
    "pixel_auroc": "Pixel AUROC",
    "pro_score": "PRO-score",
}

def load_all_metrics():
    rows = []
    for subdir in sorted(os.listdir(SUBSET_BASE)):
        mpath = os.path.join(SUBSET_BASE, subdir, "metrics.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        n_train = cfg.get("n_train", data["padim"]["n_train"])
        aug = cfg.get("augmentation", "aug" in subdir)
        row = {"experiment": subdir, "n_train": n_train, "augmentation": aug}
        for method, prefix in [("padim", "PaDiM"), ("knn", "KNN")]:
            m = data.get(method, {})
            for key in ["auroc", "f1", "precision", "recall", "pixel_auroc", "pro_score"]:
                val = m.get(key)
                if val is not None:
                    row[f"{prefix}_{key}"] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ("experiment", "augmentation", "n_train"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, {r["experiment"]: r for r in rows}


def get_experiment_path(name):
    return os.path.join(SUBSET_BASE, name)


def load_anomaly_maps(exp_name):
    exp_dir = get_experiment_path(exp_name)
    result = {}
    for key, fname in [("padim", "padim_maps.pt"), ("knn", "knn_maps.pt"), ("gt", "gt_masks.pt")]:
        path = os.path.join(exp_dir, fname)
        if os.path.exists(path):
            result[key] = torch.load(path, map_location="cpu")
        else:
            result[key] = None
    return result


def load_scores(exp_name):
    exp_dir = get_experiment_path(exp_name)
    result = {}
    for key, fname in [("padim", "padim_scores.pt"), ("knn", "knn_scores.pt")]:
        path = os.path.join(exp_dir, fname)
        if os.path.exists(path):
            result[key] = torch.load(path, map_location="cpu")
        else:
            result[key] = None
    return result


def get_test_image_paths():
    root_dir = os.path.dirname(DATASET_PATH)
    category = os.path.basename(DATASET_PATH)
    img_paths, labels, _ = load_mvtec_paths(root_dir, category, "test")
    return img_paths, labels


def load_test_images():
    img_paths, labels = get_test_image_paths()
    images = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        images.append(img)
    label_names = ["good" if l == 0 else "anomaly" for l in labels]
    return images, labels, label_names, img_paths


def get_image_info():
    img_paths, labels = get_test_image_paths()
    info = []
    for i, (p, l) in enumerate(zip(img_paths, labels)):
        parts = p.replace("\\", "/").split("/")
        defect_type = parts[-2] if len(parts) >= 2 else "unknown"
        info.append({"idx": i, "label": l, "defect": defect_type, "path": p})
    return info


def tensor_to_img(t):
    if isinstance(t, torch.Tensor):
        if t.dim() == 3 and t.shape[0] == 3:
            return t.permute(1, 2, 0).cpu().numpy()
        return t.cpu().numpy()
    return np.array(t)


def normalize_map(t):
    arr = np.array(t, dtype=np.float32)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def build_summary_table(metrics_dict):
    rows = []
    for exp_name, m in sorted(metrics_dict.items()):
        n = m["n_train"]
        aug = "Yes" if m["augmentation"] else "No"
        rows.append({
            "Experiment": exp_name,
            "N Train": n,
            "Aug": aug,
            "PaDiM AUROC": f"{m.get('PaDiM_auroc', 0):.4f}",
            "KNN AUROC": f"{m.get('KNN_auroc', 0):.4f}",
            "Gap": f"{m.get('PaDiM_auroc', 0) - m.get('KNN_auroc', 0):.4f}",
            "PaDiM F1": f"{m.get('PaDiM_f1', 0):.4f}",
            "KNN F1": f"{m.get('KNN_f1', 0):.4f}",
            "PaDiM PRO": f"{m.get('PaDiM_pro_score', 0):.4f}",
            "KNN PRO": f"{m.get('KNN_pro_score', 0):.4f}",
        })
    return pd.DataFrame(rows)


def experiment_has_maps(exp_name):
    exp_dir = get_experiment_path(exp_name)
    return (
        os.path.exists(os.path.join(exp_dir, "padim_maps.pt")) and
        os.path.exists(os.path.join(exp_dir, "knn_maps.pt"))
    )


def compute_roc_curve(labels, scores, n_thresh=1000):
    scores = np.array(scores)
    labels = np.array(labels)
    thresholds = np.linspace(scores.min(), scores.max(), n_thresh + 1)
    tpr = np.zeros(n_thresh + 1)
    fpr = np.zeros(n_thresh + 1)
    for i, t in enumerate(thresholds):
        pred = scores >= t
        tp = (pred & (labels == 1)).sum()
        fp = (pred & (labels == 0)).sum()
        fn = (~pred & (labels == 1)).sum()
        tn = (~pred & (labels == 0)).sum()
        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0
    return fpr, tpr, thresholds


# ============================================================
# LIVE INFERENCE — Dashboard Utils
# ============================================================
# Fungsi-fungsi di bawah ini dipanggil oleh dashboard/app.py secara realtime.
# Alur: upload gambar -> preprocess -> feature extraction -> PaDiM + KNN -> display
import time as _time

EXPERIMENTS_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "experiments"
)


_gauss_kernel_cache = {}

def _gaussian_kernel(sigma, kernel_size=None, device="cpu", dtype=torch.float32):
    """Buat Gaussian kernel 2D dengan caching (biar gak bikin ulang tiap gambar)."""
    if kernel_size is None:
        kernel_size = int(2 * round(3 * sigma) + 1)
    key = (sigma, kernel_size, str(device), str(dtype))
    cached = _gauss_kernel_cache.get(key)
    if cached is not None:
        return cached
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=dtype)
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
    kernel = kernel_2d.expand(1, 1, kernel_size, kernel_size)
    _gauss_kernel_cache[key] = kernel
    return kernel


def _batch_mahalanobis(delta, cov_inv_p):
    """
    Mahalanobis batch pakai bmatmul (lebih cepat dari einsum di CPU).
    
    Rumus: delta @ cov_inv_p @ delta.T
    
    B=1 (di dashboard): pakai bmm murni, optimal utk CPU (MKL parallel batch 3136)
    B>1 (di batch experiment): fallback ke einsum biar gak OOM

    delta: [B, 3136, 550] = patch test - mean
    cov_inv_p: [3136, 550, 550] = cov_inv sudah dipermute
    returns: [B, 3136] = Mahalanobis squared distance per patch
    """
    B, P, C = delta.shape
    if B == 1:
        d = delta.squeeze(0).unsqueeze(1)  # [3136, 1, 550]
        step1 = torch.bmm(d, cov_inv_p)    # [3136,1,550] @ [3136,550,550] -> [3136,1,550]
        step2 = torch.bmm(step1, d.permute(0, 2, 1))  # [3136,1,550] @ [3136,550,1] -> [3136,1,1]
        return step2.view(1, P)
    else:
        return torch.einsum("bpc,pcd,bpd->bp", delta, cov_inv_p, delta)


def _nearest_neighbor_dist(x, bank, bank_norm_sq):
    """
    Euclidean distance ke nearest neighbor di bank.
    x: [1, C] — embedding test
    bank: [N, C] — N embedding reference
    bank_norm_sq: [N] — precomputed ||b||^2
    Returns: scalar float — jarak Euclidean minimum
    """
    x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
    dots = x @ bank.T
    dist_sq = x_norm_sq + bank_norm_sq.unsqueeze(0) - 2 * dots
    return dist_sq.min().sqrt().item()


def check_bottle_ood(patches, exp_data, device):
    """
    ComboOOD score: gabung Mahalanobis mean MD² + KNN mean emb 1-NN.

    score = kc + mc
    score >= ood_threshold -> bottle (in-distribution)
    score < ood_threshold  -> non-bottle (OOD)

    patches: [1, 3136, 550]
    exp_data: dict dari load_experiment_data()
    device: "cuda" atau "cpu"

    Returns:
        is_bottle: True jika combo_score >= ood_threshold
        combo_score: kc + mc
        threshold: batas OOD
    """
    import math

    ood_threshold = exp_data.get("ood_threshold")
    if ood_threshold is None:
        return True, 0.0, None

    with torch.no_grad():
        # Mahalanobis: mean squared MD dari 3136 patch
        delta = patches - exp_data["mean"].T.unsqueeze(0)
        md_sq = _batch_mahalanobis(delta, exp_data["cov_inv_p"])
        mean_md_sq = md_sq.mean().item()
        mc = -0.5 * mean_md_sq

        # KNN: mean pooled embedding -> 1-NN ke 209 training mean embeddings
        mean_emb = patches.mean(dim=1)
        kd = _nearest_neighbor_dist(
            mean_emb,
            exp_data["ood_feature_bank"],
            exp_data["ood_feature_bank_norm_sq"],
        )
        kc = -math.sqrt(550) * math.log(kd + 1e-8)

        combo_score = kc + mc

    is_bottle = combo_score >= ood_threshold
    return is_bottle, combo_score, ood_threshold


def _setup_paths():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ["prepocessing", "feature extractor", "anomaly detection"]:
        p = os.path.join(repo_root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def list_experiments():
    """Daftar semua experiment folder di output/experiments/, diurutkan dari terbaru."""
    rows = []
    if not os.path.exists(EXPERIMENTS_BASE):
        return rows
    for d in sorted(os.listdir(EXPERIMENTS_BASE), reverse=True):
        mpath = os.path.join(EXPERIMENTS_BASE, d, "metrics.json")
        if os.path.isfile(mpath):
            with open(mpath) as f:
                rows.append((d, json.load(f)))
    return rows


_model_instance = None
_extractor_instance = None

def load_model(device):
    """
    Load ResNet-50 + MoCo v2 backbone (sekali, di-cache).
    Dipanggil pas startup dashboard, gak perlu diulang tiap upload gambar.
    """
    global _model_instance, _extractor_instance
    if _model_instance is not None:
        return _model_instance, _extractor_instance
    _setup_paths()
    from models.backbone import ResNet50Backbone
    from models.hook_feature import FeatureExtractor
    from src.config import SELECTED_LAYERS
    _model_instance = ResNet50Backbone(pretrained=True).to(device)
    _model_instance.eval()
    _extractor_instance = FeatureExtractor(model=_model_instance, selected_layers=SELECTED_LAYERS)
    return _model_instance, _extractor_instance


def load_experiment_data(exp_name, device):
    """
    Load semua data yang diperlukan untuk inference dari folder experiment.
    
    Yang di-load:
    - PaDiM: mean [550,3136] + cov_inv [550,550,3136] (sudah dipermute ke [3136,550,550])
    - KNN: feature_bank [655424,550] + norm_sq [655424] (precomputed biar cepet)
    - Dim indices [550] (channel yang dipilih pas dim reduction)
    - Threshold PaDiM + KNN dari metrics.json
    - Global pixel min/max untuk normalisasi map (dihitung ulang dari 83 test images)
    - Gaussian sigma
    """
    exp_dir = os.path.join(EXPERIMENTS_BASE, exp_name)
    _setup_paths()
    from padim import load_statistics
    import torch.nn.functional as F

    mean, cov_inv, _ = load_statistics(exp_dir)
    dim_indices = torch.load(os.path.join(exp_dir, "dim_indices.pt"), map_location="cpu")
    feature_bank = torch.load(os.path.join(exp_dir, "feature_bank.pt"), map_location="cpu")
    feature_bank_norm_sq = (feature_bank ** 2).sum(dim=1)  # precompute ||b||^2

    with open(os.path.join(exp_dir, "metrics.json")) as f:
        metrics = json.load(f)

    sigma = metrics.get("config", {}).get("gauss_sigma", 4)

    cov_inv_d = cov_inv.to(device)
    cov_inv_p = cov_inv_d.permute(2, 0, 1).contiguous()  # [3136, 550, 550]

    # Hitung global min/max dari semua 83 test images
    # (biar normalisasi konsisten antara dashboard sama experiment)
    padim_min, padim_max = None, None
    padim_map_min, padim_map_max = None, None
    ood_mean_max = None
    test_feat_path = os.path.join(exp_dir, "test_features_padim.pt")
    if os.path.exists(test_feat_path):
        test_feat = torch.load(test_feat_path, map_location="cpu")
        mean_d = mean.to(device)
        test_feat = test_feat.to(device)

        for i in range(0, len(test_feat), 16):
            batch = test_feat[i:i+16]
            delta = batch - mean_d.T.unsqueeze(0)
            scores = _batch_mahalanobis(delta, cov_inv_p)
            scores = torch.sqrt(scores.clamp(min=0))

            # Track max mean patch score per image (untuk OOD threshold)
            mean_scores = scores.mean(dim=1)
            batch_ood_max = mean_scores.max().item()
            if ood_mean_max is None or batch_ood_max > ood_mean_max:
                ood_mean_max = batch_ood_max

            maps = scores.view(-1, 1, 56, 56)
            maps = F.interpolate(maps, size=(224, 224), mode="bilinear", align_corners=False)
            kernel = _gaussian_kernel(sigma, device=device)
            maps = F.conv2d(maps, kernel, padding=kernel.shape[-1] // 2)
            if padim_map_min is None:
                padim_map_min = maps.min().item()
                padim_map_max = maps.max().item()
            else:
                padim_map_min = min(padim_map_min, maps.min().item())
                padim_map_max = max(padim_map_max, maps.max().item())

        del test_feat, mean_d
        if device == "cuda":
            torch.cuda.empty_cache()

    from src.config import COMBOOOD_THRESHOLD

    ood_feature_bank = None
    ood_feature_bank_norm_sq = None
    train_feat_path = os.path.join(exp_dir, "train_features_padim.pt")

    # Auto-generate train_features_padim.pt dari train_raw.pt + dim_indices.pt
    if not os.path.exists(train_feat_path):
        raw_path = os.path.join(exp_dir, "train_raw.pt")
        dim_path = os.path.join(exp_dir, "dim_indices.pt")
        if os.path.exists(raw_path) and os.path.exists(dim_path):
            train_raw = torch.load(raw_path, map_location="cpu")
            dim_idx = torch.load(dim_path, map_location="cpu")
            train_reduced = train_raw[:, dim_idx, :, :].contiguous()
            B, C, H, W = train_reduced.shape
            train_patches_gen = train_reduced.permute(0, 2, 3, 1).reshape(B, H * W, C)
            torch.save(train_patches_gen, train_feat_path)
            del train_raw, train_reduced, train_patches_gen

    if os.path.exists(train_feat_path):
        train_patches = torch.load(train_feat_path, map_location="cpu")  # [209, 3136, 550]
        # Mean pool: [209, 3136, 550] -> [209, 550] — untuk KNN component ComboOOD
        bank_mean = train_patches.mean(dim=1)
        bank_norm_sq = (bank_mean ** 2).sum(dim=1)
        ood_feature_bank = bank_mean          # [209, 550] (CPU)
        ood_feature_bank_norm_sq = bank_norm_sq  # [209] (CPU)
        del train_patches

    return {
        "mean": mean.to(device),                         # [550, 3136]
        "cov_inv": cov_inv.to(device),                    # [550, 550, 3136]
        "cov_inv_p": cov_inv_p,                           # [3136, 550, 550]
        "dim_indices": dim_indices.to(device),            # [550]
        "feature_bank": feature_bank,                     # [655424, 550] (CPU)
        "feature_bank_norm_sq": feature_bank_norm_sq,     # [655424] (CPU)
        "threshold_padim": metrics.get("padim", {}).get("threshold"),
        "threshold_knn": metrics.get("knn", {}).get("threshold"),
        "padim_map_min": padim_map_min,                   # global pixel min
        "padim_map_max": padim_map_max,                   # global pixel max
        "ood_threshold": COMBOOOD_THRESHOLD,              # threshold tetap: 0.0
        "ood_feature_bank": ood_feature_bank,             # [209, 550] (CPU)
        "ood_feature_bank_norm_sq": ood_feature_bank_norm_sq,  # [209] (CPU)
        "gauss_sigma": sigma,                             # sigma utk Gaussian kernel
        "metrics": metrics,                               # dict hasil experiment
    }


def preprocess_and_extract(pil_image, extractor, exp_data, device):
    """
    Proses 1 gambar dari PIL -> tensor -> feature extraction -> dim reduction -> patch embeddings.
    
    Output: [1, 3136, 550] = 1 gambar, 3136 patch, 550 channel per patch
    """
    _setup_paths()
    from prepocessing.resize import get_resize_transform
    from prepocessing.normalization import get_normalize_transform

    img_tensor = transforms.Compose([
        get_resize_transform(224),
        transforms.ToTensor(),
        get_normalize_transform()
    ])(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = extractor.extract(img_tensor)  # [1, 1792, 56, 56]

    dim_idx = exp_data["dim_indices"]  # index 550 channel terpilih
    reduced = embedding[:, dim_idx, :, :].contiguous()  # [1, 550, 56, 56]
    b, c, h, w = reduced.shape
    patches = reduced.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [1, 3136, 550]
    return patches


def infer_padim(patches, exp_data, device):
    """
    PaDiM inference untuk 1 gambar.
    
    Alur:
    - patches [1,3136,550] - mean [550,3136] = delta [1,3136,550]
    - Mahalanobis: delta @ cov_inv_p @ delta.T -> sqrt -> [1,3136] skor
    - Reshape [1,1,56,56] -> upsample 224x224 -> Gaussian blur sigma=4
    - Score = max pixel map
    - Prediksi: score_norm >= threshold_padim? ANOMALI : NORMAL
    """
    import torch.nn.functional as F

    t0 = _time.perf_counter()
    mean = exp_data["mean"]
    cov_inv_p = exp_data["cov_inv_p"]

    with torch.no_grad():
        delta = patches - mean.T.unsqueeze(0)  # [1, 3136, 550]
        patch_scores = _batch_mahalanobis(delta, cov_inv_p)
        patch_scores = torch.sqrt(patch_scores.clamp(min=0))

        padim_map = patch_scores.view(1, 1, 56, 56)
        padim_map = F.interpolate(padim_map, size=(224, 224), mode="bilinear", align_corners=False)
        sigma = exp_data.get("gauss_sigma", 4)
        kernel = _gaussian_kernel(sigma, device=device)
        padim_map = F.conv2d(padim_map, kernel, padding=kernel.shape[-1] // 2)
        padim_score = padim_map.max().item()

    padim_time = _time.perf_counter() - t0

    # Normalize score pake global min/max dari 83 test images
    padim_norm = None
    padim_pred = None
    if exp_data["padim_map_min"] is not None and exp_data["padim_map_max"] is not None:
        pmin, pmax = exp_data["padim_map_min"], exp_data["padim_map_max"]
        if pmax > pmin:
            padim_norm = (padim_score - pmin) / (pmax - pmin)
        if padim_norm is not None and exp_data["threshold_padim"] is not None:
            padim_pred = bool(padim_norm >= exp_data["threshold_padim"])

    return {
        "score": padim_score,
        "score_norm": padim_norm,
        "map_raw": padim_map.squeeze().cpu().numpy(),
        "time": padim_time,
        "is_anomaly": padim_pred,
    }


def infer_knn(patches, exp_data, device):
    """
    KNN inference untuk 1 gambar.
    
    Alur:
    - patches [3136,550] -> compute_knn_anomaly_score(feature_bank [655424,550], K=5)
      -> Euclidean ke semua 655.424 bank -> ambil 5 terkecil -> rata-rata -> [3136] skor
    - Reshape [1,1,56,56] -> upsample 224x224
    - Score = max pixel map
    - Prediksi: score >= threshold_knn? ANOMALI : NORMAL
    """
    import torch.nn.functional as F
    _setup_paths()
    from knn_baseline import compute_knn_anomaly_score

    t0 = _time.perf_counter()
    k = exp_data["metrics"].get("config", {}).get("knn_k", 5)

    with torch.no_grad():
        patch_scores_knn = compute_knn_anomaly_score(
            patches.squeeze(0), exp_data["feature_bank"], k=k,
            chunk_size=30000, bank_norm_sq=exp_data["feature_bank_norm_sq"],
        )

        knn_map = patch_scores_knn.view(1, 1, 56, 56)
        knn_map = F.interpolate(knn_map, size=(224, 224), mode="bilinear", align_corners=False)
        knn_score = knn_map.max().item()

    knn_time = _time.perf_counter() - t0

    knn_pred = None
    if exp_data["threshold_knn"] is not None:
        knn_pred = bool(knn_score >= exp_data["threshold_knn"])

    return {
        "score": knn_score,
        "map_raw": knn_map.squeeze().cpu().numpy(),
        "time": knn_time,
        "is_anomaly": knn_pred,
    }
