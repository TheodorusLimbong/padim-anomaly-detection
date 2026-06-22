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
# Live Inference
# ============================================================
import time as _time

EXPERIMENTS_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "experiments"
)


def _gaussian_kernel(sigma, kernel_size=None, device="cpu", dtype=torch.float32):
    if kernel_size is None:
        kernel_size = int(2 * round(3 * sigma) + 1)
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=dtype)
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
    return kernel_2d.expand(1, 1, kernel_size, kernel_size)


def _setup_paths():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ["prepocessing", "feature extractor", "anomaly detection"]:
        p = os.path.join(repo_root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def list_experiments():
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
    exp_dir = os.path.join(EXPERIMENTS_BASE, exp_name)
    _setup_paths()
    from padim import load_statistics
    import torch.nn.functional as F

    mean, cov_inv, _ = load_statistics(exp_dir)
    dim_indices = torch.load(os.path.join(exp_dir, "dim_indices.pt"), map_location="cpu")
    feature_bank = torch.load(os.path.join(exp_dir, "feature_bank.pt"), map_location="cpu")

    with open(os.path.join(exp_dir, "metrics.json")) as f:
        metrics = json.load(f)

    padim_min, padim_max = None, None
    padim_map_min, padim_map_max = None, None
    test_feat_path = os.path.join(exp_dir, "test_features_padim.pt")
    if os.path.exists(test_feat_path):
        test_feat = torch.load(test_feat_path, map_location="cpu")
        mean_d = mean.to(device)
        cov_inv_d = cov_inv.to(device)
        test_feat = test_feat.to(device)
        cov_inv_p = cov_inv_d.permute(2, 0, 1).contiguous()
        raw_img_scores_list = []

        for i in range(0, len(test_feat), 16):
            batch = test_feat[i:i+16]
            delta = batch - mean_d.T.unsqueeze(0)
            scores = torch.einsum("bpc,pcd,bpd->bp", delta, cov_inv_p, delta)
            scores = torch.sqrt(scores.clamp(min=0))
            maps = scores.view(-1, 1, 56, 56)
            maps = F.interpolate(maps, size=(224, 224), mode="bilinear", align_corners=False)
            kernel = _gaussian_kernel(4, device=device)
            maps = F.conv2d(maps, kernel, padding=kernel.shape[-1] // 2)
            raw_img_scores_list.append(maps.view(len(batch), -1).max(dim=1)[0])
            if padim_map_min is None:
                padim_map_min = maps.min().item()
                padim_map_max = maps.max().item()
            else:
                padim_map_min = min(padim_map_min, maps.min().item())
                padim_map_max = max(padim_map_max, maps.max().item())

        raw_img_scores = torch.cat(raw_img_scores_list)
        padim_min = raw_img_scores.min().item()
        padim_max = raw_img_scores.max().item()
        del test_feat, mean_d, cov_inv_d, cov_inv_p, raw_img_scores
        if device == "cuda":
            torch.cuda.empty_cache()

    return {
        "mean": mean.to(device),
        "cov_inv": cov_inv.to(device),
        "dim_indices": dim_indices.to(device),
        "feature_bank": feature_bank,
        "threshold_padim": metrics.get("padim", {}).get("threshold"),
        "threshold_knn": metrics.get("knn", {}).get("threshold"),
        "padim_min": padim_min,
        "padim_max": padim_max,
        "padim_map_min": padim_map_min,
        "padim_map_max": padim_map_max,
        "metrics": metrics,
    }


def infer_single_image(pil_image, extractor, exp_data, device):
    import torch.nn.functional as F

    _setup_paths()
    from knn_baseline import compute_knn_anomaly_score
    from prepocessing.resize import get_resize_transform
    from prepocessing.normalization import get_normalize_transform

    t0 = _time.perf_counter()

    img_tensor = transforms.Compose([
        get_resize_transform(224),
        transforms.ToTensor(),
        get_normalize_transform()
    ])(pil_image).unsqueeze(0).to(device)
    t1 = _time.perf_counter()

    embedding = extractor.extract(img_tensor)
    t2 = _time.perf_counter()

    dim_idx = exp_data["dim_indices"]
    reduced = embedding[:, dim_idx, :, :].contiguous()
    b, c, h, w = reduced.shape
    patches = reduced.permute(0, 2, 3, 1).reshape(b, h * w, c)
    t3 = _time.perf_counter()

    # PaDiM
    t_p0 = _time.perf_counter()
    mean, cov_inv = exp_data["mean"], exp_data["cov_inv"]
    cov_inv_p = cov_inv.permute(2, 0, 1).contiguous()
    delta = patches - mean.T.unsqueeze(0)
    patch_scores = torch.einsum("bpc,pcd,bpd->bp", delta, cov_inv_p, delta)
    patch_scores = torch.sqrt(patch_scores.clamp(min=0))

    padim_map = patch_scores.view(1, 1, 56, 56)
    padim_map = F.interpolate(padim_map, size=(224, 224), mode="bilinear", align_corners=False)
    kernel = _gaussian_kernel(4, device=device)
    padim_map = F.conv2d(padim_map, kernel, padding=kernel.shape[-1] // 2)
    padim_score = padim_map.max().item()
    padim_time = _time.perf_counter() - t_p0

    # KNN
    t_k0 = _time.perf_counter()
    k = exp_data["metrics"].get("config", {}).get("knn_k", 5)
    patch_scores_knn = compute_knn_anomaly_score(patches.squeeze(0), exp_data["feature_bank"], k=k)

    knn_map = patch_scores_knn.view(1, 1, 56, 56)
    knn_map = F.interpolate(knn_map, size=(224, 224), mode="bilinear", align_corners=False)
    knn_score = knn_map.max().item()
    knn_time = _time.perf_counter() - t_k0

    padim_norm = None
    padim_pred = None
    if exp_data["padim_min"] is not None and exp_data["padim_max"] is not None:
        pmin, pmax = exp_data["padim_min"], exp_data["padim_max"]
        if pmax > pmin:
            padim_norm = (padim_score - pmin) / (pmax - pmin)
        if padim_norm is not None and exp_data["threshold_padim"] is not None:
            padim_pred = bool(padim_norm >= exp_data["threshold_padim"])

    knn_pred = None
    if exp_data["threshold_knn"] is not None:
        knn_pred = bool(knn_score >= exp_data["threshold_knn"])

    return {
        "padim": {"score": padim_score, "score_norm": padim_norm, "map_raw": padim_map.squeeze().cpu().numpy(), "time": padim_time, "is_anomaly": padim_pred},
        "knn": {"score": knn_score, "map_raw": knn_map.squeeze().cpu().numpy(), "time": knn_time, "is_anomaly": knn_pred},
        "preprocess_time": t1 - t0,
        "feature_time": t2 - t1,
        "dim_time": t3 - t2,
    }
