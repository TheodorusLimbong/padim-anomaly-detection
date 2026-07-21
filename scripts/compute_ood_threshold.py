# Cara jalankan: python scripts/compute_ood_threshold.py
# Validasi ComboOOD threshold: pastikan semua non-bottle < COMBOOOD_THRESHOLD (0.0)
# dan semua test bottle >= COMBOOOD_THRESHOLD
#
# Output: tabel perbandingan combo_score per kategori vs threshold

import os, sys, json, random, math
import torch
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
for d in ["prepocessing", "feature extractor", "anomaly detection"]:
    p = os.path.join(REPO_ROOT, d)
    if p not in sys.path:
        sys.path.insert(0, p)

from torchvision import transforms
from models.backbone import ResNet50Backbone
from models.hook_feature import FeatureExtractor
from src.config import SELECTED_LAYERS, DATASET_PATH, IMAGE_SIZE, COMBOOOD_THRESHOLD
from padim import load_statistics

EXP_NAME = "run_20260612_152608"
EXP_DIR = os.path.join(REPO_ROOT, "output", "experiments", EXP_NAME)
DATASET_BASE = os.path.join(REPO_ROOT, "dataset", "mvtec_anomaly_detection")
N_PER_CATEGORY = 20
NON_BOTTLE_CATEGORIES = [
    "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile",
    "toothbrush", "transistor", "wood", "zipper",
]

def _batch_mahalanobis(delta, cov_inv_p):
    B, P, C = delta.shape
    if B == 1:
        d = delta.squeeze(0).unsqueeze(1)
        step1 = torch.bmm(d, cov_inv_p)
        step2 = torch.bmm(step1, d.permute(0, 2, 1))
        return step2.view(1, P)
    else:
        return torch.einsum("bpc,pcd,bpd->bp", delta, cov_inv_p, delta)

def _nearest_neighbor_dist(x, bank, bank_norm_sq):
    x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
    dots = x @ bank.T
    dist_sq = x_norm_sq + bank_norm_sq.unsqueeze(0) - 2 * dots
    return dist_sq.min().sqrt().item()

def load_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    mean, cov_inv, _ = load_statistics(EXP_DIR)
    dim_indices = torch.load(os.path.join(EXP_DIR, "dim_indices.pt"), map_location="cpu")
    cov_inv_p = cov_inv.to(device).permute(2, 0, 1).contiguous()

    # Load / generate train_feature bank untuk OOD
    train_feat_path = os.path.join(EXP_DIR, "train_features_padim.pt")
    if not os.path.exists(train_feat_path):
        raw_path = os.path.join(EXP_DIR, "train_raw.pt")
        dim_path = os.path.join(EXP_DIR, "dim_indices.pt")
        if os.path.exists(raw_path) and os.path.exists(dim_path):
            train_raw = torch.load(raw_path, map_location="cpu")
            dim_idx = torch.load(dim_path, map_location="cpu")
            train_reduced = train_raw[:, dim_idx, :, :].contiguous()
            B, C, H, W = train_reduced.shape
            train_patches = train_reduced.permute(0, 2, 3, 1).reshape(B, H * W, C)
            torch.save(train_patches, train_feat_path)
            del train_raw, train_reduced

    ood_feature_bank = None
    ood_feature_bank_norm_sq = None
    if os.path.exists(train_feat_path):
        train_patches = torch.load(train_feat_path, map_location="cpu")
        bank_mean = train_patches.mean(dim=1)
        bank_norm_sq = (bank_mean ** 2).sum(dim=1)
        ood_feature_bank = bank_mean
        ood_feature_bank_norm_sq = bank_norm_sq
        del train_patches

    print(f"Mean: {mean.shape}, Cov_inv_p: {cov_inv_p.shape}")
    print(f"Dim indices: {dim_indices.shape}")
    print(f"ComboOOD threshold: {COMBOOOD_THRESHOLD} (score >= threshold = bottle)")

    return {
        "device": device,
        "mean": mean.to(device),
        "cov_inv_p": cov_inv_p,
        "dim_indices": dim_indices.to(device),
        "ood_feature_bank": ood_feature_bank,
        "ood_feature_bank_norm_sq": ood_feature_bank_norm_sq,
    }

def load_model(device):
    model = ResNet50Backbone(pretrained=True).to(device)
    model.eval()
    extractor = FeatureExtractor(model=model, selected_layers=SELECTED_LAYERS)
    return extractor

def compute_combo_ood_score(pil_image, extractor, exp_data):
    device = exp_data["device"]
    mean = exp_data["mean"]
    cov_inv_p = exp_data["cov_inv_p"]
    dim_idx = exp_data["dim_indices"]
    ood_bank = exp_data["ood_feature_bank"]
    ood_bank_norm_sq = exp_data["ood_feature_bank_norm_sq"]

    img_tensor = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = extractor.extract(img_tensor)
        reduced = embedding[:, dim_idx, :, :].contiguous()
        b, c, h, w = reduced.shape
        patches = reduced.permute(0, 2, 3, 1).reshape(b, h * w, c)

        delta = patches - mean.T.unsqueeze(0)
        md_sq = _batch_mahalanobis(delta, cov_inv_p)
        mean_md_sq = md_sq.mean().item()
        mc = -0.5 * mean_md_sq

        mean_emb = patches.mean(dim=1)
        kd = _nearest_neighbor_dist(mean_emb, ood_bank.to(device), ood_bank_norm_sq.to(device))
        kc = -math.sqrt(550) * math.log(kd + 1e-8)

        return kc + mc

def get_good_images(category):
    path = os.path.join(DATASET_BASE, category, "train", "good")
    if not os.path.exists(path):
        return []
    files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    return files

def main():
    print("=" * 70)
    print(f"VALIDASI COMBOOD - Threshold={COMBOOOD_THRESHOLD} (non-bottle PASS jika < {COMBOOOD_THRESHOLD})")
    print("=" * 70)

    exp_data = load_experiment()
    extractor = load_model(exp_data["device"])

    # Validasi 14 non-bottle categories
    print("\n" + "-" * 70)
    print(f"{'Kategori Non-Bottle':<20} {'N':>4} {'Mean Score':>12} {'Min':>10} {'Max':>10} {'Status':>12}")
    print("-" * 70)

    all_passed = True
    for cat in NON_BOTTLE_CATEGORIES:
        files = get_good_images(cat)
        if not files:
            print(f"{cat:<20} {'N/A':>4} {'N/A':>12} {'N/A':>10} {'N/A':>10} {'NO DATA':>12}")
            continue

        selected = random.sample(files, min(N_PER_CATEGORY, len(files)))
        scores = []
        for fpath in selected:
            img = Image.open(fpath).convert("RGB")
            score = compute_combo_ood_score(img, extractor, exp_data)
            scores.append(score)

        mean_s = np.mean(scores)
        min_s = np.min(scores)
        max_s = np.max(scores)
        n = len(scores)
        # PASS = semua score < COMBOOOD_THRESHOLD (berarti ditolak sebagai non-bottle)
        rejected = "PASS" if max_s < COMBOOOD_THRESHOLD else "FAIL"
        if max_s >= COMBOOOD_THRESHOLD:
            all_passed = False

        print(f"{cat:<20} {n:>4} {mean_s:>12.4f} {min_s:>10.4f} {max_s:>10.4f} {rejected:>12}")

    print("-" * 70)
    veredict_non = "[PASS] SEMUA non-bottle ditolak dashboard" if all_passed else "[FAIL] ADA non-bottle yang lolos"
    print(f"Non-bottle: {veredict_non}")

    # Validasi bottle test images
    print("\n" + "-" * 70)
    print("Validasi bottle test images (83 images — harus >= threshold)")
    print("-" * 70)

    test_feat_path = os.path.join(EXP_DIR, "test_features_padim.pt")
    if os.path.exists(test_feat_path):
        test_patches = torch.load(test_feat_path, map_location="cpu")
        mean = exp_data["mean"]
        cov_p = exp_data["cov_inv_p"]
        ood_bank = exp_data["ood_feature_bank"].to(exp_data["device"])
        ood_bank_norm = exp_data["ood_feature_bank_norm_sq"].to(exp_data["device"])

        bottle_scores = []
        for i in range(len(test_patches)):
            batch = test_patches[i:i+1].to(exp_data["device"])
            delta = batch - mean.T.unsqueeze(0)
            md_sq = _batch_mahalanobis(delta, cov_p)
            mean_md_sq = md_sq.mean().item()
            mc = -0.5 * mean_md_sq

            mean_emb = batch.mean(dim=1)
            kd = _nearest_neighbor_dist(mean_emb, ood_bank, ood_bank_norm)
            kc = -math.sqrt(550) * math.log(kd + 1e-8)
            bottle_scores.append(kc + mc)

        bottle_scores = np.array(bottle_scores)
        n_bottle_passed = (bottle_scores >= COMBOOOD_THRESHOLD).sum()
        n_bottle_total = len(bottle_scores)

        print(f"Bottle test: min={bottle_scores.min():.4f}, max={bottle_scores.max():.4f}, mean={bottle_scores.mean():.4f}")
        print(f"Lolos (>= {COMBOOOD_THRESHOLD}): {n_bottle_passed}/{n_bottle_total} ({100*n_bottle_passed/n_bottle_total:.1f}%)")
        bottle_ok = n_bottle_passed == n_bottle_total
    else:
        print("test_features_padim.pt tidak ditemukan — skip validasi bottle")
        bottle_ok = True

    print("\n" + "=" * 70)
    if all_passed and bottle_ok:
        print(f"[PASS] SEMUA validasi OK — OOD threshold={COMBOOOD_THRESHOLD} works perfectly")
    else:
        if not all_passed:
            print(f"[FAIL] Non-bottle validation FAIL")
        if not bottle_ok:
            print(f"[FAIL] Bottle validation — {n_bottle_total - n_bottle_passed} images rejected")
    print("=" * 70)

if __name__ == "__main__":
    main()
