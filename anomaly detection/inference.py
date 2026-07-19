# Cara jalankan: (di-import oleh experiments/run_padim.py)
# Test PaDiM: compute_padim_scores(Mahalanobis + upsample + Gaussian blur + normalisasi)
# Test KNN:   compute_knn_scores(Euclidean chunked ke feature bank)

import torch
import torch.nn.functional as F
from mahalanobis import compute_patch_scores
from knn_baseline import compute_knn_anomaly_score


def _gaussian_kernel(sigma, kernel_size=None, device="cpu", dtype=torch.float32):
    """Create 2D Gaussian kernel as [1,1,K,K] tensor on given device."""
    if kernel_size is None:
        kernel_size = int(2 * round(3 * sigma) + 1)
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=dtype)
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
    return kernel_2d.expand(1, 1, kernel_size, kernel_size)


def compute_padim_scores(test_features, mean, cov_inv, img_size=224, sigma=4, batch_size=8):
    """
    Compute anomaly scores and maps for test set using PaDiM (optimized).
    Uses batched Mahalanobis + GPU Gaussian blur instead of scipy.

    Alur:
    test_features [83, 3136, 550] + mean [550, 3136] + cov_inv [550, 550, 3136]
        -> delta = test - mean (per posisi)
        -> Mahalanobis: delta @ cov_inv_p @ delta.T -> sqrt -> [83, 3136] skor patch
        -> reshape ke [83, 1, 56, 56]
        -> upsample 56->224
        -> Gaussian blur sigma=4
        -> global min-max normalization 0-1
        -> max per image = image score [83]

    returns:
        image_scores: [83] (1 angka per gambar test)
        anomaly_maps: list of [224, 224] (83 map, untuk visualisasi heatmap)
    """
    N, P, C = test_features.shape
    H = W = int(P ** 0.5)

    # Auto device: pindah ke GPU kalau ada
    device = test_features.device
    if device.type == "cpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        test_features = test_features.to(device)
        mean = mean.to(device)
        cov_inv = cov_inv.to(device)

    cov_inv_p = cov_inv.permute(2, 0, 1).contiguous()  # [P, C, C] -> [3136, 550, 550]
    gauss_kernel = _gaussian_kernel(sigma, device=device)  # kernel 9x9 utk sigma=4
    padding = gauss_kernel.shape[-1] // 2

    all_maps = []

    for start in range(0, N, batch_size):  # proses per batch biar gak OOM
        batch = test_features[start:start + batch_size]
        B = batch.shape[0]

        # Mahalanobis pakai einsum (cepat, tanpa loop)
        # delta = patch_test - mean_posisi_sama
        delta = batch - mean.T.unsqueeze(0)  # [B, 3136, 550]
        patch_scores = torch.einsum("bpc,pcd,bpd->bp", delta, cov_inv_p, delta)
        patch_scores = torch.sqrt(patch_scores.clamp(min=0))  # [B, 3136]

        # Reshape ke grid 56x56 -> upsample ke 224x224 -> Gaussian blur
        maps = patch_scores.view(B, 1, H, W)  # [B, 1, 56, 56]
        maps = F.interpolate(maps, size=(img_size, img_size), mode="bilinear", align_corners=False)
        maps = F.conv2d(maps, gauss_kernel, padding=padding)  # blur
        all_maps.append(maps)

    all_maps = torch.cat(all_maps, dim=0)  # [83, 1, 224, 224]

    # Global min-max normalization (semua map dinormalisasi ke 0-1)
    min_val = all_maps.min()
    max_val = all_maps.max()
    all_maps = (all_maps - min_val) / (max_val - min_val + 1e-8)

    # Image score = nilai pixel tertinggi di tiap map
    image_scores = all_maps.view(N, -1).max(dim=1)[0]

    return image_scores.cpu(), list(all_maps.squeeze(1).cpu())


def compute_knn_scores(test_features, feature_bank, k=1, img_size=224):
    """
    Compute anomaly scores using KNN + Euclidean distance baseline.

    Alur:
    test_features [83, 3136, 550] + feature_bank [655424, 550]
        -> tiap gambar test (loop 83x):
            -> tiap patch (3136) banding ke 655424 bank -> ambil K=5 terkecil -> rata-rata = skor patch
            -> 3136 skor patch -> reshape 56x56 -> upsample 224x224
            -> skor gambar = max dari map
    """
    N, P, C = test_features.shape
    H = W = int(P ** 0.5)
    image_scores = []
    anomaly_maps = []

    for i in range(N):  # loop tiap gambar test (83 gambar)
        embedding = test_features[i]  # [3136, 550] untuk 1 gambar
        patch_scores = compute_knn_anomaly_score(embedding, feature_bank, k)  # [3136]
        a_map = patch_scores.view(H, W)  # grid 56x56
        a_map_up = F.interpolate(
            a_map.unsqueeze(0).unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze()  # upsample ke 224x224
        img_score = a_map.max().item()  # skor gambar = max pixel
        image_scores.append(img_score)
        anomaly_maps.append(a_map_up)

    return torch.tensor(image_scores), anomaly_maps
