# Cara jalankan: (di-import oleh inference.py)
# compute_patch_scores: Mahalanobis via torch.bmm (batched matrix multiply)
# Rumus: sqrt((x-mean) @ cov_inv @ (x-mean).T)

import torch
import torch.nn.functional as F


def compute_patch_scores(embedding, mean, cov_inv):
    """
    Hitung Mahalanobis distance untuk tiap patch dalam 1 gambar.
    Rumus: sqrt((x - mean) @ cov_inv @ (x - mean).T)

    embedding: [3136, 550]  = patch test
    mean: [550, 3136]       = mean training per posisi
    cov_inv: [550, 550, 3136] = cov_inv training per posisi

    Cara:
    1. delta = patch_test - mean_posisi_sama  [3136, 550]
    2. cov_inv_p = permute ke [3136, 550, 550] (biar gampang di-bmm)
    3. delta[3136,1,550] @ cov_inv_p[3136,550,550] @ delta[3136,550,1]
       -> [3136] = 1 skor per posisi
    4. sqrt -> Mahalanobis distance

    returns: scores [3136]  Mahalanobis distance per patch
    """
    delta = embedding - mean.T  # [3136, 550]
    cov_inv_p = cov_inv.permute(2, 0, 1)  # [3136, 550, 550]
    # bmm: batched matrix multiply (langsung semua patch, ga pake loop)
    scores = torch.bmm(
        torch.bmm(delta.unsqueeze(1), cov_inv_p),  # [3136,1,550] @ [3136,550,550] -> [3136,1,550]
        delta.unsqueeze(2)                          # @ [3136,550,1] -> [3136,1,1]
    ).squeeze()                                     # -> [3136]
    return torch.sqrt(scores.clamp(min=0))


def compute_anomaly_map(test_embedding, mean, cov_inv):
    """
    Generate anomaly map from patch embeddings.

    test_embedding: [P, C]  flattened patch embeddings
    mean: [C, P]
    cov_inv: [C, C, P]
    returns: anomaly_map [H, W], image_score (float)
    """
    patch_scores = compute_patch_scores(test_embedding, mean, cov_inv)

    P = patch_scores.shape[0]
    H = W = int(P ** 0.5)
    anomaly_map = patch_scores.view(H, W)

    image_score = anomaly_map.max().item()

    return anomaly_map, image_score


def upsample_anomaly_map(anomaly_map, target_size):
    """
    Upsample anomaly map to target image size.

    anomaly_map: [H, W]
    target_size: (H_img, W_img)
    returns: anomaly_map_up [H_img, W_img]
    """
    H, W = anomaly_map.shape
    if (H, W) == target_size:
        return anomaly_map

    map_3d = anomaly_map.unsqueeze(0).unsqueeze(0)
    map_3d = F.interpolate(
        map_3d,
        size=target_size,
        mode="bilinear",
        align_corners=False
    )
    return map_3d.squeeze()
