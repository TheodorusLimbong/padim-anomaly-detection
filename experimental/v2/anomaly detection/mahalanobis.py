import torch
import torch.nn.functional as F


def compute_patch_scores(embedding, mean, cov_inv):
    """
    Compute Mahalanobis distance for each patch in a single image embedding.
    Uses batched matrix multiplication instead of Python loop (10-50x faster).

    embedding: [P, C]  (P = H*W patches, C channels)
    mean: [C, P]
    cov_inv: [C, C, P]
    returns: scores [P]  Mahalanobis distance per patch
    """
    delta = embedding - mean.T  # [P, C]
    # [P,1,C] @ [P,C,C] @ [P,C,1] → [P]
    cov_inv_p = cov_inv.permute(2, 0, 1)  # [P, C, C]
    scores = torch.bmm(
        torch.bmm(delta.unsqueeze(1), cov_inv_p),
        delta.unsqueeze(2)
    ).squeeze()
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
