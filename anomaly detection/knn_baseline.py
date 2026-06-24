import torch


def build_feature_bank(train_features):
    """
    Build feature bank from training patch embeddings.

    train_features: [N, P, C]  (N images, P patches, C channels)
    returns: feature_bank [N * P, C]
    """
    N, P, C = train_features.shape
    feature_bank = train_features.reshape(N * P, C)
    return feature_bank


def compute_knn_anomaly_score(test_embedding, feature_bank, k=1, chunk_size=20000, bank_norm_sq=None):
    """
    Compute anomaly score per patch using K-NN Euclidean distance (chunked + vectorized).

    Proses semua patch sekaligus per chunk untuk menghindari OOM.
    Formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b

    test_embedding: [P, C]  patch embeddings for one test image
    feature_bank: [M, C]  all training patch embeddings
    k: number of nearest neighbors (default 5)
    chunk_size: max number of bank entries processed at once
    bank_norm_sq: [M] pre-computed squared norms of feature_bank (optional)

    returns: patch_scores [P]  average distance to K nearest neighbors
    """
    P, C = test_embedding.shape
    M = feature_bank.shape[0]

    device = test_embedding.device
    k = min(k, M)

    topk_dist = torch.full((P, k), float("inf"), device=device)

    test_norm_sq = (test_embedding ** 2).sum(dim=1, keepdim=True)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = feature_bank[start:end].to(device)

        dots = test_embedding @ chunk.T
        if bank_norm_sq is not None:
            bank_norm = bank_norm_sq[start:end].to(device).unsqueeze(0)
        else:
            bank_norm = (chunk ** 2).sum(dim=1, keepdim=True).T

        dist_sq = test_norm_sq + bank_norm - 2 * dots
        dist_sq = torch.clamp(dist_sq, min=0)
        dist = dist_sq.sqrt()

        combined = torch.cat([topk_dist, dist], dim=1)
        topk_dist, _ = combined.topk(k, dim=1, largest=False)

    patch_scores = topk_dist.mean(dim=1)
    return patch_scores
