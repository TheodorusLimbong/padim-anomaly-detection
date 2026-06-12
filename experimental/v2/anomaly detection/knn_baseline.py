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


def compute_knn_anomaly_score(test_embedding, feature_bank, k=1, chunk_size=20000):
    """
    Compute anomaly score per patch using 1-NN Euclidean distance (chunked + vectorized).

    Proses semua patch sekaligus per chunk untuk menghindari OOM.
    Formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b

    test_embedding: [P, C]  patch embeddings for one test image
    feature_bank: [M, C]  all training patch embeddings
    k: number of nearest neighbors (proposal uses k=1)
    chunk_size: max number of bank entries processed at once

    returns: patch_scores [P]  min Euclidean distance per patch
    """
    P, C = test_embedding.shape
    M = feature_bank.shape[0]

    device = test_embedding.device

    patch_scores = torch.full((P,), float("inf"), device=device)

    test_norm_sq = (test_embedding ** 2).sum(dim=1, keepdim=True)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = feature_bank[start:end].to(device)

        dots = test_embedding @ chunk.T
        bank_norm_sq = (chunk ** 2).sum(dim=1, keepdim=True).T

        dist_sq = test_norm_sq + bank_norm_sq - 2 * dots
        dist_sq = torch.clamp(dist_sq, min=0)

        chunk_min, _ = dist_sq.sqrt().min(dim=1)

        mask = chunk_min < patch_scores
        patch_scores[mask] = chunk_min[mask]

    return patch_scores
