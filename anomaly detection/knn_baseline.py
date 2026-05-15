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


def compute_knn_anomaly_score(test_embedding, feature_bank, k=1):
    """
    Compute anomaly score per patch using 1-NN Euclidean distance.

    test_embedding: [P, C]  patch embeddings for one test image
    feature_bank: [M, C]  all training patch embeddings
    k: number of nearest neighbors (proposal uses k=1)

    returns: patch_scores [P]  min Euclidean distance per patch
    """
    P, C = test_embedding.shape
    patch_scores = torch.zeros(P)

    for i in range(P):
        patch = test_embedding[i].unsqueeze(0)
        diff = feature_bank - patch
        dists = torch.norm(diff, dim=1)
        min_dist = dists.min().item()
        patch_scores[i] = min_dist

    return patch_scores
