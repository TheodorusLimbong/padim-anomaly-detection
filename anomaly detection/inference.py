import torch
import torch.nn.functional as F
from mahalanobis import compute_anomaly_map, upsample_anomaly_map
from knn_baseline import compute_knn_anomaly_score


def compute_padim_scores(test_features, mean, cov_inv, img_size=224):
    """
    Compute anomaly scores and maps for test set using PaDiM.

    test_features: [N, P, C]
    mean: [C, P]
    cov_inv: [C, C, P]
    img_size: target size for upsampled anomaly maps

    returns:
        image_scores: [N]
        anomaly_maps: list of [H_img, W_img]
    """
    N, P, C = test_features.shape
    H = W = int(P ** 0.5)
    image_scores = []
    anomaly_maps = []

    for i in range(N):
        embedding = test_features[i]
        a_map, img_score = compute_anomaly_map(embedding, mean, cov_inv)
        a_map_up = upsample_anomaly_map(a_map, (img_size, img_size))
        image_scores.append(img_score)
        anomaly_maps.append(a_map_up)

    return torch.tensor(image_scores), anomaly_maps


def compute_knn_scores(test_features, feature_bank, k=1, img_size=224):
    """
    Compute anomaly scores using KNN + Euclidean distance baseline.

    test_features: [N, P, C]
    feature_bank: [M, C]  (all training patch embeddings)
    k: number of nearest neighbors
    img_size: target size for upsampled anomaly maps

    returns:
        image_scores: [N]
        anomaly_maps: list of [H_img, W_img]
    """
    N, P, C = test_features.shape
    H = W = int(P ** 0.5)
    image_scores = []
    anomaly_maps = []

    for i in range(N):
        embedding = test_features[i]
        patch_scores = compute_knn_anomaly_score(embedding, feature_bank, k)
        a_map = patch_scores.view(H, W)
        a_map_up = F.interpolate(
            a_map.unsqueeze(0).unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze()
        img_score = a_map.max().item()
        image_scores.append(img_score)
        anomaly_maps.append(a_map_up)

    return torch.tensor(image_scores), anomaly_maps
