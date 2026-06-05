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

    test_features: [N, P, C]
    mean: [C, P]
    cov_inv: [C, C, P]
    img_size: target size for upsampled anomaly maps
    sigma: Gaussian smoothing sigma (default 4, per PaDiM paper)
    batch_size: images processed at once (tune based on GPU memory)

    returns:
        image_scores: [N]
        anomaly_maps: list of [H_img, W_img]
    """
    N, P, C = test_features.shape
    H = W = int(P ** 0.5)

    # Auto device: use GPU if available and inputs are on CPU
    device = test_features.device
    if device.type == "cpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        test_features = test_features.to(device)
        mean = mean.to(device)
        cov_inv = cov_inv.to(device)

    cov_inv_p = cov_inv.permute(2, 0, 1).contiguous()  # [P, C, C]
    gauss_kernel = _gaussian_kernel(sigma, device=device)
    padding = gauss_kernel.shape[-1] // 2

    all_maps = []

    for start in range(0, N, batch_size):
        batch = test_features[start:start + batch_size]
        B = batch.shape[0]

        # Batched Mahalanobis via einsum: result[b,p] = delta[b,p,:] @ cov_inv_p[p,:,:] @ delta[b,p,:].T
        # Memory-efficient: only creates [B,P,C] intermediate, NOT [B,P,C,C]
        delta = batch - mean.T.unsqueeze(0)  # [B, P, C]
        patch_scores = torch.einsum("bpc,pcd,bpd->bp", delta, cov_inv_p, delta)
        patch_scores = torch.sqrt(patch_scores.clamp(min=0))  # [B, P]

        # Reshape → upsample → gaussian blur → collect
        maps = patch_scores.view(B, 1, H, W)  # [B, 1, H, W]
        maps = F.interpolate(maps, size=(img_size, img_size), mode="bilinear", align_corners=False)
        maps = F.conv2d(maps, gauss_kernel, padding=padding)
        all_maps.append(maps)

    all_maps = torch.cat(all_maps, dim=0)  # [N, 1, H_img, W_img]

    # Global min-max normalization (per PaDiM paper)
    min_val = all_maps.min()
    max_val = all_maps.max()
    all_maps = (all_maps - min_val) / (max_val - min_val + 1e-8)

    # Image-level anomaly score = max of each anomaly map
    image_scores = all_maps.view(N, -1).max(dim=1)[0]

    # Return CPU tensors for compatibility
    return image_scores.cpu(), list(all_maps.squeeze(1).cpu())


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
