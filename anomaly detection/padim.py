import torch
import os


def reduce_dim(features, n_dims=100, seed=42):
    """
    Randomly select n_dims channels from feature maps (PaDiM dim reduction).

    features: [N, C, H, W]
    returns: reduced [N, n_dims, H, W], selected indices [n_dims]
    """
    N, C, H, W = features.shape
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(C, generator=rng)[:n_dims]
    idx = idx.sort().values
    return features[:, idx, :, :].contiguous(), idx


def compute_statistics(features, n_dims=None):
    """
    Compute per-patch Gaussian statistics from training features.

    features: [N, C, H, W]
    n_dims: optional — if set, reduce dimensions via random selection first
    returns: mean [C', H*W], cov [C', C', H*W], dim_indices (if n_dims given)
    """
    dim_indices = None
    if n_dims is not None and n_dims < features.shape[1]:
        features, dim_indices = reduce_dim(features, n_dims=n_dims)

    N, C, H, W = features.shape
    features = features.view(N, C, -1)

    mean = torch.mean(features, dim=0)
    cov = torch.zeros(C, C, H * W)

    for i in range(H * W):
        cov[:, :, i] = torch.cov(features[:, :, i].T)

    if dim_indices is not None:
        return mean, cov, dim_indices
    return mean, cov


def compute_cov_inv(cov, reg=0.01):
    """
    Compute regularized inverse covariance matrix.
    
    cov: [C, C, P] where P = H*W
    reg: small regularization constant for numerical stability
    returns: cov_inv [C, C, P]
    """
    C, _, P = cov.shape
    cov_inv = torch.zeros_like(cov)
    eye = torch.eye(C, device=cov.device)

    for i in range(P):
        cov_reg = cov[:, :, i] + reg * eye
        cov_inv[:, :, i] = torch.linalg.inv(cov_reg)

    return cov_inv


def save_statistics(mean, cov_inv, save_path, file_name="padim_stats.pt", dim_indices=None):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, file_name)
    data = {"mean": mean, "cov_inv": cov_inv}
    if dim_indices is not None:
        data["dim_indices"] = dim_indices
    torch.save(data, path)
    print(f"[INFO] PaDiM statistics saved to: {path}")


def load_statistics(load_path, file_name="padim_stats.pt"):
    path = os.path.join(load_path, file_name)
    data = torch.load(path, map_location="cpu")
    print(f"[INFO] PaDiM statistics loaded from: {path}")
    return data["mean"], data["cov_inv"], data.get("dim_indices")
