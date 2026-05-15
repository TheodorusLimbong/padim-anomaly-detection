import torch
import os


def compute_statistics(features):
    """
    Compute per-patch Gaussian statistics from training features.

    features: [N, C, H, W]  (N images, C channels, H height, W width)
    returns: mean [C, H*W], cov [C, C, H*W]
    """
    N, C, H, W = features.shape
    features = features.view(N, C, -1)

    mean = torch.mean(features, dim=0)
    cov = torch.zeros(C, C, H * W)

    for i in range(H * W):
        cov[:, :, i] = torch.cov(features[:, :, i].T)

    return mean, cov


def compute_cov_inv(cov, reg=1e-6):
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


def save_statistics(mean, cov_inv, save_path, file_name="padim_stats.pt"):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, file_name)
    torch.save({"mean": mean, "cov_inv": cov_inv}, path)
    print(f"[INFO] PaDiM statistics saved to: {path}")


def load_statistics(load_path, file_name="padim_stats.pt"):
    path = os.path.join(load_path, file_name)
    data = torch.load(path, map_location="cpu")
    print(f"[INFO] PaDiM statistics loaded from: {path}")
    return data["mean"], data["cov_inv"]
