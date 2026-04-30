import torch

def compute_statistics(features):
    """
    features: [N, C, H, W]
    """
    N, C, H, W = features.shape
    features = features.view(N, C, -1)

    mean = torch.mean(features, dim=0)
    cov = torch.zeros(C, C, H * W)

    for i in range(H * W):
        cov[:, :, i] = torch.cov(features[:, :, i].T)

    return mean, cov