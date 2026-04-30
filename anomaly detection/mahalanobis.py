import torch

def mahalanobis_distance(x, mean, cov_inv):
    delta = x - mean
    dist = torch.sqrt(torch.matmul(torch.matmul(delta.T, cov_inv), delta))
    return dist