# Cara jalankan: (di-import oleh experiments/run_padim.py)
# Training PaDiM: reduce_dim(1792->550) -> compute_statistics(mean+cov per posisi) -> compute_cov_inv

import torch
import os


def reduce_dim(features, n_dims=100, seed=42):
    """
    Randomly select n_dims channels from feature maps (PaDiM dim reduction).
    1792 channel -> pilih acak 550 channel.
    Seed=1024 biar channel yang dipilih selalu sama tiap run.

    features: [N, C, H, W]  misal [209, 1792, 56, 56]
    returns: reduced [N, n_dims, H, W], selected indices [n_dims]
             misal [209, 550, 56, 56], index [550] (channel mana aja yang kepilih)
    """
    N, C, H, W = features.shape
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(C, generator=rng)[:n_dims]  # acak pilih 550 index dari 1792
    idx = idx.sort().values                           # urutkan biar konsisten
    return features[:, idx, :, :].contiguous(), idx


def compute_statistics(features, n_dims=None):
    """
    Hitung Gaussian (mean + cov) per posisi patch dari training features.
    Ini INTI dari training PaDiM.

    Input: [209, 550, 56, 56]
    Step 1: reshape -> [209, 550, 3136] (56x56=3136 patch)
    Step 2: untuk tiap posisi i (0-3135):
        ambil [209, 550] dari semua gambar di posisi i
        hitung mean[550] = rata-rata 209 sampel
        hitung cov[550x550] = kovarian 209 sampel

    Output: mean [550, 3136], cov [550, 550, 3136]
    """
    dim_indices = None
    if n_dims is not None and n_dims < features.shape[1]:
        features, dim_indices = reduce_dim(features, n_dims=n_dims)

    N, C, H, W = features.shape
    features = features.view(N, C, -1)  # [209, 550, 3136]

    mean = torch.mean(features, dim=0)  # [550, 3136] = rata-rata 209 gambar per posisi
    cov = torch.zeros(C, C, H * W)

    for i in range(H * W):              # loop 3.136 posisi
        cov[:, :, i] = torch.cov(features[:, :, i].T)  # cov dari 209 sampel [550,550]

    if dim_indices is not None:
        return mean, cov, dim_indices
    return mean, cov


def compute_cov_inv(cov, reg=0.01):
    """
    Hitung kebalikan (inverse) dari matriks kovarian, dengan regularisasi.
    
    Kenapa perlu inverse? Rumus Mahalanobis: (x-mean)^T * cov_inv * (x-mean)
    Kalau cov langsung dipakai, rumusnya kebalik.
    
    reg=0.01: tambah epsilon ke diagonal biar matriks bisa di-inverse
    (mencegah "singular matrix" karena 209 sampel < 550 dimensi)

    Input:  cov [550, 550, 3136]
    Output: cov_inv [550, 550, 3136]
    """
    C, _, P = cov.shape
    cov_inv = torch.zeros_like(cov)
    eye = torch.eye(C, device=cov.device)

    for i in range(P):  # loop 3.136 posisi
        cov_reg = cov[:, :, i] + reg * eye  # +0.01 di diagonal
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
