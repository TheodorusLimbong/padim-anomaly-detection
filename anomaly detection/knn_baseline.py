# Cara jalankan: (di-import oleh inference.py / run_padim.py)
# build_feature_bank: reshape [N,3136,550] -> [N*3136, 550] = feature bank
# compute_knn_anomaly_score: Euclidean distance chunked (30K per chunk) -> ambil K=5 terkecil -> rata-rata

import torch


def build_feature_bank(train_features):
    """
    Bangun feature bank = tumpuk SEMUA patch training jadi 1 matriks besar.
    Ini referensi untuk KNN: "gudang" semua patch normal.

    train_features: [209, 3136, 550]  (N images, P patches, C channels)
    returns: feature_bank [655424, 550]  (209*3136 = 655.424 patch)
    """
    N, P, C = train_features.shape
    feature_bank = train_features.reshape(N * P, C)  # tumpuk semua
    return feature_bank


def compute_knn_anomaly_score(test_embedding, feature_bank, k=1, chunk_size=20000, bank_norm_sq=None):
    """
    Hitung anomaly score per patch menggunakan K-NN Euclidean distance.

    Untuk 1 gambar test:
    - test_embedding [3136, 550] = 3136 patch x 550 channel
    - feature_bank [655424, 550] = gudang 655.424 patch normal
    
    Cara:
    1. Bandingkan tiap patch test ke SEMUA 655.424 patch bank
       Rumus: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*(a dot b)
       (pecah biar cepat, pakai matriks, bukan loop per patch)
    2. Ambil K=5 jarak terkecil (5 tetangga terdekat)
    3. Rata-ratakan 5 jarak itu -> skor patch
    4. Ulang untuk 3136 patch -> [3136] skor

    chunk_size=30000: bank 655K dibagi jadi ~22 chunk biar tidak OOM

    returns: patch_scores [3136]  average distance to K nearest neighbors
    """
    P, C = test_embedding.shape        # P=3136, C=550
    M = feature_bank.shape[0]          # M=655424

    device = test_embedding.device
    k = min(k, M)

    # Tempat nyimpen K jarak terkecil, diisi infinity dulu
    topk_dist = torch.full((P, k), float("inf"), device=device)

    test_norm_sq = (test_embedding ** 2).sum(dim=1, keepdim=True)  # [3136, 1]

    # Loop chunk: proses 30.000 bank patch per iterasi
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = feature_bank[start:end].to(device)  # [30000, 550]

        dots = test_embedding @ chunk.T  # [3136, 30000] = dot product
        if bank_norm_sq is not None:
            bank_norm = bank_norm_sq[start:end].to(device).unsqueeze(0)
        else:
            bank_norm = (chunk ** 2).sum(dim=1, keepdim=True).T  # [1, 30000]

        # Euclidean: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        dist_sq = test_norm_sq + bank_norm - 2 * dots  # [3136, 30000]
        dist_sq = torch.clamp(dist_sq, min=0)
        dist = dist_sq.sqrt()

        # Gabung hasil sebelumnya dengan chunk baru, ambil K terkecil
        combined = torch.cat([topk_dist, dist], dim=1)  # [3136, K+30000]
        topk_dist, _ = combined.topk(k, dim=1, largest=False)  # [3136, K]

    # Rata-rata K jarak terkecil -> skor patch
    patch_scores = topk_dist.mean(dim=1)  # [3136]
    return patch_scores
