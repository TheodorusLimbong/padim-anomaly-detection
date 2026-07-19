# Cara jalankan: (di-import oleh run_padim.py)
# reshape_embedding: [B, C, H, W] -> [B, H*W, C] = flatten spatial jadi patch embeddings
# Contoh: [209, 550, 56, 56] -> [209, 3136, 550] = 3136 patch x 550 channel

import torch


def reshape_embedding(embedding):
    """
    Ubah feature map [B, C, H, W] jadi patch embeddings [B, H*W, C].
    
    H*W = jumlah patch (spatial positions)
    C = channel per patch (feature dimension)
    
    Contoh:
      Input:  [1, 550, 56, 56] = 1 gambar, 550 channel, 56 baris, 56 kolom
      Output: [1, 3136, 550]   = 1 gambar, 3136 patch, 550 channel per patch
    
    Urutan row-major: patch[0]=(1,1), patch[1]=(1,2), ..., patch[3135]=(56,56)
    """
    b, c, h, w = embedding.shape
    embedding = embedding.permute(0, 2, 3, 1)  # [B, H, W, C]
    embedding = embedding.reshape(b, h * w, c)  # [B, H*W, C]
    return embedding
