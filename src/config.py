# Cara jalankan: (ini file config — di-import oleh semua file lain)
# SEMUA parameter eksperimen ada di sini. Ubah angka-angka di bawah untuk eksperimen.
#
# Parameter penting:
#   PADIM_N_DIMS=550  -> channel reduction (1792->550)
#   KNN_K=5           -> jumlah tetangga K-NN
#   GAUSS_SIGMA=4     -> kekuatan Gaussian blur pada anomaly map
#   SEED=1024         -> biar random selection channel konsisten tiap run

import torch
import os

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================
DATASET_PATH = "./dataset/mvtec_anomaly_detection/bottle"  # hardcoded ke bottle

# =========================
# IMAGE / DATALOADER
# =========================
IMAGE_SIZE = 224      # ResNet-50 input, feature map jadi 56x56 (224/4=56)
BATCH_SIZE = 16       # batch size untuk dataloader
NUM_WORKERS = 4       # worker threads

# =========================
# FEATURE EXTRACTION
# =========================
SELECTED_LAYERS = ["layer1", "layer2", "layer3"]  # multi-layer: 256+512+1024=1792ch

# =========================
# PADIM
# =========================
PADIM_N_DIMS = 550    # channel reduction dari 1792 ke 550 (PaDiM paper optimal ~550)
GAUSS_SIGMA = 4       # Gaussian smoothing sigma (PaDiM paper: sigma=4 untuk semua metode)

# =========================
# KNN
# =========================
KNN_K = 5             # K-NN parameter

# =========================
# OOD (OUT-OF-DISTRIBUTION) — ComboOOD
# =========================
COMBOOOD_THRESHOLD = 0.0   # threshold ComboOOD: score >= 0 = bottle, < 0 = non-bottle
                           # gap: non-bottle max=-49, bottle min=+0.2 — terpisah sempurna

# =========================
# OUTPUT
# =========================
FEATURE_OUTPUT_PATH = "output/features"  # (tidak dipakai, pake output/experiments/)

# =========================
# SEED
# =========================
SEED = 1024           # seed untuk reproducibility

# =========================
# CREATE OUTPUT DIR
# =========================
os.makedirs(FEATURE_OUTPUT_PATH, exist_ok=True)