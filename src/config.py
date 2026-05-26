import torch
import os

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================
DATASET_PATH = "./dataset/mvtec_anomaly_detection/bottle"

# =========================
# IMAGE
# =========================
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4

# =========================
# FEATURE EXTRACTION
# =========================
SELECTED_LAYERS = ["layer1", "layer2", "layer3"]

# =========================
# PADIM
# =========================
PADIM_N_DIMS = 100

# =========================
# OUTPUT
# =========================
FEATURE_OUTPUT_PATH = "output/features"

# =========================
# SEED
# =========================
SEED = 42

# =========================
# CREATE OUTPUT DIR
# =========================
os.makedirs(FEATURE_OUTPUT_PATH, exist_ok=True)