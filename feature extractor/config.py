import os

# =========================
# DATASET
# =========================
DATASET_PATH = "dataset/mvtec_anomaly_detection/bottle"

# =========================
# OUTPUT
# =========================
FEATURE_OUTPUT_PATH = "output/features"

# =========================
# IMAGE
# =========================
IMAGE_SIZE = 256
BATCH_SIZE = 8

# =========================
# FEATURE EXTRACTION
# =========================
SELECTED_LAYERS = ["layer1", "layer2", "layer3"]
DEVICE = "cuda"

# =========================
# SAVE
# =========================
os.makedirs(FEATURE_OUTPUT_PATH, exist_ok=True)