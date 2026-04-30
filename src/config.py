import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = "./dataset/mvtec_anomaly_detection/bottle"
IMAGE_SIZE = 224

BATCH_SIZE = 32
NUM_WORKERS = 4

FEATURE_DIM = 100
LAYERS = ["layer1", "layer2", "layer3"]

SEED = 42