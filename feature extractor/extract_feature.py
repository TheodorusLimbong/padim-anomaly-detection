# Cara jalankan: (di-import oleh run_padim.py, jarang dipakai langsung)
# Ekstraksi fitur multi-layer (layer1+layer2+layer3) menggunakan ResNet-50 + MoCo v2

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from config import *

from models.backbone import ResNet50Backbone
from models.hook_feature import FeatureExtractor

from utils.feature_utils import reshape_embedding
from utils.save_utils import save_features


# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
# =========================
# DATASET
# =========================
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "train"),
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# MODEL
# =========================
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

model = ResNet50Backbone(pretrained=True)
model = model.to(device)
model.eval()

extractor = FeatureExtractor(
    model=model,
    selected_layers=SELECTED_LAYERS
)

# =========================
# FEATURE EXTRACTION
# =========================
all_features = []

print("===== FEATURE EXTRACTION START =====")

for batch_idx, (images, labels) in enumerate(train_loader):
    images = images.to(device)

    embedding = extractor.extract(images)

    patch_embedding = reshape_embedding(embedding)

    all_features.append(patch_embedding.cpu())

    print(f"Batch {batch_idx + 1} processed")


# =========================
# CONCATENATE
# =========================
all_features = torch.cat(all_features, dim=0)

print(f"Final Feature Shape: {all_features.shape}")
# =========================
# SAVE
# =========================
save_features(
    features=all_features,
    save_path=FEATURE_OUTPUT_PATH,
    file_name="train_features.pt"
)

print("===== FEATURE EXTRACTION FINISHED =====")