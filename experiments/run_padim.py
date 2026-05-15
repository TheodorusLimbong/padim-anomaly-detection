import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ["feature extractor", "anomaly detection"]:
    sys.path.insert(0, os.path.join(repo_root, d))

from models.backbone import ResNet50Backbone
from models.hook_feature import FeatureExtractor
from utils.feature_utils import reshape_embedding
from padim import compute_statistics
from config import SELECTED_LAYERS, IMAGE_SIZE, DATASET_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "train"),
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

def run():
    device = torch.device(DEVICE)
    model = ResNet50Backbone(pretrained=True).to(device)
    model.eval()

    extractor = FeatureExtractor(model=model, selected_layers=SELECTED_LAYERS)

    all_features = []
    for images, _ in train_loader:
        images = images.to(device)
        embedding = extractor.extract(images)
        patch_embedding = reshape_embedding(embedding)
        all_features.append(patch_embedding.cpu())

    features = torch.cat(all_features, dim=0)
    mean, cov = compute_statistics(features)
    print("Training PaDiM selesai")

if __name__ == "__main__":
    run()