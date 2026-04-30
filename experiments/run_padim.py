from preprocessing.dataset_loader import load_dataset
from feature_extractor.resnet import get_resnet
from feature_extractor.extract_features import extract_features
from anomaly_detection.padim import compute_statistics
from src.config import DEVICE, DATASET_PATH

def run():
    loader = load_dataset(DATASET_PATH, train=True)
    
    model = get_resnet().to(DEVICE)
    
    features = extract_features(model, loader, DEVICE)
    
    mean, cov = compute_statistics(features)
    
    print("Training PaDiM selesai")

if __name__ == "__main__":
    run()

from preprocessing.dataloader import get_dataloader

train_loader = get_dataloader(
    root_dir="dataset/mvtec_anomaly_detection",
    phase="train"
)

test_loader = get_dataloader(
    root_dir="dataset/mvtec_anomaly_detection",
    phase="test"
)