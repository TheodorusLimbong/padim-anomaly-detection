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