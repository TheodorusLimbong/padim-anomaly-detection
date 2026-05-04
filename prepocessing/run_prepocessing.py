from dataset_loader import get_dataloader
from collections import Counter

def run_preprocessing():
    root_dir = "dataset/mvtec_anomaly_detection"

    train_loader = get_dataloader(root_dir, "train")
    test_loader = get_dataloader(root_dir, "test")

    print("===== PREPROCESSING CHECK =====")

    # 🔹 ambil 1 batch (tetap boleh)
    train_sample = next(iter(train_loader))
    test_sample = next(iter(test_loader))

    print("\nTRAIN SAMPLE:")
    print(train_sample["image"].shape, train_sample["label"])

    print("\nTEST SAMPLE:")
    print(test_sample["image"].shape, test_sample["label"])

    # 🔥 CEK DISTRIBUSI SELURUH DATA
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch["label"].tolist())

    test_labels = []
    for batch in test_loader:
        test_labels.extend(batch["label"].tolist())

    print("\nTRAIN DISTRIBUTION:")
    print(Counter(train_labels))

    print("\nTEST DISTRIBUTION:")
    print(Counter(test_labels))


if __name__ == "__main__":
    run_preprocessing()