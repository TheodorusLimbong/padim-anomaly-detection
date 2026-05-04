from preprocessing.dataloader import get_dataloader


def run_preprocessing():
    root_dir = "dataset/mvtec_anomaly_detection"

    train_loader = get_dataloader(root_dir, "train")
    test_loader = get_dataloader(root_dir, "test")

    print("===== PREPROCESSING CHECK =====")

    train_sample = next(iter(train_loader))
    test_sample = next(iter(test_loader))

    print("\nTRAIN:")
    print(train_sample["image"].shape, train_sample["label"])

    print("\nTEST:")
    print(test_sample["image"].shape, test_sample["label"])


if __name__ == "__main__":
    run_preprocessing()