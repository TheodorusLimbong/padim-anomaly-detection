from preprocessing.dataloader import get_dataloader


def run_preprocessing():
    root_dir = "dataset/mvtec_anomaly_detection"

    print("===== PREPROCESSING START =====")

    train_loader = get_dataloader(
        root_dir=root_dir,
        category="bottle",
        phase="train",
        batch_size=8
    )

    test_loader = get_dataloader(
        root_dir=root_dir,
        category="bottle",
        phase="test",
        batch_size=8
    )

    # ===== CEK SAMPLE =====
    train_sample = next(iter(train_loader))
    test_sample = next(iter(test_loader))

    print("\n--- TRAIN SAMPLE ---")
    print("Shape:", train_sample["image"].shape)
    print("Label:", train_sample["label"])

    print("\n--- TEST SAMPLE ---")
    print("Shape:", test_sample["image"].shape)
    print("Label:", test_sample["label"])

    print("\n===== PREPROCESSING DONE =====")

    return train_loader, test_loader


if __name__ == "__main__":
    run_preprocessing()