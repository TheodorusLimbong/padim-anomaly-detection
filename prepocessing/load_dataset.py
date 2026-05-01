import os
from typing import List, Tuple, Optional


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(VALID_EXTENSIONS)


def _validate_path(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] {name} not found: {path}")


def _load_images_from_folder(folder_path: str) -> List[str]:
    return [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if is_image_file(f)
    ]


def load_mvtec_paths(
    root_dir: str,
    category: str = "bottle",
    phase: str = "train"
) -> Tuple[List[str], List[int], List[Optional[str]]]:

    img_paths: List[str] = []
    labels: List[int] = []
    mask_paths: List[Optional[str]] = []

    base_path = os.path.join(root_dir, category)
    _validate_path(base_path, "Category folder")

    # ================= TRAIN =================
    if phase == "train":
        train_path = os.path.join(base_path, "train", "good")
        _validate_path(train_path, "Train good folder")

        images = _load_images_from_folder(train_path)

        img_paths.extend(images)
        labels.extend([0] * len(images))
        mask_paths.extend([None] * len(images))

    # ================= TEST =================
    elif phase == "test":
        test_path = os.path.join(base_path, "test")
        gt_path = os.path.join(base_path, "ground_truth")

        _validate_path(test_path, "Test folder")

        for defect_type in sorted(os.listdir(test_path)):
            defect_folder = os.path.join(test_path, defect_type)

            if not os.path.isdir(defect_folder):
                continue

            images = _load_images_from_folder(defect_folder)

            for img_path in images:
                img_paths.append(img_path)

                if defect_type == "good":
                    labels.append(0)
                    mask_paths.append(None)
                else:
                    labels.append(1)

                    img_name = os.path.basename(img_path)
                    mask_name = img_name.replace(".png", "_mask.png")
                    mask_path = os.path.join(gt_path, defect_type, mask_name)

                    mask_paths.append(mask_path if os.path.exists(mask_path) else None)

    else:
        raise ValueError("phase must be 'train' or 'test'")

    if not img_paths:
        raise RuntimeError("[ERROR] No images found — check dataset structure.")

    return img_paths, labels, mask_paths


# ================= TEST =================
if __name__ == "__main__":
    root_dir = "dataset/mvtec_anomaly_detection"

    print("\n===== TRAIN CHECK =====")
    train_imgs, train_labels, _ = load_mvtec_paths(root_dir, "bottle", "train")

    print(f"Total images : {len(train_imgs)}")
    print(f"Normal count : {sum(train_labels)}")

    print("\n===== TEST CHECK =====")
    test_imgs, test_labels, test_masks = load_mvtec_paths(root_dir, "bottle", "test")

    print(f"Total images : {len(test_imgs)}")
    print(f"Normal count : {sum(1 for l in test_labels if l == 0)}")
    print(f"Anomaly count: {sum(1 for l in test_labels if l == 1)}")

    # Sample anomaly
    for i, label in enumerate(test_labels):
        if label == 1:
            print("\nSample anomaly:")
            print("Image:", test_imgs[i])
            print("Mask :", test_masks[i])
            break