import os
from typing import List, Tuple

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(VALID_EXTENSIONS)


def load_mvtec_paths(
    root_dir: str,
    category: str = "bottle",
    phase: str = "train"
) -> Tuple[List[str], List[int], List[str]]:

    img_paths = []
    labels = []
    mask_paths = []

    # ================= BASE PATH =================
    base_path = os.path.join(root_dir, category)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"[ERROR] Category not found: {base_path}")

    # ================= TRAIN =================
    if phase == "train":
        train_good_path = os.path.join(base_path, "train", "good")

        if not os.path.exists(train_good_path):
            raise FileNotFoundError(f"[ERROR] Train good folder not found: {train_good_path}")

        for img_name in sorted(os.listdir(train_good_path)):
            if not is_image_file(img_name):
                continue

            img_paths.append(os.path.join(train_good_path, img_name))
            labels.append(0)
            mask_paths.append(None)

    # ================= TEST =================
    elif phase == "test":
        test_path = os.path.join(base_path, "test")
        gt_path = os.path.join(base_path, "ground_truth")

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"[ERROR] Test folder not found: {test_path}")

        defect_types = sorted(os.listdir(test_path))

        for defect in defect_types:
            defect_folder = os.path.join(test_path, defect)

            if not os.path.isdir(defect_folder):
                continue

            for img_name in sorted(os.listdir(defect_folder)):
                if not is_image_file(img_name):
                    continue

                img_full_path = os.path.join(defect_folder, img_name)
                img_paths.append(img_full_path)

                # ===== NORMAL =====
                if defect == "good":
                    labels.append(0)
                    mask_paths.append(None)

                # ===== ANOMALY =====
                else:
                    labels.append(1)

                    mask_file = img_name.replace(".png", "_mask.png")
                    mask_full_path = os.path.join(gt_path, defect, mask_file)

                    if os.path.exists(mask_full_path):
                        mask_paths.append(mask_full_path)
                    else:
                        mask_paths.append(None)

    else:
        raise ValueError("phase must be 'train' or 'test'")

    if len(img_paths) == 0:
        raise RuntimeError("[ERROR] No images found — check dataset path!")

    return img_paths, labels, mask_paths


# ================= TEST =================
if __name__ == "__main__":
    root_dir = "dataset/mvtec_anomaly_detection"

    print("\n===== TRAIN CHECK =====")
    train_imgs, train_labels, _ = load_mvtec_paths(root_dir, "bottle", "train")

    print("Total:", len(train_imgs))
    print("Anomaly:", sum(train_labels))

    print("\n===== TEST CHECK =====")
    test_imgs, test_labels, test_masks = load_mvtec_paths(root_dir, "bottle", "test")

    print("Total:", len(test_imgs))
    print("Normal:", sum(1 for l in test_labels if l == 0))
    print("Anomaly:", sum(1 for l in test_labels if l == 1))

    # sample anomaly
    for i, l in enumerate(test_labels):
        if l == 1:
            print("\nSample anomaly:")
            print("Image:", test_imgs[i])
            print("Mask :", test_masks[i])
            break