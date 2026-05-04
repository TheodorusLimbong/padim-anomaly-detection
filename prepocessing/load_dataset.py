import os
from typing import List, Tuple, Optional

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(VALID_EXTENSIONS)


def load_mvtec_paths(
    root_dir: str,
    category: str = "bottle",
    phase: str = "train"
) -> Tuple[List[str], List[int], List[Optional[str]]]:

    img_paths, labels, mask_paths = [], [], []

    base_path = os.path.join(root_dir, category)

    if phase == "train":
        folder = os.path.join(base_path, "train", "good")

        for f in sorted(os.listdir(folder)):
            if is_image_file(f):
                img_paths.append(os.path.join(folder, f))
                labels.append(0)
                mask_paths.append(None)

    elif phase == "test":
        test_path = os.path.join(base_path, "test")
        gt_path = os.path.join(base_path, "ground_truth")

        for defect in sorted(os.listdir(test_path)):
            defect_folder = os.path.join(test_path, defect)

            for f in sorted(os.listdir(defect_folder)):
                if not is_image_file(f):
                    continue

                img_paths.append(os.path.join(defect_folder, f))

                if defect == "good":
                    labels.append(0)
                    mask_paths.append(None)
                else:
                    labels.append(1)
                    mask = os.path.join(gt_path, defect, f.replace(".png", "_mask.png"))
                    mask_paths.append(mask if os.path.exists(mask) else None)

    return img_paths, labels, mask_paths