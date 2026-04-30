import os


def load_mvtec_paths(root_dir, category="bottle", phase="train"):
    img_paths = []
    labels = []
    mask_paths = []

    dataset_path = os.path.join(root_dir, category)

    if phase == "train":
        img_dir = os.path.join(dataset_path, "train", "good")

        for img_name in os.listdir(img_dir):
            img_paths.append(os.path.join(img_dir, img_name))
            labels.append(0)
            mask_paths.append(None)

    else:  # test
        test_dir = os.path.join(dataset_path, "test")

        for defect_type in os.listdir(test_dir):
            defect_path = os.path.join(test_dir, defect_type)

            for img_name in os.listdir(defect_path):
                img_paths.append(os.path.join(defect_path, img_name))

                if defect_type == "good":
                    labels.append(0)
                    mask_paths.append(None)
                else:
                    labels.append(1)

                    mask_path = os.path.join(
                        dataset_path,
                        "ground_truth",
                        defect_type,
                        img_name.replace(".png", "_mask.png")
                    )
                    mask_paths.append(mask_path)

    return img_paths, labels, mask_paths