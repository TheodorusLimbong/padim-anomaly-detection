import os
from PIL import Image
from torchvision import transforms

from augmentation import get_augmentation_transform


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def generate_augmented_dataset(
    input_dir: str,
    output_dir: str,
    category: str = "bottle"
):
    src_path = os.path.join(input_dir, category, "train", "good")

    aug_types = {
        "original": None,
        "flip": transforms.RandomHorizontalFlip(p=1.0),
        "rotation": transforms.RandomRotation(5),
        "color": transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        )
    }

    for aug_name, aug_transform in aug_types.items():
        print(f"Processing: {aug_name}")

        dst_folder = os.path.join(
            output_dir,
            category,
            "train",
            f"good_{aug_name}"
        )

        for img_name in os.listdir(src_path):
            img_path = os.path.join(src_path, img_name)

            if not img_name.endswith(".png"):
                continue

            img = Image.open(img_path).convert("RGB")

            if aug_transform:
                img = aug_transform(img)

            save_path = os.path.join(dst_folder, img_name)
            save_image(img, save_path)

    print("Augmented dataset created successfully!")


if __name__ == "__main__":
    generate_augmented_dataset(
        input_dir="dataset/mvtec_anomaly_detection",
        output_dir="dataset_augmented/mvtec_anomaly_detection"
    )