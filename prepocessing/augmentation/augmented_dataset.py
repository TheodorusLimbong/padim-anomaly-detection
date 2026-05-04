import os
import torch
from PIL import Image
from torchvision import transforms
from augmentation import get_augmentation_transform


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


def get_aug_types():
    return {
        "original": None,

        "flip": transforms.RandomHorizontalFlip(p=1.0),

        "rotation": transforms.RandomRotation(10),

        "color": transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),

        "blur": transforms.GaussianBlur(kernel_size=3),

        "noise": transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.01),
            transforms.ToPILImage()
        ]),

        # 🔥 full pipeline (paling penting untuk eksperimen)
        "combined": get_augmentation_transform()
    }


def generate_augmented_dataset(
    input_dir: str,
    output_dir: str,
    category: str = "bottle"
):
    src_path = os.path.join(input_dir, category, "train", "good")

    aug_types = get_aug_types()

    for aug_name, aug_transform in aug_types.items():
        print(f"Processing: {aug_name}")

        dst_folder = os.path.join(
            output_dir,
            category,
            "train",
            f"good_{aug_name}"
        )

        for img_name in os.listdir(src_path):
            if not img_name.endswith(".png"):
                continue

            img_path = os.path.join(src_path, img_name)
            img = Image.open(img_path).convert("RGB")

            if aug_transform:
                img_aug = aug_transform(img)
            else:
                img_aug = img

            save_path = os.path.join(dst_folder, img_name)
            save_image(img_aug, save_path)

    print("Augmented dataset created")


if __name__ == "__main__":
    generate_augmented_dataset(
        input_dir="dataset/mvtec_anomaly_detection",
        output_dir="dataset_augmented/mvtec_anomaly_detection"
    )