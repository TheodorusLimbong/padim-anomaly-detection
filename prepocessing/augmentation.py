import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_augmentation_transform(
    flip_prob: float = 0.5,
    jitter_prob: float = 0.5,
    rotation_deg: int = 5
) -> transforms.Compose:
    """
    Lightweight augmentation for anomaly detection (PaDiM/SSL).

    Notes:
    - Keep augmentation mild to preserve normal distribution
    - Avoid strong geometric distortion
    """

    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=flip_prob),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            )
        ], p=jitter_prob),

        transforms.RandomRotation(
            degrees=rotation_deg,
            interpolation=InterpolationMode.BILINEAR
        ),
    ])


# ================= TEST =================
if __name__ == "__main__":
    img_path = "dataset/mvtec_anomaly_detection/bottle/train/good/000.png"

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")

    aug = get_augmentation_transform()

    print("===== AUGMENTATION TEST =====")
    for i in range(3):
        aug_img = aug(img)
        print(f"Augmentation {i+1} applied | Size: {aug_img.size} | Mode: {aug_img.mode}")