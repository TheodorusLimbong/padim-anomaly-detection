import os
from typing import List
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


def get_normalize_transform() -> transforms.Normalize:
    """
    Normalize image tensor using ImageNet statistics.
    Required for pretrained CNN (ResNet/MoCo).
    """
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ================= TEST =================
if __name__ == "__main__":
    img_path = "dataset/mvtec_anomaly_detection/bottle/train/good/000.png"

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # ===== LOAD IMAGE =====
    img = Image.open(img_path).convert("RGB")

    # ===== PIPELINE =====
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        get_normalize_transform()
    ])

    tensor = transform(img)

    # ===== OUTPUT =====
    print("===== NORMALIZE TEST =====")
    print(f"Shape : {tensor.shape}")          # [3, 256, 256]
    print(f"Min   : {tensor.min().item():.4f}")
    print(f"Max   : {tensor.max().item():.4f}")
    print(f"Mean  : {tensor.mean().item():.4f}")
    print(f"Std   : {tensor.std().item():.4f}")
    print(f"Dtype : {tensor.dtype}")