import os
from typing import Tuple
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_resize_transform(
    img_size: int = 256,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: bool = True
) -> transforms.Resize:
    """
    Resize transform for input images.

    Args:
        img_size (int): target size (H, W)
        interpolation (InterpolationMode): resize method
        antialias (bool): apply anti-aliasing (recommended)

    Returns:
        transforms.Resize
    """
    return transforms.Resize(
        size=(img_size, img_size),
        interpolation=interpolation,
        antialias=antialias
    )


# ================= TEST =================
if __name__ == "__main__":
    img_path = "dataset/mvtec_anomaly_detection/bottle/train/good/000.png"

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load image (always RGB for CNN)
    img = Image.open(img_path).convert("RGB")

    # Apply resize (use 256 for consistency)
    transform = get_resize_transform(img_size=256)
    resized = transform(img)

    print("===== RESIZE TEST =====")
    print(f"Original size : {img.size}")        # (W, H)
    print(f"Resized size  : {resized.size}")    # (W, H)
    print(f"Mode          : {resized.mode}")    # RGB
    print(f"Type          : {type(resized)}")   # PIL.Image.Image