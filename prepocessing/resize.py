from torchvision import transforms
from PIL import Image
import os


def get_resize_transform(img_size=256):
    return transforms.Resize((img_size, img_size))


if __name__ == "__main__":
    img_path = "dataset/mvtec_anomaly_detection/bottle/train/good/000.png"

    # ===== VALIDASI PATH =====
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # ===== LOAD IMAGE =====
    img = Image.open(img_path).convert("RGB")

    # ===== APPLY TRANSFORM =====
    transform = get_resize_transform(244)
    resized = transform(img)

    # ===== OUTPUT =====
    print("===== RESIZE TEST =====")
    print(f"Original size : {img.size}")      # (W, H)
    print(f"Resized size  : {resized.size}")  # (W, H)
    print(f"Mode          : {resized.mode}")  # RGB