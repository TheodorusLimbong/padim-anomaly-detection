from torchvision import transforms
from PIL import Image
import os


def get_normalize_transform():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


if __name__ == "__main__":
    img_path = "dataset/mvtec_anomaly_detection/bottle/train/good/000.png"

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # ===== LOAD IMAGE =====
    img = Image.open(img_path).convert("RGB")

    # ===== PIPELINE =====
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        get_normalize_transform()
    ])

    tensor = transform(img)

    # ===== OUTPUT =====
    print("===== NORMALIZE TEST =====")
    print("Shape :", tensor.shape)  # [3, 256, 256]
    print("Min   :", tensor.min().item())
    print("Max   :", tensor.max().item())
    print("Mean  :", tensor.mean().item())