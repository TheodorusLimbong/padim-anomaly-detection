# Cara jalankan: python scripts/gen_augmentation_figure.py
# Generate visualisasi 5 jenis augmentasi untuk BAB 4 skripsi
# Output: output/figures/hasil_augmentasi.png

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


def main():
    img_path = r"D:\skripsi\self supervised\code\dataset\mvtec_anomaly_detection\bottle\train\good\000.png"
    out_path = r"D:\skripsi\self supervised\code\output\figures\hasil_augmentasi.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = Image.open(img_path).convert("RGB")

    resize = transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)

    noise_transform = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.01),
        transforms.ToPILImage(),
    ])

    augs = {
        "Original": None,
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),
        "RandomRotation": transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
        "ColorJitter": transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        "AddGaussianNoise": noise_transform,
    }

    keys = list(augs.keys())
    images = []
    titles = []
    for key in keys:
        aug_fn = augs[key]
        img_resized = resize(img)
        if aug_fn is not None:
            img_aug = aug_fn(img_resized)
        else:
            img_aug = img_resized
        images.append(img_aug)
        titles.append(key)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("Visualisasi Berbagai Augmentasi pada Citra Botol", fontsize=14, y=0.98)

    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    for idx, (row, col) in enumerate(positions):
        ax = axes[row][col]
        ax.imshow(images[idx])
        ax.set_title(titles[idx], fontsize=10, pad=6)
        ax.axis("off")

    axes[1][2].axis("off")

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
