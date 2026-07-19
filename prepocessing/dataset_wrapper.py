# Cara jalankan: (di-import oleh run_padim.py)
# Dataset wrapper untuk MVTec AD. Load gambar + label + ground truth mask.
# Train transform: resize + augmentasi + normalize (dari build_transform.py)
# Test transform: resize + normalize (tanpa augmentasi)

from torch.utils.data import Dataset
from PIL import Image
import torch
import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from load_dataset import load_mvtec_paths
from build_transform import build_transform


class MVTecDataset(Dataset):
    def __init__(self, root_dir, category="bottle", phase="train"):
        self.phase = phase

        self.img_paths, self.labels, self.mask_paths = load_mvtec_paths(
            root_dir, category, phase
        )

        self.transform = build_transform(is_train=(phase == "train"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)

        sample = {
            "image": img,
            "label": self.labels[idx],
        }

        if self.phase != "train":
            mask_path = self.mask_paths[idx]


            if mask_path is not None:
                mask = Image.open(mask_path).convert("L")
                mask = transforms.Resize(
                    (224, 224),
                    interpolation=InterpolationMode.NEAREST 
                )(mask)
                mask = transforms.ToTensor()(mask)
            else:
                mask = torch.zeros(1, img.shape[1], img.shape[2])

            sample["mask"] = mask

        return sample