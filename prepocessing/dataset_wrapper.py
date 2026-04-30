from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

from .load_dataset import load_mvtec_paths
from .build_transform import build_transform


class MVTecDataset(Dataset):
    def __init__(
        self,
        root_dir,
        category="bottle",
        phase="train",
        img_size=256,
        use_augmentation=True
    ):
        self.img_paths, self.labels, self.mask_paths = load_mvtec_paths(
            root_dir, category, phase
        )

        self.transform = build_transform(
            img_size=img_size,
            is_train=(phase == "train"),
            use_augmentation=use_augmentation
        )

        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = transforms.Resize((self.img_size, self.img_size))(mask)
            mask = transforms.ToTensor()(mask)
        else:
            mask = None

        return {
            "image": image,
            "label": label,
            "mask": mask,
            "path": img_path
        }