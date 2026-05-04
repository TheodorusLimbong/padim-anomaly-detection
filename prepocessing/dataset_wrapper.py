from torch.utils.data import Dataset
from PIL import Image

from preprocessing.load_dataset import load_mvtec_paths
from preprocessing.build_transform import build_transform


class MVTecDataset(Dataset):
    def __init__(self, root_dir, category="bottle", phase="train"):
        self.img_paths, self.labels, self.mask_paths = load_mvtec_paths(
            root_dir, category, phase
        )

        self.transform = build_transform(is_train=(phase == "train"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "label": self.labels[idx],
            "mask": self.mask_paths[idx]
        }