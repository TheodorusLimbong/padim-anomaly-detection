from torch.utils.data import DataLoader
from preprocessing.dataset_wrapper import MVTecDataset


def get_dataloader(root_dir, phase="train", batch_size=8):
    dataset = MVTecDataset(root_dir=root_dir, phase=phase)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == "train")
    )