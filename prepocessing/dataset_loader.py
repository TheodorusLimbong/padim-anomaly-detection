from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

def load_dataset(path, train=True):
    dataset = datasets.ImageFolder(
        root=path,
        transform=get_transforms(train)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=NUM_WORKERS
    )
    
    return loader