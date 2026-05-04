from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_augmentation_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            )
        ], p=0.5),

        transforms.RandomRotation(
            degrees=5,
            interpolation=InterpolationMode.BILINEAR
        ),
    ])