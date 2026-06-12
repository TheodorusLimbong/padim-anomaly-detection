from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


def get_augmentation_transform():
    return transforms.Compose([

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=10,
            interpolation=InterpolationMode.BILINEAR
        ),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )
        ], p=0.7),

        transforms.ToTensor(),

        transforms.RandomApply([
            AddGaussianNoise(0., 0.01)
        ], p=0.3),
    ])