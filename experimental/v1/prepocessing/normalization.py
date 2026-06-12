from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_normalize_transform():
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)