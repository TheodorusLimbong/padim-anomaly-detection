from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_resize_transform(img_size=224):
    return transforms.Resize(
        (img_size, img_size),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True
    )