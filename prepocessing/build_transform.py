from torchvision import transforms
from .resize import get_resize_transform
from .normalize import get_normalize_transform
from .augmentation import get_augmentation_transform

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def build_transform(img_size=256, is_train=True, use_augmentation=True):
    resize = get_resize_transform(img_size)
    normalize = get_normalize_transform()

    transform_list = [resize]

    if is_train and use_augmentation:
        transform_list.append(get_augmentation_transform())

    transform_list.extend([
        transforms.ToTensor(),
        normalize
    ])

    return transforms.Compose(transform_list)
if __name__ == "__main__":
    from PIL import Image

    img = Image.open("dataset/mvtec_anomaly_detection/bottle/train/good/000.png")

    transform = build_transform(img_size=256, is_train=True)

    output = transform(img)

    print("Final tensor shape:", output.shape)