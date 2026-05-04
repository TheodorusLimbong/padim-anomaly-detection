from torchvision import transforms
from resize import get_resize_transform
from augmentation.augmentation import get_augmentation_transform
from normalization import get_normalize_transform


def build_transform(img_size=256, is_train=True):

    if is_train:
        return transforms.Compose([
            get_resize_transform(img_size),
            get_augmentation_transform(),  # ✔ sudah include ToTensor
            get_normalize_transform()
        ])
    else:
        return transforms.Compose([
            get_resize_transform(img_size),
            transforms.ToTensor(),  # ✔ hanya di test
            get_normalize_transform()
        ])