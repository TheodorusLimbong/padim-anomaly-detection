# Cara jalankan: (di-import oleh dataset_wrapper.py dan semua varian pipeline)
#
# TRAIN transform:   resize(224) -> augmentasi (Flip, Rotation, ColorJitter, Noise) -> normalize
# TEST transform:    resize(224) -> normalize (TANPA augmentasi)
#
# Catatan: PaDiM paper TIDAK pakai augmentasi untuk standard MVTec AD.
# Tapi implementasi ini default PAKAI augmentasi (bisa dimatikan di v1/ atau subset --no-aug)

from torchvision import transforms
from resize import get_resize_transform
from augmentation.augmentation import get_augmentation_transform
from normalization import get_normalize_transform


def build_transform(img_size=224, is_train=True):
    """
    Bangun transform pipeline untuk training atau test.
    
    Training: resize -> augmentasi (flip, rotation, color jitter, noise) -> normalize
    Test:     resize -> normalize (tanpa augmentasi)
    """
    if is_train:
        return transforms.Compose([
            get_resize_transform(img_size),
            get_augmentation_transform(),
            get_normalize_transform(),
        ])
    else:
        return transforms.Compose([
            get_resize_transform(img_size),
            get_normalize_transform(),
        ])
    else:
        return transforms.Compose([
            get_resize_transform(img_size),
            transforms.ToTensor(),  # ✔ hanya di test
            get_normalize_transform()
        ])