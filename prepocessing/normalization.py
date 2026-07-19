# Cara jalankan: (di-import oleh build_transform.py)
# Normalisasi ImageNet: mean [0.485,0.456,0.406], std [0.229,0.224,0.225]
# Ini standard untuk ResNet-50 (pretrained on ImageNet)

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_normalize_transform():
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)