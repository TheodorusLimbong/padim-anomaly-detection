from torchvision import transforms
from resize import get_resize_transform
from normalization import get_normalize_transform


def build_transform(img_size=224, is_train=True):
    return transforms.Compose([
        get_resize_transform(img_size),
        transforms.ToTensor(),
        get_normalize_transform()
    ])
