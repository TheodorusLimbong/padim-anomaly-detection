from torchvision import transforms
from preprocessing.resize import get_resize_transform
from preprocessing.augmentation import get_augmentation_transform
from preprocessing.normalize import get_normalize_transform


def build_transform(img_size=256, is_train=True):

    transform_list = [get_resize_transform(img_size)]

    if is_train:
        transform_list.append(get_augmentation_transform())

    transform_list.extend([
        transforms.ToTensor(),
        get_normalize_transform()
    ])

    return transforms.Compose(transform_list)