from torchvision import transforms
from prepocessing.resize import get_resize_transform
from prepocessing.normalization import get_normalize_transform
from prepocessing.augmentation import get_augmentation_transform


def build_transform(img_size=256, is_train=True):
    resize = get_resize_transform(img_size)
    normalize = get_normalize_transform()

    transform_list = [resize]

    if is_train:
        transform_list.append(get_augmentation_transform())

    transform_list.extend([
        transforms.ToTensor(),
        normalize
    ])

    return transforms.Compose(transform_list)


# ================= TEST =================
if __name__ == "__main__":
    from PIL import Image

    img = Image.open("dataset/mvtec_anomaly_detection/bottle/train/good/000.png").convert("RGB")

    train_transform = build_transform(is_train=True)
    test_transform = build_transform(is_train=False)

    train_out = train_transform(img)
    test_out = test_transform(img)

    print("Train shape:", train_out.shape)
    print("Test shape :", test_out.shape)