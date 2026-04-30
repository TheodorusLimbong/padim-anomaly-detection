from torchvision import transforms


def get_augmentation_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
    ])
if __name__ == "__main__":
    from PIL import Image

    img = Image.open("dataset/mvtec_anomaly_detection/bottle/train/good/000.png")

    aug = get_augmentation_transform()
    aug_img = aug(img)

    print("Augmentation applied successfully")