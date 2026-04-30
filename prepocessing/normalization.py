from torchvision import transforms


def get_normalize_transform():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms

    img = Image.open("dataset/mvtec_anomaly_detection/bottle/train/good/000.png")

    transform = transforms.Compose([
        transforms.ToTensor(),
        get_normalize_transform()
    ])

    tensor = transform(img)

    print("Tensor shape:", tensor.shape)
    print("Min:", tensor.min().item())
    print("Max:", tensor.max().item())