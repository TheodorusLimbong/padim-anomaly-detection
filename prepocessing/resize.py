from torchvision import transforms


def get_resize_transform(img_size=256):
    return transforms.Resize((img_size, img_size))
if __name__ == "__main__":
    from PIL import Image

    img = Image.open("dataset/mvtec_anomaly_detection/bottle/train/good/000.png")

    transform = get_resize_transform(256)
    resized = transform(img)

    print("Original size:", img.size)
    print("Resized size:", resized.size)