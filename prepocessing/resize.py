from torchvision import transforms


def get_resize_transform(img_size=256):
    return transforms.Resize((img_size, img_size))