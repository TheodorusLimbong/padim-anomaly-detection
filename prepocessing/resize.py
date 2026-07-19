# Cara jalankan: (di-import oleh build_transform.py)
# Resize gambar ke 224x224
# Catatan: PaDiM paper pakai Resize(256) -> CenterCrop(224), ini langsung Resize(224)

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_resize_transform(img_size=224):
    return transforms.Resize(
        (img_size, img_size),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True
    )