"""Debug: Check which test image indices map to which files."""
import sys, os
sys.path.insert(0, r"D:\skripsi\self supervised\code")
sys.path.insert(0, os.path.join(r"D:\skripsi\self supervised\code", "prepocessing"))

from load_dataset import load_mvtec_paths
from src.config import DATASET_PATH

root_dir = os.path.dirname(DATASET_PATH)
category = os.path.basename(DATASET_PATH)

img_paths, labels, _ = load_mvtec_paths(root_dir, category, "test")

# Group by defect type
by_defect = {}
for i, (p, l) in enumerate(zip(img_paths, labels)):
    parts = p.replace("\\", "/").split("/")
    defect_type = parts[-2]
    fname = parts[-1]
    if defect_type not in by_defect:
        by_defect[defect_type] = []
    by_defect[defect_type].append((i, fname, l))

for dt, items in by_defect.items():
    print(f"\n{dt} (label={items[0][2]}):")
    for idx, fname, label in items:
        print(f"  [{idx}] {fname}")
