import os
import torch

def save_features(features, save_path, file_name):
    os.makedirs(save_path, exist_ok=True)

    full_path = os.path.join(save_path, file_name)

    torch.save(features, full_path)

    print(f"[INFO] Features saved to: {full_path}")