# Cara jalankan: (di-import oleh extract_feature.py)
# Load ResNet-50 dengan pretrained MoCo v2 weights dari Facebook Research
# Forward hook di layer1, layer2, layer3 untuk multi-layer feature extraction

import os
import torch
import torch.nn as nn
from torchvision.models import resnet50

MOCO_URL = (
    "https://dl.fbaipublicfiles.com/moco/"
    "moco_checkpoints/moco_v2_200ep/"
    "moco_v2_200ep_pretrain.pth.tar"
)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models"
)


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 + MoCo v2 pretrained weights.
    Output: dictionary layer1..layer4 untuk multi-layer feature extraction.
    
    Layer sizes:
      layer1: [B, 256, 56, 56]   (stride 4)
      layer2: [B, 512, 28, 28]   (stride 8)
      layer3: [B, 1024, 14, 14]  (stride 16)
      layer4: [B, 2048, 7, 7]    (stride 32, tidak dipakai)
    
    MoCo v2: self-supervised learning dari Facebook Research.
    Bobot didownload otomatis dari URL atau dari file lokal models/moco_v2_200ep_pretrain.pth.tar
    """
    def __init__(self, pretrained=True):
        super().__init__()

        model = resnet50(weights=None)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if pretrained:
            self._load_moco_weights()

    def _load_moco_weights(self):
        """
        Load MoCo v2 weights. Format checkpoint dari Facebook:
          state_dict keys: "module.encoder_q.layer1.0.conv1.weight" dst.
          Kita ambil yang prefix "module.encoder_q." saja, buang "fc" (classifier).
        """
        local_path = os.path.join(MODELS_DIR, "moco_v2_200ep_pretrain.pth.tar")
        if os.path.exists(local_path):
            print(f"[INFO] Loading MoCo v2 from local: {local_path}")
            checkpoint = torch.load(local_path, map_location="cpu")
        else:
            print("[INFO] Downloading MoCo v2 pretrained weights...")
            checkpoint = torch.hub.load_state_dict_from_url(
                MOCO_URL, map_location="cpu"
            )
        state_dict = checkpoint["state_dict"]

        # Map checkpoint keys ke model keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("module.encoder_q.") or k.startswith(
                "module.encoder_q.fc"
            ):
                continue
            name = k.replace("module.encoder_q.", "")
            new_state_dict[name] = v

        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys ({len(missing)}): "
                  f"{[k for k in missing if 'num_batches' not in k][:5]}...")
        if unexpected:
            print(f"[WARN] Unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:3]}...")
        print("[INFO] MoCo v2 weights loaded successfully")

    def forward(self, x):
        """Forward pass, return dictionary of all 4 layers."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)  # [B, 256, 56, 56]
        out2 = self.layer2(out1)  # [B, 512, 28, 28]
        out3 = self.layer3(out2)  # [B, 1024, 14, 14]
        out4 = self.layer4(out3)  # [B, 2048, 7, 7]

        return {
            "layer1": out1,
            "layer2": out2,
            "layer3": out3,
            "layer4": out4,
        }
