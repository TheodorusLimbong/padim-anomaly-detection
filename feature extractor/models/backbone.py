import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        model = resnet50(weights="IMAGENET1K_V1" if pretrained else None)

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.stem(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return {
            "layer1": out1,
            "layer2": out2,
            "layer3": out3,
            "layer4": out4,
        }