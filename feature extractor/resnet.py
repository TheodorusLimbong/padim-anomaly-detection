import torch.nn as nn
from torchvision.models import resnet50

def get_resnet():
    model = resnet50(pretrained=True)
    
    # Remove FC layer
    model = nn.Sequential(*list(model.children())[:-2])
    
    return model