import torch

def extract_features(model, dataloader, device):
    model.eval()
    features = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            output = model(images)
            features.append(output.cpu())

    return torch.cat(features, dim=0)