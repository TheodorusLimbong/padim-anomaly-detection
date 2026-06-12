import torch
import torch.nn.functional as F


def embedding_concat(x, y):
    """Align multi-resolution features via unfold+fold (PaDiM original)."""
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)

    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)

    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2, device=x.device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)

    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


class FeatureExtractor:
    def __init__(self, model, selected_layers):
        self.model = model
        self.selected_layers = selected_layers

    def extract(self, images):
        with torch.no_grad():
            outputs = self.model(images)

        features = outputs[self.selected_layers[0]]
        for layer_name in self.selected_layers[1:]:
            features = embedding_concat(features, outputs[layer_name])

        return features
