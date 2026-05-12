import torch
import torch.nn.functional as F


class FeatureExtractor:
    def __init__(self, model, selected_layers):
        self.model = model
        self.selected_layers = selected_layers

    def extract(self, images):
        with torch.no_grad():
            outputs = self.model(images)

        features = []

        target_size = outputs[self.selected_layers[0]].shape[-2:]

        for layer_name in self.selected_layers:
            feature = outputs[layer_name]

            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False
                )

            features.append(feature)

        embedding = torch.cat(features, dim=1)

        return embedding