# Cara jalankan: (di-import oleh backbone.py)
# Forward hooks: tangkap output layer1(256ch,56x56), layer2(512ch,28x28), layer3(1024ch,14x14)
# embedding_concat: upsample layer2(28->56) + layer3(14->56) -> concat -> [N,1792,56,56]

import torch
import torch.nn.functional as F


def embedding_concat(x, y):
    """
    Gabungkan 2 feature map dengan resolusi berbeda.
    Digunakan untuk multi-layer feature concatenation di PaDiM.
    
    Contoh: x = layer1 [B, 256, 56, 56], y = layer2 [B, 512, 28, 28]
    - Unfold x dengan kernel 2x2 -> [B, 256*4, 28, 28]
    - Concatenate dengan y -> [B, 256*4+512, 28, 28]
    - Fold kembali ke 56x56 -> [B, (256+512), 56, 56]
    
    Hasil: layer1(256ch) + layer2(512ch) + layer3(1024ch) = 1792 channel di 56x56
    """
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)  # scaling factor (misal 56/28=2 atau 56/14=4)

    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)

    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2, device=x.device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)

    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


class FeatureExtractor:
    """
    Feature extractor dengan forward hooks di layer terpilih.
    Menggabungkan multi-layer features jadi 1 tensor.

    Output: [B, 1792, 56, 56] (256+512+1024 = 1792 channel)
    """
    def __init__(self, model, selected_layers):
        self.model = model
        self.selected_layers = selected_layers

    def extract(self, images):
        with torch.no_grad():
            outputs = self.model(images)  # dict layer1..layer4

        # Mulai dari layer pertama, concat layer berikutnya
        features = outputs[self.selected_layers[0]]
        for layer_name in self.selected_layers[1:]:
            features = embedding_concat(features, outputs[layer_name])

        return features  # [B, 1792, 56, 56]
