import torch



def reshape_embedding(embedding):
    """
    Convert feature map into patch embeddings.

    Input:
        [B, C, H, W]

    Output:
        [B, H*W, C]
    """

    b, c, h, w = embedding.shape

    embedding = embedding.permute(0, 2, 3, 1)
    embedding = embedding.reshape(b, h * w, c)

    return embedding