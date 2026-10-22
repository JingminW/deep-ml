import numpy as np

def vit_encode(image, patch_size: int, W_patch, cls_token, pos_embed, W_qkv, W_o, num_heads: int):
    """
    Forward pass of a minimal ViT encoder block.

    Returns the final embedding of the CLS token as a list of length D.
    """
    image = np.array(image)
    W_patch = np.array(W_patch)
    cls_token = np.array(cls_token)
    W_qkv = np.array(W_qkv)
    W_o = np.array(W_o)

    H, W, C = image.shape
    P = patch_size
    D = W_patch.shape[1]
    N = (H // P) * (W // P)
    L = N + 1
    d_h = D // num_heads

    patches = image.reshape(H // P, P, W // P, P, C)
    patches = patches.transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(N, P * P * C)

    patch_embeddings = patches @ W_patch
    tokens = np.concatenate([cls_token[None, :], patch_embeddings], axis=0)
    tokens = tokens + pos_embed

    qkv = tokens @ W_qkv
    Q, K, V = np.split(qkv, 3, axis = -1)

    Q = Q.reshape(L, num_heads, d_h).transpose(1, 0, 2)
    K = K.reshape(L, num_heads, d_h).transpose(1, 0, 2)
    V = V.reshape(L, num_heads, d_h).transpose(1, 0, 2)

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_h)
    scores = scores - np.max(scores, axis = -1, keepdims=True)
    attention = np.exp(scores)
    attention /= np.sum(attention, axis = -1, keepdims = True)

    output = attention @ V
    output = output.transpose(1, 0, 2).reshape(L, D)
    output = output @ W_o
    return output[0].tolist()




