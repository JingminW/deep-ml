import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias = False)
        self.k_proj = nn.Linear(d_model, d_model, bias = False)
        self.v_proj = nn.Linear(d_model, d_model, bias = False)
        self.out_proj = nn.Linear(d_model, d_model, bias = False)

        self.d_head = d_model // num_heads

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(batch_size, seq_len, -1, self.d_head).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, -1, self.d_head).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, -1, self.d_head).permute(0, 2, 1, 3)

        attention = torch.einsum("bhqd,bhkd->bhqk", Q, K)
        if mask is not None:
            attention = attention + mask
        attention = attention / (self.d_head ** (1/2))

        attention = torch.softmax(attention, dim = -1)
        attention = torch.einsum("bhqk,bhkd->bhqd", attention, V)
        
        attention = attention.permute(0, 2, 1, 3)
        attention = attention.reshape(batch_size, seq_len, -1)

        output = self.out_proj(attention)
        return output





