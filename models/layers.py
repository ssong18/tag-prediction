import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules import GELU, LayerNorm


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(embed_dim, embed_dim)
        self.w2 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x, attn_mask=None):
        """
        :param x: type:(torch.Tensor) shape:(N:batch_size, S:seq_len, E:embed_dim)
        :return x: type:(torch.Tensor) shape:(N, S, E:input_dim)
        """
        N, S, E = x.shape
        x = x.reshape(N*S, E)
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = x.reshape(N, S, E)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.w_qs = nn.Linear(embed_dim, embed_dim)
        self.w_ks = nn.Linear(embed_dim, embed_dim)
        self.w_vs = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        :param x: (N, S, E)
        :param attn_mask: (N, S, S) (torch.bool)
        :return:
        """
        N, S, E = x.shape
        # Pass through pre-attention projection

        q = self.w_qs(x.reshape(-1, E)).reshape(N, S, self.num_heads, self.d_k)
        k = self.w_ks(x.reshape(-1, E)).reshape(N, S, self.num_heads, self.d_k)
        v = self.w_vs(x.reshape(-1, E)).reshape(N, S, self.num_heads, self.d_k)

        # Transpose to not attend different heads
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Attention
        attn_logits = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.d_k)  # (N, H, S, S)
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask.float().unsqueeze(1) == 1, -1e9)
        attn_weights = torch.exp(F.log_softmax(attn_logits, dim=-1))

        o = torch.matmul(attn_weights, v)
        o = o.permute(0, 2, 1, 3)
        o = o.reshape(N, S, E)
        o = self.w_o(o)

        return o


class SetEncoder(nn.Module):
    def __init__(self, layer_type, embed_dim, num_heads, dropout):
        super(SetEncoder, self).__init__()
        self.norm = LayerNorm(embed_dim, eps=1e-6)
        if layer_type == 'rFF':
            self.layer = FeedForward(embed_dim=embed_dim,
                                     dropout=dropout)
        elif layer_type == 'rSA':
            self.layer = MultiHeadAttention(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        """
        :param x: type:(torch.Tensor) shape:(S:seq_len, N:batch_size, I:input_dim)
        :param attn_mask: type:(torch.Tensor) shape:()
        :return x: type:(torch.Tensor) shape:(S, N, I)
        """
        y = self.norm(x)
        y = self.layer(y, attn_mask=attn_mask)
        x = x + self.dropout(y)

        return x
