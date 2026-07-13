
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch
import math
import torch.nn.functional as F
from ..utils.attention_module import RelPosMultiHeadSelfAttention, DeformableMultiHeadedAttention, ContextualMultiHeadAttention, MultiHeadAttention


class CoPE(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(
            torch.zeros(1, head_dim, npos_max)
        )

    def forward(self, query, attn_logits):
        # compute positions
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)



class ContextualMultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, causal=False, max_pos_encoding=5000):
        super(ContextualMultiHeadedSelfAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = ContextualMultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        inputs = self.layer_norm(inputs)
        x = self.attention(inputs, inputs, inputs, mask)
        return self.dropout(x)

class ContextualMultiHeadedCrossAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, causal=False, max_pos_encoding=5000):
        super(ContextualMultiHeadedCrossAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = ContextualMultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, memory: Tensor, mask: Optional[Tensor] = None):
        inputs = self.layer_norm(inputs)
        x = self.attention(inputs, memory, memory, mask)
        return self.dropout(x)

class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, causal=False, max_pos_encoding=5000):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, Q: Tensor, mask: Optional[Tensor] = None):
        q = self.layer_norm(Q)
        x, _ = self.attention(q, Q,  Q, mask)
        return self.dropout(x)

class RelativeMultiheadSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, causal=False, max_pos_encoding=5000):
        super(RelativeMultiheadSelfAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelPosMultiHeadSelfAttention(d_model, num_heads, causal, max_pos_encoding)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, Q: Tensor, mask: Optional[Tensor] = None):
        q = self.layer_norm(Q)
        x, _, _ = self.attention(q, Q,  Q, mask)
        return self.dropout(x)

class GlossFreeAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, cope=False, causal=False, max_pos_encoding=5000):
        super(GlossFreeAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = DeformableMultiHeadedAttention(query_type='attention',
            size=d_model,
            query_nb=7,
            num_heads=num_heads,
            cope=cope)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        inputs = self.layer_norm(inputs)
        x = self.attention(inputs, inputs, inputs, mask)
        return self.dropout(x)

class MultiHeadedCrossAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, causal=False, max_pos_encoding=5000):
        super(MultiHeadedCrossAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None):
        q = self.layer_norm(Q)
        x, _ = self.attention(q, K,  V, mask)
        return self.dropout(x)