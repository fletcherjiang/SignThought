import torch
import torch.nn as nn
from torch import Tensor

from main.transformer_layers import MultiHeadedAttention


class ThinkingLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        ff_size: int = 1024,
        dropout: float = 0.1,
        causal_self_attn: bool = True,
    ) -> None:
        super().__init__()
        self.causal_self_attn = causal_self_attn

        self.self_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ff_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.self_attn = MultiHeadedAttention(
            num_heads=num_heads, size=hidden_size, dropout=dropout
        )
        self.cross_attn = MultiHeadedAttention(
            num_heads=num_heads, size=hidden_size, dropout=dropout
        )

        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, thoughts: Tensor, memory: Tensor, src_mask: Tensor = None
    ) -> Tensor:
        """
        :param thoughts: [B, K, D]
        :param memory: encoder_output [B, Ts, D]
        :param src_mask: [B, 1, Ts] or [B, Ts]
        """
        bsz, steps, _ = thoughts.size()

        # 因果自注意力（可选）
        self_mask = None
        if self.causal_self_attn:
            causal = torch.tril(
                torch.ones(steps, steps, device=thoughts.device, dtype=torch.bool)
            )
            self_mask = causal.unsqueeze(0).expand(bsz, -1, -1)

        t = self.self_norm(thoughts)
        self_out = self.self_attn(t, t, t, mask=self_mask)
        thoughts = thoughts + self.dropout(self_out)

        # 思维对视频的交叉注意力
        c = self.cross_norm(thoughts)
        cross_out = self.cross_attn(c, memory, memory, mask=src_mask)
        thoughts = thoughts + self.dropout(cross_out)

        # FFN
        f = self.ff_norm(thoughts)
        f = self.ff(f)
        thoughts = thoughts + f

        return thoughts


class LatentCoT(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        ff_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
        K: int = 6,
        causal_self_attn: bool = True,
        num_segments: int = 1,
        sinkhorn_iters: int = 3,
        sinkhorn_tau: float = 1.0,
        segmentation_temperature: float = 0.1,
        retrieval_mode: str = "dense",
        sparse_topk: int = 0,
        deformable_radius: int = 1,
        enable_decoder_routing: bool = False,
        routing_bias_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.K = K

        # 分段与对齐相关超参（保留配置，不参与当前前向计算）
        self.num_segments = num_segments
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.segmentation_temperature = segmentation_temperature
        self.retrieval_mode = retrieval_mode
        self.sparse_topk = sparse_topk
        self.deformable_radius = deformable_radius
        self.enable_decoder_routing = enable_decoder_routing
        self.routing_bias_scale = routing_bias_scale

        self.thought_queries = nn.Parameter(torch.randn(K, hidden_size))
        self.layers = nn.ModuleList(
            [
                ThinkingLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ff_size=ff_size,
                    dropout=dropout,
                    causal_self_attn=causal_self_attn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, encoder_output: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        :param encoder_output: [B, Ts, D]
        :param src_mask: [B, 1, Ts] or [B, Ts]
        :return: thought_states [B, K, D]
        """
        bsz = encoder_output.size(0)
        thoughts = self.thought_queries.unsqueeze(0).expand(bsz, -1, -1)

        mask = src_mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            mask = mask.bool()

        for layer in self.layers:
            thoughts = layer(thoughts, encoder_output, mask)

        return thoughts

    def __repr__(self):
        return (
            f"LatentCoT("
            f"K={self.K}, "
            f"hidden_size={self.hidden_size}, "
            f"layers={len(self.layers)}, "
            f"num_segments={self.num_segments}, "
            f"retrieval_mode={self.retrieval_mode}, "
            f"enable_decoder_routing={self.enable_decoder_routing})"
        )
