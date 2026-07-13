import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from main.transformer_layers import MultiHeadedAttention


def _normalize_src_mask(src_mask: Tensor, length: int) -> Tensor:
    if src_mask is None:
        raise ValueError("src_mask is required for routed temporal evidence")
    if src_mask.dim() == 3:
        src_mask = src_mask.squeeze(1)
    if src_mask.dim() != 2:
        raise ValueError("src_mask must have shape [B, T] or [B, 1, T]")
    return src_mask[:, :length].bool()


class SoftTemporalSegmentation(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_segments: int,
        temperature: float = 1.0,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.temperature = temperature
        self.eps = eps
        self.boundary_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_segments),
        )

    def forward(self, memory: Tensor, src_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Build M differentiable segment tokens from frame evidence.

        :param memory: encoder output [B, T, D]
        :param src_mask: source mask [B, 1, T] or [B, T]
        :return: segment_tokens [B, M, D], segment_weights [B, M, T]
        """
        batch_size, src_len, _ = memory.size()
        mask = _normalize_src_mask(src_mask, src_len)
        mask_f = mask.to(dtype=memory.dtype)
        valid_lengths = mask_f.sum(dim=-1).clamp_min(1.0)

        summary = (memory * mask_f.unsqueeze(-1)).sum(dim=1)
        summary = summary / valid_lengths.unsqueeze(-1).clamp_min(self.eps)

        lengths = F.softplus(self.boundary_mlp(summary)) + self.eps
        proportions = lengths / lengths.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        cumulative = torch.cumsum(proportions, dim=-1)

        starts = torch.cat(
            [memory.new_zeros(batch_size, 1), cumulative[:, :-1]], dim=-1
        )
        ends = cumulative
        valid_span = (valid_lengths - 1.0).unsqueeze(-1)
        start_boundaries = 1.0 + valid_span * starts
        end_boundaries = 1.0 + valid_span * ends

        frame_ranks = torch.cumsum(mask_f, dim=-1)
        temperature = max(float(self.temperature), self.eps)
        left = torch.sigmoid(
            (frame_ranks.unsqueeze(1) - start_boundaries.unsqueeze(-1)) / temperature
        )
        right = torch.sigmoid(
            (frame_ranks.unsqueeze(1) - end_boundaries.unsqueeze(-1)) / temperature
        )
        windows = (left - right).clamp_min(0.0) * mask_f.unsqueeze(1)

        denom = windows.sum(dim=-1, keepdim=True)
        uniform = mask_f.unsqueeze(1) / valid_lengths.view(batch_size, 1, 1)
        segment_weights = torch.where(
            denom > self.eps,
            windows / denom.clamp_min(self.eps),
            uniform.expand(-1, self.num_segments, -1),
        )
        segment_tokens = torch.bmm(segment_weights, memory)
        return segment_tokens, segment_weights


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
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.self_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ff_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.self_attn = MultiHeadedAttention(
            num_heads=num_heads, size=hidden_size, dropout=dropout
        )
        self.cross_attn = MultiHeadedAttention(
            num_heads=num_heads, size=hidden_size, dropout=dropout
        )

        self.binding_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.binding_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_p = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_e = nn.Linear(hidden_size, hidden_size, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        thoughts: Tensor,
        memory: Tensor,
        src_mask: Tensor = None,
        segment_tokens: Tensor = None,
        segment_weights: Tensor = None,
        sinkhorn_iters: int = 3,
        sinkhorn_tau: float = 1.0,
        retrieval_mode: str = "dense",
        sparse_topk: int = 0,
        deformable_radius: int = 1,
        routing_bias_scale: float = 1.0,
        eps: float = 1.0e-6,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
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
        routing_cache = None
        cross_bias = None
        if segment_tokens is not None and segment_weights is not None:
            binding = self._sinkhorn_binding(
                thoughts=c,
                segment_tokens=segment_tokens,
                iters=sinkhorn_iters,
                tau=sinkhorn_tau,
            )
            routed_summary = torch.bmm(binding, segment_tokens)
            temporal_prior = torch.bmm(binding, segment_weights)
            cross_bias = self._routed_attention_bias(
                routed_summary=routed_summary,
                memory=memory,
                temporal_prior=temporal_prior,
                retrieval_mode=retrieval_mode,
                sparse_topk=sparse_topk,
                deformable_radius=deformable_radius,
                scale=routing_bias_scale,
                eps=eps,
            )
            routing_cache = {
                "binding": binding,
                "routed_summary": routed_summary,
                "temporal_prior": temporal_prior,
            }

        cross_out = self.cross_attn(
            c, memory, memory, mask=src_mask, attn_bias=cross_bias
        )
        thoughts = thoughts + self.dropout(cross_out)

        # FFN
        f = self.ff_norm(thoughts)
        f = self.ff(f)
        thoughts = thoughts + f

        return thoughts, routing_cache

    def _sinkhorn_binding(
        self,
        thoughts: Tensor,
        segment_tokens: Tensor,
        iters: int,
        tau: float,
    ) -> Tensor:
        q = self.binding_q(thoughts)
        k = self.binding_k(segment_tokens)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))

        if iters <= 0:
            return F.softmax(scores / max(float(tau), 1.0e-6), dim=-1)

        log_a = scores / max(float(tau), 1.0e-6)
        col_budget = thoughts.size(1) / float(segment_tokens.size(1))
        for _ in range(iters):
            log_a = log_a - torch.logsumexp(log_a, dim=-1, keepdim=True)
            log_a = (
                log_a
                - torch.logsumexp(log_a, dim=-2, keepdim=True)
                + math.log(col_budget)
            )
        log_a = log_a - torch.logsumexp(log_a, dim=-1, keepdim=True)
        return torch.exp(log_a)

    def _routed_attention_bias(
        self,
        routed_summary: Tensor,
        memory: Tensor,
        temporal_prior: Tensor,
        retrieval_mode: str,
        sparse_topk: int,
        deformable_radius: int,
        scale: float,
        eps: float,
    ) -> Tensor:
        if scale == 0:
            return None

        batch_size, num_thoughts, _ = routed_summary.size()
        src_len = memory.size(1)
        p = self.bias_p(routed_summary)
        e = self.bias_e(memory)
        p = p.view(batch_size, num_thoughts, self.num_heads, self.head_size)
        e = e.view(batch_size, src_len, self.num_heads, self.head_size)
        p = p.transpose(1, 2)
        e = e.transpose(1, 2)
        content_bias = torch.matmul(p, e.transpose(2, 3)) / math.sqrt(self.head_size)
        temporal_bias = torch.log(temporal_prior.clamp_min(eps)).unsqueeze(1)
        bias = content_bias + temporal_bias

        retrieval_mask = self._retrieval_mask(
            temporal_prior=temporal_prior,
            mode=retrieval_mode,
            sparse_topk=sparse_topk,
            deformable_radius=deformable_radius,
        )
        if retrieval_mask is not None:
            bias = bias.masked_fill(~retrieval_mask.unsqueeze(1), float("-inf"))

        return scale * bias

    @staticmethod
    def _retrieval_mask(
        temporal_prior: Tensor,
        mode: str,
        sparse_topk: int,
        deformable_radius: int,
    ) -> Tensor:
        if mode == "dense":
            return None

        batch_size, num_thoughts, src_len = temporal_prior.size()
        if mode == "sparse":
            if sparse_topk <= 0:
                raise ValueError("sparse retrieval requires thinking.sparse_topk > 0")
            topk = min(sparse_topk, src_len)
            indices = temporal_prior.topk(topk, dim=-1).indices
            mask = torch.zeros(
                batch_size,
                num_thoughts,
                src_len,
                device=temporal_prior.device,
                dtype=torch.bool,
            )
            return mask.scatter(-1, indices, True)

        if mode == "deformable":
            if deformable_radius < 0:
                raise ValueError(
                    "deformable retrieval requires thinking.deformable_radius >= 0"
                )
            positions = torch.arange(
                src_len, device=temporal_prior.device, dtype=temporal_prior.dtype
            )
            center = (temporal_prior * positions.view(1, 1, -1)).sum(dim=-1)
            distance = (positions.view(1, 1, -1) - center.unsqueeze(-1)).abs()
            mask = distance <= float(deformable_radius)
            nearest = distance.argmin(dim=-1, keepdim=True)
            return mask.scatter(-1, nearest, True)

        raise NotImplementedError(
            "Unsupported thinking.retrieval_mode '{}'. Expected one of: "
            "'dense', 'sparse', 'deformable'.".format(mode)
        )


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
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.K = K

        self.num_segments = num_segments
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.segmentation_temperature = segmentation_temperature
        if retrieval_mode not in {"dense", "sparse", "deformable"}:
            raise NotImplementedError(
                "Unsupported thinking.retrieval_mode '{}'. Expected one of: "
                "'dense', 'sparse', 'deformable'.".format(retrieval_mode)
            )
        self.retrieval_mode = retrieval_mode
        self.sparse_topk = sparse_topk
        self.deformable_radius = deformable_radius
        self.enable_decoder_routing = enable_decoder_routing
        self.routing_bias_scale = routing_bias_scale
        self.eps = eps

        self.thought_queries = nn.Parameter(torch.randn(K, hidden_size))
        self.segmenter = SoftTemporalSegmentation(
            hidden_size=hidden_size,
            num_segments=max(num_segments, 1),
            temperature=segmentation_temperature,
            eps=eps,
        )
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

    def forward(
        self, encoder_output: Tensor, src_mask: Tensor = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        :param encoder_output: [B, Ts, D]
        :param src_mask: [B, 1, Ts] or [B, Ts]
        :return: thought_states [B, K, D], routing cache
        """
        bsz = encoder_output.size(0)
        thoughts = self.thought_queries.unsqueeze(0).expand(bsz, -1, -1)

        mask = src_mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            mask = mask.bool()

        segment_tokens, segment_weights = self.segmenter(encoder_output, mask)
        routing_cache = {
            "segment_tokens": segment_tokens,
            "segment_weights": segment_weights,
            "enable_decoder_routing": self.enable_decoder_routing,
            "routing_bias_scale": self.routing_bias_scale,
            "eps": self.eps,
        }
        for layer in self.layers:
            thoughts, layer_cache = layer(
                thoughts=thoughts,
                memory=encoder_output,
                src_mask=mask,
                segment_tokens=segment_tokens,
                segment_weights=segment_weights,
                sinkhorn_iters=self.sinkhorn_iters,
                sinkhorn_tau=self.sinkhorn_tau,
                retrieval_mode=self.retrieval_mode,
                sparse_topk=self.sparse_topk,
                deformable_radius=self.deformable_radius,
                routing_bias_scale=self.routing_bias_scale,
                eps=self.eps,
            )
            if layer_cache is not None:
                routing_cache.update(layer_cache)

        return thoughts, routing_cache

    def structural_loss(
        self,
        routing_cache: Dict[str, Tensor],
        lambda_mono: float = 0.0,
        lambda_cont: float = 0.0,
        mono_margin: float = 1.0,
    ) -> Tensor:
        binding = routing_cache.get("binding") if routing_cache else None
        if binding is None:
            return None

        loss = binding.new_zeros(())
        if lambda_mono:
            loss = loss + lambda_mono * self.monotonicity_loss(
                binding=binding, margin=mono_margin
            )
        if lambda_cont:
            loss = loss + lambda_cont * self.contiguity_loss(binding=binding)
        return loss

    @staticmethod
    def monotonicity_loss(binding: Tensor, margin: float = 1.0) -> Tensor:
        num_segments = binding.size(-1)
        positions = torch.arange(
            1, num_segments + 1, device=binding.device, dtype=binding.dtype
        )
        expected_index = (binding * positions.view(1, 1, -1)).sum(dim=-1)
        violations = F.relu(
            expected_index[:, :-1] - expected_index[:, 1:] + margin
        )
        return violations.mean()

    @staticmethod
    def contiguity_loss(binding: Tensor) -> Tensor:
        if binding.size(-1) < 2:
            return binding.new_zeros(())
        return torch.abs(binding[:, :, 1:] - binding[:, :, :-1]).sum(dim=-1).mean()

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
