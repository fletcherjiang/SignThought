# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch import Tensor
from .encoder_module.ResidualModule import ResidualConnectionModule
from .encoder_module.MHSA_RPE import MultiHeadedSelfAttentionModule, MultiHeadedCrossAttentionModule, \
    ContextualMultiHeadedSelfAttentionModule, ContextualMultiHeadedCrossAttentionModule, RelPosMultiHeadSelfAttention, RelativeMultiheadSelfAttentionModule
from .encoder_module.Convolution import ConvModule, PointwiseConv1d, ConvModuleOriginal
from .encoder_module.FeedForward import FeedForwardModule
from .utils.attention_module import DeformableMultiHeadedAttention, ContextualMultiHeadAttention

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor = None,
        attn_bias: Tensor = None,
        return_attention: bool = False,
    ):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M] or [B, N, M]
        :param attn_bias: optional additive attention-logit bias. Accepted shapes
            are [B, N, M], [B, 1, N, M], or [B, H, N, M].
        :param return_attention: if True, also return pre-dropout attention probs.
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)
            elif attn_bias.dim() != 4:
                raise ValueError("attn_bias must have 3 or 4 dimensions")
            scores = scores + attn_bias.to(dtype=scores.dtype)

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention_probs = attention
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        if return_attention:
            return output, attention_probs
        return output


# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


# pylint: disable=arguments-differ

class CA_TransformerEncoderLayer(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.
    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(CA_TransformerEncoderLayer, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.att = ResidualConnectionModule(
            module=MultiHeadedSelfAttentionModule(
                d_model=encoder_dim,
                num_heads=num_attention_heads,
                dropout_p=attention_dropout_p,
            ),
        )
        self.convMod = ResidualConnectionModule(
            module=ConvModule(
                in_channels=encoder_dim,
                kernel_size=conv_kernel_size,
                expansion_factor=conv_expansion_factor,
                dropout_p=conv_dropout_p,
            )
        )

        self.FF2 = ResidualConnectionModule(
            module=PositionwiseFeedForward(input_size=encoder_dim, ff_size=feed_forward_expansion_factor * encoder_dim,
                                           dropout=feed_forward_dropout_p),
            module_factor=0.5,
        )
        self.layerNORM = nn.LayerNorm(encoder_dim)

    def forward(self, inputs: Tensor, mask: Tensor = None) -> Tensor:
        x_att = self.att(inputs, mask=mask)
        x = self.convMod(x_att)
        x = self.FF2(x)
        x = self.layerNORM(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(
            size=size,
            num_heads=num_heads,
            dropout=dropout
        )

        self.thought_trg_att = MultiHeadedAttention(
            size=size,
            num_heads=num_heads,
            dropout=dropout
        )

        self.src_trg_att = MultiHeadedAttention(
            size=size,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feed_forward = PositionwiseFeedForward(input_size=size,ff_size=ff_size,dropout=dropout)
        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.thought_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_thought = nn.Dropout(dropout)

        self.output_norm = nn.LayerNorm(size)

    def forward(
        self,
        x: Tensor = None,
        memory: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
        thought_memory: Tensor = None,
        thought_mask: Tensor = None,
        thought_routing: dict = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # thought-chain attention (if provided)
        thought_attention = None
        if thought_memory is not None:
            h_thought_norm = self.thought_layer_norm(h1)
            h_thought, thought_attention = self.thought_trg_att(
                h_thought_norm,
                thought_memory,
                thought_memory,
                mask=thought_mask,
                return_attention=True,
            )
            h_thought = self.dropout_thought(h_thought) + h1
        else:
            h_thought = h1

        routing_bias = self._decoder_routing_bias(
            thought_attention=thought_attention,
            thought_routing=thought_routing,
            memory=memory,
        )

        # source-target attention
        h1_norm = self.dec_layer_norm(h_thought)
        h2 = self.src_trg_att(
            h1_norm, memory, memory, mask=src_mask, attn_bias=routing_bias
        )
        h2 = self.dropout2(h2) + h_thought

        # final feed-forward layer
        h2_norm = self.output_norm(h2)
        h3 = self.feed_forward(h2_norm)
        h3 = self.dropout3(h3) + h2

        return h3

    @staticmethod
    def _decoder_routing_bias(
        thought_attention: Tensor = None,
        thought_routing: dict = None,
        memory: Tensor = None,
    ) -> Tensor:
        if thought_attention is None or not thought_routing:
            return None
        if not thought_routing.get("enable_decoder_routing", False):
            return None

        binding = thought_routing.get("binding")
        segment_weights = thought_routing.get("segment_weights")
        if binding is None or segment_weights is None:
            return None

        scale = thought_routing.get("routing_bias_scale", 1.0)
        if scale == 0:
            return None

        # Head-averaged token-to-thought attention α, then w = α A Wseg.
        thought_alpha = thought_attention.mean(dim=1)
        token_segment = torch.bmm(thought_alpha, binding)
        token_frame = torch.bmm(token_segment, segment_weights)

        if memory is not None and token_frame.size(-1) != memory.size(1):
            token_frame = token_frame[..., : memory.size(1)]

        eps = thought_routing.get("eps", 1.0e-6)
        return scale * torch.log(token_frame.clamp_min(eps))
