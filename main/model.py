# coding: utf-8
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from itertools import groupby
from main.initialization import initialize_model
from main.embeddings import Embeddings, SpatialEmbeddings
from main.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from main.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from main.search import beam_search, greedy
from main.vocabulary import (
    TextVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from main.batch import Batch
from main.helpers import freeze_params
from main.thinking import LatentCoT
from torch import Tensor
from typing import Union


class SignModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        txt_vocab: TextVocabulary,
        thinking: LatentCoT = None,
    ):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param sgn_embed: spatial feature frame embeddings
        :param txt_embed: spoken language word embedding
        :param txt_vocab: spoken language vocabulary
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed
        self.thinking = thinking

        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

    # pylint: disable=arguments-differ
    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths
        )

        unroll_steps = txt_input.size(1)
        decoder_outputs = self.decode(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            sgn_mask=sgn_mask,
            txt_input=txt_input,
            unroll_steps=unroll_steps,
            txt_mask=txt_mask,
            thought_states=self._compute_thoughts(
                encoder_output=encoder_output, sgn_mask=sgn_mask
            ),
        )

        return decoder_outputs

    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(
            embed_src=self.sgn_embed(x=sgn, mask=sgn_mask),
            src_length=sgn_length,
            mask=sgn_mask,
        )

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
        thought_states: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
            thought_states=thought_states,
        )

    def _compute_thoughts(self, encoder_output: Tensor, sgn_mask: Tensor) -> Tensor:
        if self.thinking is None:
            return None
        return self.thinking(encoder_output=encoder_output, src_mask=sgn_mask)

    def get_loss_for_batch(
        self,
        batch: Batch,
        translation_loss_function: nn.Module,
        translation_loss_weight: float,
    ) -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
        :param translation_loss_weight: Weight for translation loss
        :return: translation_loss: sum of losses over non-pad elements in the batch
        """
        # Do a forward pass
        decoder_outputs = self.forward(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
        )

        word_outputs, _, _, _ = decoder_outputs
        # Calculate Translation Loss
        txt_log_probs = F.log_softmax(word_outputs, dim=-1)
        translation_loss = (
            translation_loss_function(txt_log_probs, batch.txt)
            * translation_loss_weight
        )

        return translation_loss

    def run_batch(
        self,
        batch: Batch,
        translation_beam_size: int = 4,
        translation_beam_alpha: float = 7,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        encoder_output, encoder_hidden = self.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
        )
        thought_states = self._compute_thoughts(
            encoder_output=encoder_output, sgn_mask=batch.sgn_mask
        )

        # greedy decoding
        if translation_beam_size < 2:
            stacked_txt_output, stacked_attention_scores = greedy(
                encoder_hidden=encoder_hidden,
                encoder_output=encoder_output,
                src_mask=batch.sgn_mask,
                embed=self.txt_embed,
                bos_index=self.txt_bos_index,
                eos_index=self.txt_eos_index,
                decoder=self.decoder,
                max_output_length=translation_max_output_length,
                thought_states=thought_states,
            )
        else:  # beam size
            stacked_txt_output, stacked_attention_scores = beam_search(
                size=translation_beam_size,
                encoder_hidden=encoder_hidden,
                encoder_output=encoder_output,
                src_mask=batch.sgn_mask,
                embed=self.txt_embed,
                max_output_length=translation_max_output_length,
                alpha=translation_beam_alpha,
                eos_index=self.txt_eos_index,
                pad_index=self.txt_pad_index,
                bos_index=self.txt_bos_index,
                decoder=self.decoder,
                thought_states=thought_states,
            )

        return stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s,\n"
            "\tthinking=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
                self.thinking,
            )
        )


def build_model(
    cfg: dict,
    sgn_dim: int,
    txt_vocab: TextVocabulary,
    multimodal: bool = False,
    thinking_cfg: dict = None,
) -> SignModel:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param sgn_dim: feature dimension of the sign input
    :param txt_vocab: target vocabulary
    :param multimodal: whether to use multimodal features
    :return: built and initialized model
    """
    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )
    else:
        encoder = RecurrentEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )

    # build decoder
    txt_embed: Union[Embeddings, None] = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
        )
    dec_dropout = cfg["decoder"].get("dropout", 0.0)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        decoder = TransformerDecoder(
            **cfg["decoder"],
            encoder=encoder,
            vocab_size=len(txt_vocab),
            emb_size=txt_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )
    else:
        decoder = RecurrentDecoder(
            **cfg["decoder"],
            encoder=encoder,
            vocab_size=len(txt_vocab),
            emb_size=txt_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )

    _model_thinking_cfg = cfg.get("thinking", {})
    _top_thinking_cfg = thinking_cfg if thinking_cfg is not None else {}
    _tcfg = {**_model_thinking_cfg, **_top_thinking_cfg}
    thinking = LatentCoT(
        hidden_size=_tcfg.get("hidden_size", cfg["encoder"]["hidden_size"]),
        num_heads=_tcfg.get("num_heads", cfg["encoder"]["num_heads"]),
        ff_size=_tcfg.get("ff_size", cfg["encoder"].get("ff_size", 4 * cfg["encoder"]["hidden_size"])),
        num_layers=_tcfg.get("num_layers", 2),
        dropout=_tcfg.get("dropout", cfg["encoder"].get("dropout", 0.1)),
        K=_tcfg.get("K", 6),
        causal_self_attn=_tcfg.get("causal_self_attn", True),
        num_segments=_tcfg.get("num_segments", 1),
        sinkhorn_iters=_tcfg.get("sinkhorn_iters", 3),
        sinkhorn_tau=_tcfg.get("sinkhorn_tau", 1.0),
        segmentation_temperature=_tcfg.get("segmentation_temperature", 0.1),
        retrieval_mode=_tcfg.get("retrieval_mode", "dense"),
        sparse_topk=_tcfg.get("sparse_topk", 0),
        deformable_radius=_tcfg.get("deformable_radius", 1),
        enable_decoder_routing=_tcfg.get("enable_decoder_routing", False),
        routing_bias_scale=_tcfg.get("routing_bias_scale", 1.0),
    )

    model = SignModel(
        encoder=encoder,
        decoder=decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        txt_vocab=txt_vocab,
        thinking=thinking,
    )

    # tie softmax layer with txt embeddings
    if cfg.get("tied_softmax", False):
        if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
            model.decoder.output_layer.weight = txt_embed.lut.weight
        else:
            raise ValueError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer."
                )

    # custom initialization of model parameters
    initialize_model(model, cfg, txt_padding_idx)
    return model
