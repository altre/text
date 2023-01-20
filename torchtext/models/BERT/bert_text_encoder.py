# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torchtext.models.BERT.lbert_text_embedding import BERTTextEmbeddings
from torchtext.models.BERT.layers.attention import MultiHeadAttention, SelfAttention
from torchtext.models.BERT.layers.mlp import MLP


class TransformerOutput(NamedTuple):
    last_hidden_state: Optional[torch.Tensor] = None
    pooler_output: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    image_labels: Optional[torch.Tensor] = None


class BERTTextEncoder(nn.Module):
    """
    General text transformer encoder with embeddings, following BERT.
    Can be constructed with any user-provided embeddings and encoder.

    Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L870

    Attributes:
        embeddings (nn.Module): Module that projects text token ids into embeddings.
            See :py:class: `torchtext.modules.layers.text_embedding.BERTTextEmbeddings` for interface.
        encoder (nn.Module): Module for transformer encoder. See :py:class:
            `torchtext.modules.layers.transformer.TransformerEncoder` for interface.
        layernorm (nn.Module, optional): Module for layernorm to be applied after encoder. Defaults to ``None``.
        pooler (nn.Module, optional): Module for pooler to be applied after layernorm. Defaults to ``None``.
        weight_init_fn (Callable, optional): function for custom weight initialization of both the transformer
            encoder and embeddings. See :py:func: `torchtext.models.flava.transformer.init_transformer_weights`
            as an example. Defaults to ``None``.

    Args:
        input_ids (Tensor, optional): Tensor of input vocab token ids of shape [batch, seq_len].
        attention_mask (Tensor, optional): Tensor indicating which tokens to attend to, shape [batch, seq_len]
        token_type_ids (Tensor, optional): Tensor of input token type ids of shape [batch, seq_len]. In BERT,
            used to indicate whether a word is in sentence A or B for next sentence prediction
        position_ids (Tensor, optional): Tensor of input position ids of shape [batch, seq_len]
        inputs_embeds (Tensor, optional): Tensor of input embeddings of shape [batch, hidden_size],
            if embeddings are calculated elsewhere

    Raises:
        ValueError: if input_ids and inputs_embeds are both ``None``.
    """

    def __init__(
        self,
        embeddings: nn.Module,
        encoder: nn.Module,
        layernorm: Optional[nn.Module] = None,
        pooler: Optional[nn.Module] = None,
        weight_init_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.embeddings = embeddings
        self.encoder = encoder
        # TODO: could be upstreamed to TransformerEncoder?
        self.layernorm = layernorm
        self.pooler = pooler

        if weight_init_fn:
            self.apply(weight_init_fn)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("input_ids or inputs_embeds must not be None")

        # only mask out padding token if no mask specified
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            if hasattr(self.embeddings, "pad_token_id"):
                attention_mask[input_ids == self.embeddings.pad_token_id] = 0

        # massage attention mask to correct shape for transformer
        attention_mask = get_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_output = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            return_attn_weights=return_attn_weights,
            return_hidden_states=return_hidden_states,
        )

        last_hidden_state = encoder_output.last_hidden_state
        pooled_output = encoder_output.pooler_output
        if self.layernorm:
            last_hidden_state = self.layernorm(last_hidden_state)
        if self.pooler:
            pooled_output = self.pooler(last_hidden_state)

        return TransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


def get_extended_attention_mask(attention_mask: Tensor) -> Tensor:
    """Makes attention masks broadcastable along head and sequence dimensions.

    Accepting two types of attention masks:
        - Causal: masks that prevent attending to future positions of dimensions
            ``(batch_size, query_seq_len, key_seq_len)``
        - Padding: masks that prevent attending to token paddings of dimensions
            ``(batch_size, seq_len)``

    Args:
        attention_mask (Tensor):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.

    Returns:
        extended_attention_mask (Tensor):
            The broadcastable attention mask, with the same dtype as ``attention_mask.dtype``.
    """
    if attention_mask.dim() == 4:
        # Mask has already been broadcasted to the correct shape (either
        # [batch_size, num_heads, query_seq_length, key_seq_length] for causal case or
        # [batch_size, num_heads, seq_length, seq_length] for padding case)
        extended_attention_mask = attention_mask
    elif attention_mask.dim() == 3:
        # We can provide a self-attention mask of dimensions [batch_size, query_seq_length, key_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads,
        # [batch_size, num_heads, query_seq_length, key_seq_length].
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for attention_mask (shape {})".format(attention_mask.shape)
        )

    extended_attention_mask = extended_attention_mask.to(
        dtype=attention_mask.dtype
    )  # fp16 compatibility

    return extended_attention_mask


def bert_text_encoder(
    # transformer encoder params
    hidden_size: int = 768,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    dropout: float = 0.1,
    transform_act_fn: Callable[..., nn.Module] = nn.GELU,
    layer_norm_eps: float = 1e-12,
    norm_first: bool = False,
    # text embedding params
    vocab_size: int = 30522,
    max_position_embeddings: int = 512,
    type_vocab_size: int = 2,
    pad_token_id: int = 0,
    offset_pos_ids: bool = False,
    # layernorm and pooler
    layernorm: Optional[nn.Module] = None,
    pooler: Optional[nn.Module] = None,
    weight_init_fn: Optional[Callable] = None,
) -> BERTTextEncoder:
    """
    Returns a BERTTextEncoder with default params identical to HuggingFace's ``bert-base-uncased``.
    Ref: https://huggingface.co/bert-base-uncased/resolve/main/config.json. See :py:class:
    `torchtext.modules.layers.text_embedding.BERTTextEmbeddings` and :py:class:
    `torchtext.modules.layers.transformer.TransformerEncoder` for details on parameters.
    """
    embeddings = BERTTextEmbeddings(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        offset_pos_ids=offset_pos_ids,
    )
    encoder = TransformerEncoder(
        n_layer=num_hidden_layers,
        d_model=hidden_size,
        n_head=num_attention_heads,
        dim_feedforward=intermediate_size,
        dropout=dropout,
        activation=transform_act_fn,
        layer_norm_eps=layer_norm_eps,
        norm_first=norm_first,
    )
    return BERTTextEncoder(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=layernorm,
        pooler=pooler,
        weight_init_fn=weight_init_fn,
    )


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer is made up of multihead self-attention and feedforward blocks,
    based on the architecture in "Attention Is All You Need" (Vaswani et al. 2017). Similar to
    ``nn.TransformerEncoderLayer``, but uses a custom ``MultiHeadAttention`` that supports
    n-dimensional inputs (including sequences, images, video) and head-masking.

    Attributes:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention, cross-attention,
            and feedforward. Otherwise, layer norm is done after.

    Args:
        hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate self-attention on.
        attention_mask (Tensor, optional): mask to be applied to self-attention inputs, ``hidden_states``. See
            ``MultiHeadAttention`` for shape requirements.
        head_mask (Tensor, optional): mask to be applied to self-attention inputs after softmax and dropout,
            before matrix multiplication with values. See ``MultiHeadAttention`` for shape requirements.
        return_attn_weights (bool, optional): return attention probabilities in addition to attention output.
            Defaults to False.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        layer_norm_eps: float = 1e-12,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        # attention block
        self.attention = MultiHeadAttention(
            dim_q=d_model,
            dim_kv=d_model,
            n_head=n_head,
            attn_module=SelfAttention(dropout),
        )
        self.attention_dropout = nn.Dropout(dropout)
        # feedforward block
        self.feedforward = MLP(
            d_model, d_model, dim_feedforward, dropout=dropout, activation=activation
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        # layernorms
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _attention_block(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        output, attn_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            return_attn_weights=True,
        )
        output = self.attention_dropout(output)
        return output, attn_weights

    def _feedforward_block(self, hidden_states: Tensor) -> Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = hidden_states
        inputs = self.attention_layernorm(x)
        attn_output, attn_weights = self._attention_block(
            inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        attn_residual = attn_output + x
        ff_residual = attn_residual + self._feedforward_block(
            self.feedforward_layernorm(attn_residual)
        )
        if return_attn_weights:
            return ff_residual, attn_weights
        else:
            return ff_residual

    def _forward_postnorm(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = hidden_states
        attn_output, attn_weights = self._attention_block(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        attn_residual = attn_output + x
        attn_residual = self.attention_layernorm(attn_residual)
        ff_residual = attn_residual + self._feedforward_block(attn_residual)
        outputs = self.feedforward_layernorm(ff_residual)
        if return_attn_weights:
            return outputs, attn_weights
        else:
            return outputs

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.norm_first:
            return self._forward_prenorm(
                hidden_states,
                attention_mask,
                head_mask,
                return_attn_weights,
            )
        else:
            return self._forward_postnorm(
                hidden_states,
                attention_mask,
                head_mask,
                return_attn_weights,
            )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layer: int,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        layer_norm_eps: float = 1e-12,
        norm_first: bool = False,
        final_layer_norm_eps: Optional[float] = None,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    n_head,
                    dim_feedforward,
                    dropout,
                    activation,
                    layer_norm_eps,
                    norm_first,
                )
                for _ in range(n_layer)
            ]
        )
        self.final_layer_norm = None
        if final_layer_norm_eps:
            self.final_layer_norm = Fp32LayerNorm(d_model, eps=final_layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerOutput:
        all_hidden_states: Tuple[Tensor, ...] = () if return_hidden_states else None
        all_self_attentions: Tuple[Tensor, ...] = () if return_attn_weights else None

        for layer_module in self.layer:
            if return_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                return_attn_weights=return_attn_weights,
            )

            if return_attn_weights:
                hidden_states = layer_outputs[0]
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            else:
                hidden_states = layer_outputs

        if return_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        return TransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)
