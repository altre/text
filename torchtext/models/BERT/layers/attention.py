# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F


def shift_dim(
    x: Tensor, src_dim: int = -1, dest_dim: int = -1, make_contiguous: bool = True
) -> Tensor:
    """Permutes tensor x by moving src_dim to dest_dim.
    i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)

    Code taken from VideoGPT
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/utils.py

    Args:
        x (Tensor): input Tensor you want to permute
        src_dim (int, optional): the axis you want to move. Negative indexing supported. Defaults to -1.
        dest_dim (int, optional): the axis you want to move to. Negative indexing supported. Defaults to -1.
        make_contiguous (bool, optional): if you want the output tensor to be contiguous in memory. Defaults to True.

    Returns:
        Tensor: permuted Tensor
    """
    n_dims = len(x.shape)
    # Remap negative dim
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


class SelfAttention(nn.Module):
    """Computes attention over the entire n-dimensional input.

    Args:
        attn_dropout (float, optional): Probability of dropout after softmax. Default is ``0.0``.
    """

    def __init__(self, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_dropout = attn_dropout

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q (Tensor): Query input of shape ``(b, h, d1, ..., dn, dim_q)`` where ``h`` is the number of
                attention heads, ``(d1, ..., dn)`` are the query latent dimensions and ``dim_q`` is the dimension
                of the query embeddings.
            k, v (Tensor): Key/value input of shape ``(b, h, d1', ..., dn', dim_kv)`` where ``h`` is the number
                of attention heads, ``(d1', ..., dn')`` are the key/value latent dimensions and ``dim_kv`` is
                the dimension of the key/value embeddings.
            attention_mask (Tensor, optional): Tensor of shape ``(b, h, q_dn, k_dn)`` where ``q_dn`` is the
                dimension of the flattened query input along its latent dimensions and ``k_dn`` that of the
                flattened key input. Contains 1s for positions to attend to and 0s for masked positions.
            head_mask (Tensor, optional): Tensor of shape ``(b, h, q_dn, k_dn)``.
                Contains 1s for positions to attend to and 0s for masked positions.

        Returns:
            A tuple of output tensor and attention probabilities.
        """
        _, _, *shape, _ = q.shape

        # flatten to b, h, (d1, ..., dn), dim_q/dim_kv
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out, attn_probs = scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            head_mask=head_mask,
            attn_dropout=self.attn_dropout if self.training else 0.0,
        )

        return out.unflatten(2, shape), attn_probs


class MultiHeadAttention(nn.Module):
    """Computes multihead attention with flexible attention mechanism and caching for fast decoding.

    Multihead attention linearly projects and divides queries, keys, and values into
    multiple 'heads'. This enables the computation of attention multiple times in
    parallel, creating more varied representations and allows the model to jointly
    attend to information from different representation subspaces at different positions,
    as described in `"Attention Is All You Need (Vaswani et al. 2017)"<https://arxiv.org/pdf/1706.03762.pdf>`_.

    Args:
        dim_q (int): Dimensionality of query embedding vector.
        dim_kv (int): Dimensionality of key/value embedding vector.
        n_head (int): Number of attention heads.
        attn_module (nn.Module): Module of attention mechanism to use. Default is ``SelfAttention``.
            See :class:`~torchmultimodal.modules.layers.attention.SelfAttention` for API details.
        add_bias (bool): Whether to add bias to the q, k, v, linear layers or not. Default is ``True``.

    Attributes:
        cache (Dict[str, Tensor]): Dictionary that stores past key/value vectors.

    Raises:
        ValueError: When ``dim_q`` or ``dim_kv`` is not divisible by ``n_head``.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        n_head: int,
        attn_module: nn.Module = SelfAttention(),
        add_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim_q % n_head != 0 or dim_kv % n_head != 0:
            raise ValueError(
                "The hidden size of q, k, v must be a multiple of the number of attention heads."
            )

        self.d_qk = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head
        self.query = nn.Linear(dim_q, n_head * self.d_qk, bias=add_bias)  # q
        self.key = nn.Linear(dim_kv, n_head * self.d_qk, bias=add_bias)  # k
        self.value = nn.Linear(dim_kv, n_head * self.d_v, bias=add_bias)  # v
        self.output = nn.Linear(n_head * self.d_v, dim_q, bias=True)  # c

        self.attn = attn_module

        self.cache: Optional[Dict[str, Tensor]] = None

    def forward(
        self,
        q: Tensor,
        kv: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
        use_cache: bool = False,
        causal: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            q (Tensor): Query of shape ``(b, d1, ..., dn, dim_q)`` or ``(b, seq_len, dim_q)``
                (for autoregressive decoding it's typical to pass in flattened tensors).
            kv (Tensor, optional): Key (and value) of shape ``(b, d1', ..., dn', dim_kv)`` or
                ``(b, seq_len', dim_kv)``. If this argument is specified, cross-attention will be applied.
                Default is ``None``.
            attention_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)`` where ``q_dn`` is
                the dimension of the axis to compute attention on of the query and ``k_dn`` that of the key.
                If the input tensors are flattened across the entire latent dimensions, ``q_dn = d1 x ... x dn``
                and ``k_dn = d1' x ... x dn'``. Contains 1s for positions to attend to and 0s
                for masked positions.
            head_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
                Contains 1s for positions to attend to and 0s for masked positions.
            use_cache (bool): If ``True``, caches past ``k`` and ``v`` tensors for faster decoding.
                If ``False``, recomputes ``k`` and ``v`` for each decoding step. Default is ``False``.
            causal (bool): Whether to use causal attention or not. Default is ``False``.

        Returns:
            * If ``return_attn_weights`` is ``True``: A tuple of output tensor and attention probabilities.
            * If ``return_attn_weights`` is ``False``: A single output tensor.

        Raises:
            TypeError: An error occurred when ``causal`` is ``True`` and ``attn_module`` is ``AxialAttention``.
        """
        # If kv is specified use those inputs for cross-attention, otherwise use q
        k = v = q if kv is None else kv
        # compute q
        q = split_multihead(self.query(q), self.n_head)

        # For causal k, v are provided step-wise so we should always compute them
        # For non-causal skip computing k, v if they have been cached
        if causal or not self.cache:
            k = split_multihead(self.key(k), self.n_head)
            v = split_multihead(self.value(v), self.n_head)

        # fast decoding by caching past key, value tensors
        if use_cache:
            if not self.cache:
                # initialize the cache with the present k, v
                self.cache = dict(k=k.clone(), v=v.clone())
            else:
                if causal:
                    # append present k, v to past k, v
                    # for autoregressive decoding inputs are flattened as 1D sequences
                    # so are the cached tensors: (b, n_heads, seq_len, c)
                    k_, v_ = self.cache["k"], self.cache["v"]
                    self.cache["k"] = torch.cat([k_, k], dim=2)
                    self.cache["v"] = torch.cat([v_, v], dim=2)
                # override the present k, v with the cache
                k, v = self.cache["k"], self.cache["v"]

        a, attn_probs = self.attn(q, k, v, attention_mask, head_mask)
        a = merge_multihead(a)
        a = self.output(a)

        if return_attn_weights:
            return a, attn_probs
        else:
            return a


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Optional[Tensor] = None,
    head_mask: Optional[Tensor] = None,
    attn_dropout: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Similar to PyTorch Core's _scaled_dot_product_attention but generalized
    to handle n-dimensional input tokens (images, video) and support multihead.
    Computes attention as described in Attention Is All You Need (Vaswani et al. 2017)

    Args:
        q (Tensor): Query of shape ``(b, h, d1, ..., dn, dim_qk)`` or ``(b, h, seq_len, dim_qk)`` where
            ``h`` is number of attention heads, ``d1, ..., dn`` are latent dimensions and ``dim_qk` is
            the embedding dim of the query tensor.
        k (Tensor): Key of shape ``(b, h, d1', ...., dn', dim_qk)`` or ``(b, h, seq_len', dim_qk)`` where
            ``h`` is the number of attention heads, ``d1', ..., dn'` are latent dimensions and ``dim_qk``
            is the key embedding dim aligned with query embedding dim,
            see :class:`~torchmultimodal.modules.layers.attention.MultiHeadAttention`.
        v (Tensor): Value of shape ``(b, h, d1', ..., dn', dim_v)`` or ``(b, h, seq_len', dim_v)`` where
            ``h`` is the number of attention heads, ``d1', ..., dn'`` are latent dimensions and ``dim_v``
            is the embedding dim of the value tensor.
        attention_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
            Contains 1s for positions to attend to and 0s for masked positions. Applied before softmax.
        head_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
            Contains 1s for positions to attend to and 0s for masked positions.
            Applied after dropout, before matrix multiplication with values.
        attn_dropout (float): Probability of dropout after softmax. Default is ``0.0``.

    Returns:
        A tuple of output tensor and attention probabilities.
    """

    # Take the dot product between "query" and "key" and scale to get the raw attention scores.
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / torch.sqrt(torch.tensor(q.shape[-1]))
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor with the computed attention weights
    # at the positions we want to attend and -inf for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    if attention_mask is not None:
        attn = attn.masked_fill(attention_mask == 0, float("-inf"))
    # Normalize the attention scores to probabilities
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)  # b, h, d1, ..., q_dn, k_dn
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn = F.dropout(attn, p=attn_dropout)
    # Mask heads if we want to
    if head_mask is not None:
        attn = attn * head_mask
    # For each query sum over the key/value dim with attention weights
    a = torch.matmul(attn, v)  # b, h, d1, ..., q_dn, c

    return a, attn


def split_multihead(x: Tensor, n_head: int) -> Tensor:
    """Splits channel dimension of input tensor of size (b, d1, ..., dn, c)
    into multiple heads, (b, n_head, d1, ..., dn, c // n_head)"""
    x = x.unflatten(-1, (n_head, -1))
    # Rearrange to put head dim first, (b, n_head, d1, ..., dn, c // n_head)
    x = shift_dim(x, -2, 1)
    return x


def merge_multihead(x: Tensor) -> Tensor:
    """Moves head dim back to original location and concatenates heads
    (b, n_head, d1, ..., dn, c // n_head) -> (b, d1, ..., dn, c)"""
    return shift_dim(x, 1, -2).flatten(start_dim=-2)
