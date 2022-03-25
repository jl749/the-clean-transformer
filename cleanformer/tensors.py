"""
any constant tensors are defined here.
they will be registered as buffers.
"""
import numpy as np
import torch


def subsequent_mask(max_length: int) -> torch.LongTensor:
    """
    :param: max_length (L)
    Subsequently allow positions
    1 0 0
    1 1 0
    1 1 1
    :return: (L, L)
    """
    ones = torch.ones(size=(max_length, max_length))  # (L, L)
    mask = torch.tril(ones, diagonal=1).long()  # (L, L) -> (L, L)
    return mask


def pos_encodings(max_length: int, hidden_size: int) -> torch.Tensor:
    """
    max_length: L
    hidden_size: H
    :return: (L, H)
    """
    positions = torch.arange(max_length).view(-1, 1)  # scalar -> (L, 1)
    freqs = 0.0001**(torch.arange(hidden_size)[::2] / hidden_size).view(1, -1)  # (1, H/2)
    encodings = torch.zeros(size=(max_length, hidden_size))  # (L, H)

    # fill in the pairs by broadcast-multiplying freqs to positions; sin=odd_index cos=even_index
    encodings[:, ::2] = torch.sin(freqs * positions)  # (1, 50) * (10, 1) --> (10, 50)
    encodings[:, 1::2] = torch.cos(freqs * positions)
    # why the same frequency?
    # A: so that dist(PE(pos + k) - PE(pos)) stays constant

    return encodings


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 key_mask: torch.LongTensor) -> torch.Tensor:
    """
    Definition: Attention(Q, K, V) = softmax(Q dot K^T / sqrt(d_k)) dot V
    What it does: soft-align values with respect to the similarities of their keys to each query
    param q: (..., L, H)
    param k: (..., L, H)
    param v: (..., L, H)
    key_mask: (..., L, L)
    """
    # Q * K^T:  compute query-key similarities
    sims = q @ k.transpose(-1, -2)  # torch.einsum("...qh,...kh->...qk", q, k)

    # Q * K^T / sqrt(d_k): down-scale similarities to prevent gradient vanishing
    sims /= np.sqrt(k.shape[-1])

    # apply padding mask and/or subsequent mask
    sims = sims.masked_fill(key_mask == 0, float("-inf"))

    # softmax(Q * K^T / sqrt(d_k)): normalise the sims over keys
    attentions = torch.softmax(sims, dim=-1)  # (..., L, L)

    # softmax(Q * K^T / sqrt(d_k)) * V: soft-align values with respect to each query
    alignments = attentions @ v  # torch.einsum("...qv,...vh->...qh", attentions, v)
    return alignments
