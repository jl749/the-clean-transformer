"""
any constant tensors are defined here.
they will be registered as buffers.
"""
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
