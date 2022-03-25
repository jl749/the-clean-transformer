import torch
from cleanformer.models import MultiHeadAttentionLayer


def main():
    hidden_size = 512
    encoding_size = 256
    heads = 8

    N = 10
    L = 30

    # layer = MultiHeadAttentionLayer(hidden_size, encoding_size, heads, max_length=L, masked=False)
    layer = MultiHeadAttentionLayer(hidden_size, encoding_size, heads, max_length=L, masked=True)

    q = torch.rand(size=[N, L, hidden_size])
    k = torch.rand(size=[N, L, hidden_size])
    v = torch.rand(size=[N, L, hidden_size])

    key_padding_mask = torch.ones(size=[N, L]).long()

    out = layer.forward(q, k, v, key_padding_mask)
    print(out)


if __name__ == '__main__':
    main()
