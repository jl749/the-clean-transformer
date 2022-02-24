import torch
from cleanformer.models import MultiHeadAttentionLayer


def main():
    hidden_size = 512
    heads = 8

    N = 10
    L = 30

    layer = MultiHeadAttentionLayer(hidden_size, heads, max_length=L, masked=False)

    q = torch.rand(size=[N, L, hidden_size])
    k = torch.rand(size=[N, L, hidden_size])
    v = torch.rand(size=[N, L, hidden_size])

    out = layer.forward(q, k, v)
    print(out)


if __name__ == '__main__':
    main()
