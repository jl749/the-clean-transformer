import math
from typing import Dict, Tuple
from cleanformer.tensors import subsequent_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


# TODO: implement transformer
class Transformer(LightningModule):
    def __init__(self, hidden_size: int, encoding_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):  # noqa
        super().__init__()

        # Enable Lightning to store all the provided arguments under the self.hparams attribute
        # These hyperparameters will also be stored within the model checkpoint
        self.save_hyperparameters()

        # ==================== trainable layers ==================== #
        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                             embedding_dim=hidden_size)  # (vocab_size, hidden_size) table
        self.encoder = Encoder(hidden_size, encoding_size, heads, max_length)
        self.decoder = Decoder(hidden_size, encoding_size, heads, max_length)
        # ==================== trainable layers ==================== #

    # Use for inference only
    def forward(self, src_ids: torch.LongTensor, tgt_ids: torch.Tensor,
                src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        src_ids, tgt_ids --> output hidden vector (N, L, H)

        return hidden vector output from decoder
        """

        src = self.token_embeddings(src_ids)  # (N, L) --> (N, L, H)
        tgt = self.token_embeddings(tgt_ids)  # (N, L) --> (N, L, H)

        # POSITIONAL ENCODING
        # TODO: later

        context_info = self.encoder(src)  # (N, L, H) --> (N, L, H), shape does not change; add contextual info
        hidden = self.decoder(tgt, context_info)  # (N, L, H) --> (N, L, H), decoder takes 2 inputs
        return hidden

    # the complete training loop
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> Dict:
        """
        batch --> output of your DataLoader
        """
        X, Y = batch  # (N, 2, 2, L) (N, L)
        # [ N = batch size, 2(src(KO) | target(ENG)), 2(ids | mask(padding)), L(sequence max length) ]
        # Decoder input starts with [BOS] token

        # encoder inputs  (N, L)
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # decoder inputs  (N, L)
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]

        hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)
        classifier = self.token_embeddings.weight  # (V, H)  V = BoW classes
        logits = torch.einsum("nlh,vh->nvl", hidden, classifier)  # logits --> NCL (channels first)
        loss = F.cross_entropy(logits, Y)  # CEE requires (N, classes, L), (N, L)
        loss = loss.sum()  # (N,) -> scalar

        return {
            "loss": loss,
        }

    # custom method for "INFERENCE" only
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        param X (N, 2, 2, L)
        return label_ids (N, L)
        """

        # encoder inputs  (N, L)
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # decoder inputs  (N, L)  --  when doing the inference ['[BOS]', '[PAD]', '[PAD]' ...]
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]

        for time in range(0, self.hparams['max_length'] - 1):
            # --- (N, L, H)
            hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)  # (N, L, H)
            classifier = self.token_embeddings.weight  # (V, H)  V = BoW classes

            # (N, L, V) --> foreach tokenized word in L show prob distribution
            logits = torch.einsum("nlh,vh->nlv", hidden,
                                  classifier)  # TODO: THIS IS greedy decoding, look up beem search algo!!!
            ids = torch.argmax(logits, dim=2)  # (N, L, V)  -->  (N, L); highest prob indexes

            # pass current output to next input
            next_id = ids[:, time]  # (N, L) -->  (N,)

            tgt_ids[:, time + 1] = next_id
            tgt_key_padding_mask[:, time + 1] = 0  # not padding anymore

        label_ids = tgt_ids  # final sequence ids (N, L)
        return label_ids


class Encoder(nn.Module):

    def __init__(self, hidden_size: int, encoding_size: int, heads: int, max_length: int) -> None:
        """
        hidden_size: original feature level
        encoding_size: encoded feature level (reduced)
        heads: num of heads
        max_length: max token length
        """
        super().__init__()
        self.multiHead_selfAttention_layer = MultiHeadAttentionLayer(hidden_size, encoding_size,
                                                                     heads, max_length, masked=False)
        # TODO - ffn

    # override
    def forward(self, x: torch.Tensor):
        """
        x: (N, L, H)
        return vector with contectual meanings encoded
        """
        contexts = self.multiHead_selfAttention_layer.forward(q=x, k=x, v=x)
        return contexts


class Decoder(nn.Module):

    def __init__(self, hidden_size: int, encoding_size: int, heads: int, max_length: int) -> None:
        """
        hidden_size: original feature level
        encoding_size: encoded feature level (reduced)
        heads: num of heads
        max_length: max token length
        """
        super().__init__()
        self.masked_multiHead_selfAttention_layer = MultiHeadAttentionLayer(hidden_size, encoding_size,
                                                                            heads, max_length, masked=True)

    # override
    def forward(self, x: torch.Tensor, context_info: torch.Tensor):
        """
        x: (N, L, H)
        memory: encoder output
        return vector with contectual meanings encoded
        """
        contexts = self.multiHead_selfAttention_layer.forward(q=x, k=x, v=x)
        return contexts

        # TODO: ffn(feed-forward), residual connection


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hidden_size: int, encoding_size: int, heads: int, max_length: int, masked: bool) -> None:
        """
        hidden_size = H
        encoding_size = E
        heads = number of self attention heads
        max_length = max length of L
        masked = masked MultiHeadAttention?
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.heads = heads  # how many heads?
        self.max_length = max_length  # L
        self.masked = masked

        self.linear_o = nn.Linear(encoding_size * heads, hidden_size)

    # override
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: (N, L, H)
        k: (N, L, H)
        v: (N, L, H)
        return contexts (N, L, H)
        """
        N, _, _ = q.size()

        tmp = AttentionLayer(self.hidden_size, self.encoding_size, self.max_length, self.masked)  # (N, L * heads, E)
        result = tmp(q, k, v)
        for _ in range(self.heads - 1):
            tmp = AttentionLayer(self.hidden_size, self.encoding_size, self.max_length, self.masked)
            head = tmp(q, k, v)  # (N, L, E)
            result = torch.cat((result, head), dim=2)

        context = self.linear_o(result)  # (N, L, H)

        return context


class AttentionLayer(nn.Module):

    def __init__(self, hidden_size: int, encoding_size: int, max_length: int, masked: bool) -> None:
        super().__init__()
        self.linear_q = nn.Linear(hidden_size, encoding_size)
        self.linear_k = nn.Linear(hidden_size, encoding_size)
        self.linear_v = nn.Linear(hidden_size, encoding_size)
        self.linear_o = nn.Linear(encoding_size, hidden_size)
        self.masked = masked

        # const tensor in register_buffer
        # 나중에 model.device("cuda") 모델과 함께 상수텐서도 같이 GPU load
        self.register_buffer("subsequent_mask", subsequent_mask(max_length))

    def forward(self, q, k, v) -> torch.Tensor:
        N, L, _ = q.size()

        q = self.linear_q(q)  # (N, L, E)
        k = self.linear_k(k)  # (N, L, E)
        v = self.linear_v(v)  # (N, L, E)

        sim = q @ k.permute(0, 2, 1)  # (N, L, L)

        if self.masked:
            mask = subsequent_mask(L).repeat(N, 1, 1)  # (L, L) --> (N, L, L)
            sim = sim.masked_fill(mask == 0, value=float("-inf"))

        attention = F.softmax(sim / math.sqrt(L), dim=-1)  # (N, L, L)
        context = attention @ v  # (N, L, L) @ (N, L, E) --> (N, L, E)

        return context
