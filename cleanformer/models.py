import math
from typing import Dict, Tuple, List

from tqdm import tqdm

from cleanformer.tensors import subsequent_mask, pos_encodings, scaled_dot_product_attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy


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
        self.encoder = Encoder(hidden_size, encoding_size, heads, max_length, depth, ffn_size, dropout)
        self.decoder = Decoder(hidden_size, encoding_size, heads, max_length, depth, ffn_size, dropout)
        # ==================== trainable layers ==================== #

        # TODO: accuracy
        self.acc_train = Accuracy(ignore_index=pad_token_id)
        self.acc_val = Accuracy(ignore_index=pad_token_id)
        self.acc_test = Accuracy(ignore_index=pad_token_id)

        self.register_buffer("pos_encodings", pos_encodings(max_length, hidden_size))  # (L, H)

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
        src += self.pos_encodings  # (N, L, H) + (L, H) --> (N, L, H)
        tgt += self.pos_encodings  # (N, L, H) + (L, H) --> (N, L, H)

        # (N, L, H) --> (N, L, H), shape does not change; add contextual info
        context_info = self.encoder(src, src_key_padding_mask)
        hidden = self.decoder(tgt, context_info, tgt_key_padding_mask, src_key_padding_mask)  # decoder takes 2 inputs

        return hidden

    def on_train_start(self):
        # many deep transformer models are initialised with so-called "Xavier initialisation"
        # refer to: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        for param in tqdm(self.parameters(), desc="initialising weights..."):
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def on_train_batch_end(self, outputs: dict, *args, **kwargs):
        self.log("Train/Loss", outputs['loss'])

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams['lr'],
                                     betas=(0.9, 0.98), eps=1e-9)
        return {
            'optimizer': optimizer
        }

    # custom method that returns loss, logit_output
    def _step(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X: (N, 2, 2, L)
        Y: (N, L)
        """
        # encoder inputs  (N, L)
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # decoder inputs  (N, L)
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]

        hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)
        classifier = self.token_embeddings.weight  # (V, H)  V = BoW classes
        logits = torch.einsum("nlh,vh->nvl", hidden, classifier)  # logits --> NCL (channels first)
        loss = F.cross_entropy(logits, Y)  # CEE requires (N, classes, L), (N, L)
        loss = loss.sum()  # (N,) -> scalar

        return loss, logits

    # the complete training loop
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> Dict:
        """
        batch: output of your DataLoader
        return: a scalar tensor containing the loss for this batch
        """
        X, Y = batch  # (N, 2, 2, L) (N, L)
        # [ N = batch size, 2(src(KO) | target(ENG)), 2(ids | mask(padding)), L(sequence max length) ]
        # Decoder input starts with [BOS] token

        loss, logits = self._step(X, Y)

        return {
            "loss": loss,
        }

    # custom method used for "INFERENCE" only
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


class FeedForward(torch.nn.Module):
    """
    position-wise feedforward network.
    """

    def __init__(self, hidden_size: int, ffn_size: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.ReLU(),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: (N, L, H)
        :return: x (hidden): (N, L, H)
        """
        return self.layers(x)


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, encoding_size: int, heads: int, max_length: int,
                 ffn_size: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttentionLayer(hidden_size, encoding_size, max_length, heads)
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)

    def forward(self, x: torch.Tensor, x_key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param x (N, L, H)
        :param x_key_padding_mask (N, L)
        :return: src_hidden: (N, L, H)
        """
        # contextualised x with itself
        x = self.self_attention(q=x, k=x, v=x, key_padding_mask=x_key_padding_mask) + x  # residual
        # apply linear transformation to each positional identically but independently
        x = self.ffn(x) + x  # residual
        return x


class Encoder(nn.Module):

    def __init__(self, hidden_size: int, encoding_size: int, heads: int, max_length: int,
                 depth: int, ffn_size: int, dropout: float) -> None:
        """
        hidden_size: original feature level
        encoding_size: encoded feature level (reduced)
        heads: num of heads
        max_length: max token length
        depth: how many times to repeat encoder
        """
        super().__init__()
        self.self_attention = MultiHeadAttentionLayer(hidden_size, encoding_size, heads, max_length)
        # TODO - ffn, EncoderLayer
        self.layers = torch.nn.ModuleList([
            EncoderLayer(hidden_size, encoding_size, heads, max_length, ffn_size, dropout)
            for _ in range(depth)
        ])

    # override
    def forward(self, x: torch.Tensor, x_key_padding_mask: torch.LongTensor):
        """
        x: (N, L, H)
        return a vector with contextual meanings encoded
        """
        for layer in self.layers:
            x = layer(x, x_key_padding_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, encoding_size: int, heads: int, max_length: int,
                 ffn_size: int, dropout: float):
        super().__init__()

        # masked, multi-head self-attention layer
        self.masked_mhsa_layer = MultiHeadAttentionLayer(hidden_size, encoding_size, heads, max_length, masked=True)
        # not masked, multi-head encoder-decoder attention layer
        self.mheda_layer = MultiHeadAttentionLayer(hidden_size, encoding_size, heads, max_length)
        # position-wise feed-forward network
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                x_key_padding_mask: torch.LongTensor, memory_key_padding_mask: torch.LongTensor) \
            -> torch.Tensor:
        """
        :param: x (N, L, H)
        :param: memory (the output of the encoder) (N, L, H)
        :param: x_key_padding_mask  (N, L)
        :param: memory_key_padding_mask (N, L)
        :return: x (contextualised)
        """
        # contextualise x with itself
        x = self.masked_mhsa_layer.forward(q=x, k=x, v=x, key_padding_mask=x_key_padding_mask) + x  # residual
        # soft-align memory with respect to x
        x = self.mheda_layer.forward(q=x, k=memory, v=memory, key_padding_mask=memory_key_padding_mask) + x  # residual
        # apply linear transformation to each position independently but identically
        x = self.ffn(x) + x  # residual
        return x


class Decoder(nn.Module):

    def __init__(self, hidden_size: int, encoding_size: int, heads: int, max_length: int,
                 depth: int, ffn_size: int, dropout: float) -> None:
        """
        hidden_size: original feature level
        encoding_size: encoded feature level (reduced)
        heads: num of heads
        max_length: max token length
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([
            DecoderLayer(hidden_size, encoding_size, heads, max_length, ffn_size, dropout)
            for _ in range(depth)
        ])

    # override
    def forward(self, x: torch.Tensor, encoder_contexts: torch.Tensor,
                x_key_padding_mask: torch.LongTensor, c_key_padding_mask: torch.LongTensor):
        """
        x: (N, L, H)
        encoder_contexts: encoder output
        x_key_padding_mask: x key_padding_mask
        c_key_padding_mask: anchor_contexts key_padding_mask
        return vector with contextual meanings encoded
        """
        contexts = self.masked_self_attention(q=x, k=x, v=x, key_padding_mask=x_key_padding_mask)
        alignments = self.encoder_decoder_attention(q=contexts, k=encoder_contexts, v=encoder_contexts,
                                                    key_padding_mask=c_key_padding_mask)
        return alignments

        # TODO: ffn(feed-forward), residual connection


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hidden_size: int, encoding_size: int, heads: int, max_length: int, masked: bool = False) -> None:
        """
        hidden_size = H
        encoding_size = E
        heads = number of self attention heads
        max_length = max length of L
        masked = masked MultiHeadAttention?
        """
        super().__init__()
        assert encoding_size % heads == 0, "hidden_size(H) --> encoding_size(E) must be divisible by # of heads"
        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.heads = heads  # how many heads?
        self.max_length = max_length  # L
        self.masked = masked

        self.linear_q = nn.Linear(hidden_size, encoding_size)
        self.linear_k = nn.Linear(hidden_size, encoding_size)
        self.linear_v = nn.Linear(hidden_size, encoding_size)
        self.linear_o = nn.Linear(encoding_size, hidden_size)

        self.norm = torch.nn.LayerNorm(hidden_size)

        # const tensor (for masked attention)
        self.register_buffer("subsequent_mask", subsequent_mask(max_length))  # (L, L)

    # override
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        q: (N, L, H)
        k: (N, L, H)
        v: (N, L, H)
        key_padding_mask: (N, L)
        return contexts (N, L, H)
        """
        N, _, _ = q.size()

        q = self.linear_q(q)  # (N, L, E)
        k = self.linear_k(k)  # (N, L, E)
        v = self.linear_v(v)  # (N, L, E)

        q = q.reshape(N, self.max_length, self.heads, self.encoding_size // self.heads)  # (N, L, heads, head_size)
        k = k.reshape(N, self.max_length, self.heads, self.encoding_size // self.heads)  # (N, L, heads, head_size)
        v = v.reshape(N, self.max_length, self.heads, self.encoding_size // self.heads)  # (N, L, heads, head_size)
        q = q.permute(0, 2, 1, 3)  # (N, heads, L, head_size)
        k = k.permute(0, 2, 1, 3)  # (N, heads, L, head_size)
        v = v.permute(0, 2, 1, 3)  # (N, heads, L, head_size)

        # (N, L) --> (N, heads, L, L)
        key_mask = key_padding_mask.reshape(N, 1, self.max_length, 1).repeat(1, self.heads, 1, self.max_length)

        if self.masked:  # for auto-regressive inference at decoder
            key_subsequent_mask = self.subsequent_mask.unsqueeze(0).unsqueeze(0).\
                expand(N, self.heads, -1, -1)  # (L, L) -> (1, 1, L, L) -> (N, heads, L, L)
            key_mask = torch.logical_and(key_mask, key_subsequent_mask).long()

        contexts = scaled_dot_product_attention(q, k, v, key_mask)
        # concat(head_1, head_2, ... head_heads): concatenate multiple alignments
        # (N, heads, L, head_size) -> (N, L, heads, head_size) -> (N, L, E)
        concats = contexts.transpose(1, 2).contiguous() \
            .reshape(-1, self.max_length, self.encoding_size)  # .contiguous() --> merging heads, head_size

        # concat(head_1, head_2, ... head_heads) * W_o: aggregate alignments
        contexts = self.linear_o(concats)  # (N, L, E) * (E, H) -> (N, L, H)
        return self.norm(contexts)  # layer normalisation
