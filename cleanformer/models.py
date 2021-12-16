from typing import Dict, Tuple
from cleanformer.tensors import subsequent_mask
import torch
from pytorch_lightning import LightningModule


class Transformer(LightningModule):
    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):  # noqa
        super().__init__()

        # inherited func SAVE ALL hyperparms in hparmas dictionary
        self.save_hyperparameters()
        
        # TODO: implement transformer

        # A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)  # (vocab_size, hidden_size) embedding table
        
        self.encoder = Encoder(hidden_size, heads, max_length)
        self.decoder = Decoder(hidden_size, heads, max_length)


    # override
    def forward(self, src_ids: torch.LongTensor, tgt_ids: torch.Tensor,
                    src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        src_ids, tgt_ids --> output hidden vector (N, L, H)

        return hidden vector output from decoder
        """
        
        src = self.token_embeddings(src_ids)  # linear layer (N, L) --> (N, L, H)
        tgt = self.token_embeddings(tgt_ids)  # (N, L) --> (N, L, H)

        # POSITIONAL ENCODING
        # TODO: later

        memory = self.encoder.forward(src)  # (N, L, H) --> (N, L, H)
        hidden = self.decoder.forward(tgt, memory)  # (N, L, H) --> (N, L, H)
        return hidden

    # override
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> Dict:
        X, Y = batch  # (N, 2, 2, L) (N, L)
        # [ N = batch size, 2(src(KO) / target(ENG)), 2(ids / mask(padding)), L(sequence max length) ]
        
        # encoder inputs  (N, L)
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # decoder inputs  (N, L)
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]

        hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)
        classifier = self.token_embeddings.weight  # (V, H)  V = BoW classes
        logits = torch.einsum("nlh,vh->nvl", hidden, classifier)
        loss = torch.nn.functional.cross_entropy(logits, Y)  # CEE requires (N, classes, L), (N, L)
        loss = loss.sum()  # (N,) -> (,)
        
        return {
            "loss": loss,
        }

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        param X (N, 2, 2, L)
        return label_ids (N, L)
        """
        
        # encoder inputs  (N, L)
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # decoder inputs  (N, L)
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]

        for time in range(0, self.hparams['max_length']-1):
            # --- (N, L, H)
            hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)  # (N, L, H)
            classifier = self.token_embeddings.weight  # (V, H)  V = BoW classes

            logits = torch.einsum("nlh,vh->nlv", hidden, classifier)  # THIS IS greedy decoding, look up beem search algo!!!
            ids = torch.argmax(logits, dim=2) # (N, L, V)  -->  (N, L)

            # pass current output to next input
            next_id = ids[:, time]  # (N, L) -->  (N, )

            tgt_ids[:, time+1] = next_id
            tgt_key_padding_mask[:, time+1] = 0  # not padding anymore

        label_ids = tgt_ids  # final sequence ids (N, L)
        return label_ids



class Encoder(torch.nn.Module):

    def __init__(self, hidden_size: int, heads: int, max_length: int) -> None:
        super().__init__()

        self.multiHead_selfAttention_layer = MultiHeadAttentionLayer(hidden_size, heads, max_length, masked=False)
        # TODO - ffn

    # override
    def forward(self, x: torch.Tensor):
        """
        x: (N, L, H)
        return vector with contectual meanings encoded
        """
        contexts = self.multiHead_selfAttention_layer.forward(q=x, k=x, v=x)
        return contexts


class Decoder(torch.nn.Module):
    
    def __init__(self, hidden_size: int, heads: int, max_length: int) -> None:
        super().__init__()
        self.masked_multiHead_selfAttention_layer = MultiHeadAttentionLayer(hidden_size, heads, max_length, masked=True)

    # override
    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        """
        x: (N, L, H)
        memory: encoder output
        return vector with contectual meanings encoded
        """
        contexts = self.multiHead_selfAttention_layer.forward(q=x, k=x, v=x)
        return contexts

        # TODO: ffn(feed-forward), residual connection



class MultiHeadAttentionLayer(torch.nn.Module):
    
    def __init__(self, hidden_size: int, heads: int, max_length: int, masked: bool) -> None:
        """
        hidden_size = H
        heads = number of self attention heads
        max_length = max length of L
        masked = masked MultiHeadAttention?
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads  # how many heads?
        self.max_length = max_length
        self.masked = masked

        assert self.hidden_size % self.heads == 0  # hidden_size H must be divisible by heads

        self.linear_q = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_k = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_v = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_o = torch.nn.Linear(hidden_size, hidden_size)

        # const tensor in register_buffer
        # 나중에 model.device("cuda") 모델과 함께 상수텐서도 같이 GPU load
        self.register_buffer("subsequent_mask", subsequent_mask(max_length))

    # override
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: (N, L, H)
        k: (N, L, H)
        v: (N, L, H)
        """
        N, _, _ = q.size()

        q = self.linear_q(q)  # (N, L, H) * (H, H)  -->  (N, L, H)
        k = self.linear_k(k)  # (N, L, H) * (H, H)  -->  (N, L, H)
        v = self.linear_v(v)  # (N, L, H) * (H, H)  -->  (N, L, H)

        head_size = self.hidden_size//self.heads
        # split heads: (N, L, H)  -->  (N, L/heads, heads)
        q = q.reshape(N, self.max_length, self.heads, head_size)
        v = v.reshape(N, self.max_length, self.heads, head_size)
        k = k.reshape(N, self.max_length, self.heads, head_size)

        sims = torch.einsum("nqhs,nkhs->nhqk", q, k)  # (N, L, heads, head_size) * (N, L, heads, head_size) --> (N, heads, L, L)

        # masking (auto-regressive)
        if self.masked:
            # mask = subsequent_mask(L)  # (L, L)  CUDA err, do not create tensor in func, create inside constructor
            mask = self.subsequent_mask.reahspe(1, 1, self.max_length, self.max_length)\
                        .expand(N, self.heads, -1, -1)  # (1, 1, L, L)  -->  (N, heads, L, L)
            sims = torch.masked_fill(sims, mask == 0, value=float('-inf'))  # 

        attentions = torch.softmax(sims, dim=3)  # (N, L(q.length), L(k.length)),  foreach query calcuate keys softmax

        contexts = torch.einsum("nhqk,nkhs->nqhs", attentions, v)  # (N, heads, L, L) * (N, L, H, head_size)  -->  (N, L, heads, head_size)
        contexts = contexts.reshape(N, self.max_length, self.hidden_size)  # (N, L, heads, head_size)  -->  (N, L, H)
        contexts = self.linear_o(contexts)  # (N, L, H)  -->  (N, L, H)
        return contexts