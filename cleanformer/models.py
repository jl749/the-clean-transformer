from typing import Dict, Tuple
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
        
        self.encoder = Encoder()
        self.decoder = Decoder()


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
    raise NotImplementedError

class Decoder(torch.nn.Module):
    raise NotImplementedError