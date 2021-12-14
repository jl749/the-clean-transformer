from typing import Dict, Tuple
import torch
from pytorch_lightning import LightningModule


class Transformer(LightningModule):
    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):  # noqa
        super().__init__()
        self.save_hyperparameters()
        
        # TODO: implement transformer
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)  # vocab_size -> hidden_size
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src_ids: torch.LongTensor, tgt_ids: torch.Tensor,
                    src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
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
        
        # encoder inputs
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # decoder inputs
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]

        hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)
        classifier = self.token_embeddings.weight  # (V, H)  V = BoW classes
        logits = torch.einsum("nlh,vh->nvl", hidden, classifier)
        loss = torch.nn.functional.cross_entropy(logits, Y)  # CEE requires (N, classes, L), (N, L)
        loss = loss.sum()  # (N,) -> (,)
        
        return {
            "loss": loss,
        }


class Encoder(torch.nn.Module):
    raise NotImplementedError

class Decoder(torch.nn.Module):
    raise NotImplementedError