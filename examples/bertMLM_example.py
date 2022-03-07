"""
https://gist.github.com/eubinecto/ac6e218f84647a03295bcba5d81647fa
"""

from transformers import BertForMaskedLM, BertTokenizer
from torch.nn import functional as F


BATCH = [
    ("I understand Tesla's vision.", "Haha, that's a nice [MASK]."),  # pun?
    ("[MASK] are monkey's favourite fruit.", None)  # bananas? - this is something GPT1 cannot do.,
]
BERT_MODEL = "bert-base-uncased"


def main():
    global BATCH, BERT_MODEL
    # the pre-trained model and tokenizer (may take a while if they have not been downloaded yet)
    # the models will be saved to ~/.cache/transformers
    mlm = BertForMaskedLM.from_pretrained(BERT_MODEL)  # Masked Language Model
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    print(mlm.config)
    # encode the batch into input_ids, token_type_ids and attention_mask
    encoded = tokenizer(BATCH,
                        add_special_tokens=True,
                        return_tensors="pt",
                        truncation=True,
                        padding=True)
    # mlm houses a pretrained bert_ucl model
    # mlm.bert(encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids']), **encoded for **kwargs
    outputs = mlm.bert(**encoded)
    H = outputs[0]  # the hidden representation of the batch.
    print(H.size())  # (N=2, L) -> (N=2, L, Hidden=768)
    # get the hidden representations for the masked tokens
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    masked_ids = [
        input_id.tolist().index(mask_id)
        for input_id in encoded['input_ids']
    ]  # masked index
    hidden_masked_1 = H[0, masked_ids[0], :]  # (N, L, 768) -> (1, 768)
    hidden_masked_2 = H[1, masked_ids[1], :]  # (N, L, 768) -> (1, 768)
    # mlm also houses the masked language model (just another FFN layer). here, the outputs are logits.
    logits_1 = mlm.cls(hidden_masked_1)  # (1, 768) * (768, |S|) -> (1, |S|)
    logits_2 = mlm.cls(hidden_masked_2)  # (1, 768) * (768, |S|) -> (1, |S|)
    # normalise them to  [0, 1]
    probs_1 = F.softmax(logits_1, dim=0)
    probs_2 = F.softmax(logits_2, dim=0)
    # decode the predictions, the probability distributions.
    pred_1 = [
        (tokenizer.decode([idx]), logit)
        for idx, logit in enumerate(probs_1.tolist())
    ]
    pred_2 = [
        (tokenizer.decode([idx]), logit)
        for idx, logit in enumerate(probs_2.tolist())
    ]
    # sort them in descending order.
    print(sorted(pred_1, key=lambda x: x[1], reverse=True)[:10])  # will "pun" appear in the top 10's?
    print(sorted(pred_2, key=lambda x: x[1], reverse=True)[:10])  # will "bananas" appear in the 10's?


if __name__ == '__main__':
    main()
