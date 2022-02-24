"""https://github.com/jl749/the-clean-transformer/blob/f692255556693915e5051eaf27845dff0f286e06/cleanformer
/datamodules.py#L49-L50 """
from cleanformer.builders import TrainInputsBuilder, LabelsBuilder
from cleanformer.fetchers import fetch_tokenizer


tokenizer = fetch_tokenizer("eubinecto", ver="wp")
max_length = 10

inputs_builder = TrainInputsBuilder(tokenizer=tokenizer, max_length=max_length)
labels_builder = LabelsBuilder(tokenizer=tokenizer, max_length=max_length)

kors = ["난 널 사랑해"]
engs = ["I love you"]

X = inputs_builder(srcs=kors, tgts=engs)  # (N, 2, 2, L)
Y = labels_builder(tgts=engs)  # (N, L)

print(X)
print(Y)
