with open("input.txt", "r") as file:
    text = file.read()

print(len(text))

# create a mapping of unique characters to integers
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)
print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])
