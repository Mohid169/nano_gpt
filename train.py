with open("input.txt", "r") as file:
    text = file.read()

# print(len(text))

# create a mapping of unique characters to integers
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(chars)
# print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# print(encode("hii there"))
# print(decode(encode("hii there")))

import torch

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(len(train_data), len(val_data))
block_size = 8
train_data = train_data[: block_size + 1]
print(train_data)

x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
# print(f"when input is {context} the target is {target}")


torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will we process in parallel
block_size = 8  # context length for predictions


def get_batch(split):
    # generate a small batch of data batch of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
for b in range(batch_size):  # batch size of the tensor
    for t in range(block_size):  # time dimension
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"when input is {context} the target is {target}")
