import torch

with open("bible.txt",'r',encoding='utf-8') as file:
    content = file.read()
chars = sorted(set(content))
vocab_size = len(chars)
strtoint = {ch: i for i, ch in enumerate(chars)}
inttostr = {i: ch for i, ch in enumerate(chars)}
def encoder(s):
    return [strtoint[c] for c in s]
def decoder(ids):
    return ''.join([inttostr[i] for i in ids])


ids = encoder(content)
data = torch.tensor(ids, dtype=torch.long)
print(data.shape)
print(data[:20])
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
block_size = 8
batch_size = 4
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y
x, y = get_batch('train')
print(x.shape, y.shape)
print(decoder(x[0].tolist()))
print(decoder(y[0].tolist()))
# print(len(content))
# print(vocab_size)
# print(''.join(chars))
# print(content[:500])