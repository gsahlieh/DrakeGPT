import torch
import torch.nn as nn
from torch.nn import functional as F
from model import DrakeGPT


# CONSTANTS
batch_size = 64
block_size = 256
learning_rate = 3e-4
train_split_portion = 0.9
max_iters = 5000
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_heads = 12
n_layers = 12
dropout = 0.2


# LOAD DATA
with open('training-data.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# CREATE ENCODING AND DECODING FUNCTIONS
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


# ENCODE ALL DATA
data = torch.tensor(encode(text), dtype=torch.long)


# CREATE TRAIN AND TEST SPLITS
n = int(train_split_portion * len(data))
train_data = data[:n]
val_data = data[n:]


# CREATE MINI-BATCH
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# CREATE MODEL INSTANCE
model = DrakeGPT()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()))

# CREATE OPTIMISER
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# TRAIN MODEL
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# SAVE TRAINED MODEL
torch.save(model.state_dict(), 'model.pth')
