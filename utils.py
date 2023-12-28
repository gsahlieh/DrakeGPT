import torch
from config import block_size, batch_size, device

# CREATE ENCODING AND DECODING DICTIONARIES + DEFINE VOCAB SIZE
def create_dicts(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}  # string to index dict
    itos = {i: ch for i, ch in enumerate(chars)}  # index to string dict
    return stoi, itos, vocab_size

# ENCODING FUNCTION
def encode(s, stoi): 
    return [stoi[c] for c in s]  # encode string to list of indices

# DECODING FUNCTION
def decode(l, itos): 
    return ''.join([itos[i] for i in l])  # decode list of indices to string

# ENCODE ALL TRAINING DATA
def encode_all_data(text, stoi):
    return torch.tensor(encode(text, stoi), dtype=torch.long)

# CREATE MINI-BATCH
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ESTIMATE LOSS
@torch.no_grad()
def estimate_loss(model, get_batch, train_data, eval_data, eval_iters):
    out = {}
    model.eval()
    data = {'train': train_data, 'val': eval_data}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data[split])
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out