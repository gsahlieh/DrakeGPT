import torch
import torch.nn as nn
from torch.nn import functional as F
from config import learning_rate, train_split_portion, max_iters, eval_interval, device, eval_iters
from utils import encode, create_dicts, get_batch, estimate_loss
from model import DrakeGPT


# LOAD DATA
with open('training-data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
stoi, _, _ = create_dicts(text)
data = torch.tensor(encode(text, stoi), dtype=torch.long)


# CREATE TRAIN AND TEST SPLITS
n = int(train_split_portion * len(data))
train_data = data[:n]
val_data = data[n:]

# CREATE MODEL INSTANCE
model = DrakeGPT()
m = model.to(device) # use GPU if available

# CREATE OPTIMISER
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# TRAIN MODEL
for iter in range(max_iters):

    # Evaluate model on train and val data
    if iter % eval_interval == 0:
        losses = estimate_loss(model, get_batch, train_data, val_data, eval_iters)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch(train_data)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# SAVE TRAINED MODEL
torch.save(model.state_dict(), 'my-model.pth')
