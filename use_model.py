# IMPORTS
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import DrakeGPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


# LOAD TRAINED MODEL
model = DrakeGPT()
model.load_state_dict(torch.load(
    'drake-model.pth', map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

# GENERATE TEXT
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(
    context, max_new_tokens=500)[0].tolist())
with open('output10.txt', 'w') as f:
    f.write(generated_text)
