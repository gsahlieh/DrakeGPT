# IMPORTS
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import device
from utils import decode, create_dicts
from model import DrakeGPT

# LOAD DATA
with open('training-data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
_, itos, _ = create_dicts(text)

# LOAD TRAINED MODEL
model = DrakeGPT()
model.load_state_dict(torch.load('my-model.pth', map_location=torch.device('cpu'))) # specify .pth file and if using GPU, change 'cpu' to 'cuda'
model = model.to(device)
model.eval()

# GENERATE TEXT
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist(), itos) # change max_new_tokens to change length of generated text
with open('output.txt', 'w') as f: # specify output file
    f.write(generated_text)
