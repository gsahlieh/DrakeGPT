"""
Full definition of a GPT Language Model, all of it in this single file.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import block_size, train_split_portion, device, eval_iters, n_embd, n_heads, n_layers, dropout
from utils import create_dicts, encode_all_data


# LOAD DATA
with open('training-data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
stoi, itos, vocab_size = create_dicts(text)
data = encode_all_data(text, stoi)


# CREATE TRAIN AND TEST SPLITS
n = int(train_split_portion * len(data)) # train/val split = 90/10
train_data = data[:n]
val_data = data[n:]


# SINGLE ATTENTION HEAD
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # Standard query, key, value projections for all tokens
        q = self.query(x) # (B, T, head_size)
        k = self.key(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # lower triangular mask to prevent attending to future tokens
        wei = F.softmax(wei, dim=-1) # for probabilities

        out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)

        return out


# MULTIHEAD ATTENTION
class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concatenate all heads on the `head` axis
        out = self.proj(out) # Project back down to `n_embd` dimensions
        out = self.dropout(out)
        return out


# FEED FORWARD LAYER
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        # Multilayer Perceptron (MLP) with one hidden layer 
        self.net = nn.Sequential(
            # standard hidden layer size of 4 * n_embd
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# BLOCK
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connection + pre-layer norm
        x = x + self.ffwd(self.ln2(x)) # Residual connection + pre-layer norm
        return x


# GPT MODEL
class DrakeGPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # token embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # position embedding table
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads=n_heads) for _ in range(n_layers)]) # main body of model
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # output layer for to produce logits over vocab

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Sampling tokens one at a time
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, 1, C) -> (B, C)
            probs = F.softmax(logits, dim=-1) # convert logits to probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample single token
            idx = torch.cat((idx, idx_next), dim=1) # append to sequence
        return idx


model = DrakeGPT()
