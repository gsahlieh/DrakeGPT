# IMPORTS
import torch
import torch.nn as nn
from torch.nn import functional as F


# PROPER CONSTANTS
# batch_size = 64
# block_size = 256
# learning_rate = 3e-4
# train_split_portion = 0.9
# max_iters = 5000  # Number of back props and parameter updates
# eval_interval = 500  # After how many iterations we evaluate the model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_heads = 12
# n_layers = 12
# dropout = 0.2

# HYPERPARAMETERS I CAN GET WORKING ON MY PC
batch_size = 16
block_size = 8
learning_rate = 3e-4
train_split_portion = 0.9
max_iters = 5000  # Number of back props and parameter updates
eval_interval = 500  # After how many iterations we evaluate the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_heads = 4
n_layers = 6
dropout = 0.2


# LOAD DATA
with open('training-data.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# CREATE ENCODING AND DECODING FUNCTIONS
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# Encode
def encode(s): return [stoi[c] for c in s]
# Decode
def decode(l): return ''.join([itos[i] for i in l])


# ENCODE ALL DATA
data = torch.tensor(encode(text), dtype=torch.long)


# CREATE TRAIN AND TEST SPLITS
n = int(train_split_portion * len(data))
train_data = data[:n]
val_data = data[n:]


# CREATE MINI-BATCH FROM TRAIN OR TEST DATA
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


class Head(nn.Module):
    """A single attention head"""

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # A buffer is basically a constant that is created
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**(-0.5)

        # Reason for indexing is that idx might be collected at the end of a file where the sequence is shorter than block_size, hence tril needs to be reshaped
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Randomly prevents some tokens from communicating with other tokens, think of the placement here as industry standard
        wei = F.softmax(wei, dim=-1)

        out = wei @ v

        return out


# MULTIHEAD ATTENTION
# Difference between self-attention and multihead attention is for a specific token, instead of breaking down into q, k, v each of size = head_size, it is broken down into n_head numbers of q, k, v each of size = head_size/n_head. This means that n_head queries can be asked (n_head query vectors) across all other tokens using their key vectors for that specific query
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(n_heads)])

        # This projection layer just provides more computation after concatenating the outputs of each head. Doesn't hurt. Generally however, this projection layer would also be used to shrink the concatenation of the heads to the original embedding size as well as provide additional computation
        self.proj = nn.Linear(head_size * n_heads, n_embd)

        # Dropout (prevents overfitting, regularisation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


# Feed Forward layer: Basically allows more computation to happen on the output of the multi-attention layer by adding parameters
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Reason for multipying with 4 is because in the MLP paper Andrej referenced, they had an inner-layer dimensionality of 4 times the input and output, hence allowing more computation while not requiring chancing input and output dimensions (think about course 4 of DL coursera with MobileNetv2 and depth-wise computations that happened in the middle of the network)

            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),

            # The adding of this is just because the paper says so, basically facilitates the expanding and contracting of the network to allow for more computation to take place within the feedforward layer
            nn.Linear(4 * n_embd, n_embd),

            # Dropout (prevents overfitting, regularisation)
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# BLOCK
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads

        # Create a head layer. Basically tok_emb + pos_emb is passed in, q, k, v are created for each token and then the after attention calculation, each token has a new value respective to their context
        # COMMUNICATION (between tokens)
        self.sa = MultiHeadAttention(n_heads, head_size)

        # This is a feed forward layer which takes in the output of the multihead attention layer and outputs a new value for each token by multiplying by a weight matrix
        # COMPUTATION
        self.ffwd = FeedForward(n_embd)

        # NORMALISATION BEFORE MULTIHEAD ATTENTION (different from original paper which does normalisation after multihead attention, recent innovations have shown that normalisation before multihead attention is better). Think of this normalisation as similar to z-score normalisation but with learnable parameters to make it better as well as epsilon to prevent division by 0
        self.ln1 = nn.LayerNorm(n_embd)

        # NORMALISATION BEFORE FEEDFORWARD (same, is different from original paper which does normalisation after feedforward, recent innovations have shown that normalisation before feedforward is better)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections here. Helps eliminate the vanishing gradient problem which particularly affects deep networks
        # See normalisations are also happening before input into the multihead attention and feedforward layers
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# BIGRAM MODEL
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Basically a table which stores a value for each position in the sequence and each feature in the embedding, there is a value stored. This value is then added to the embedding of each token at that position for all batches
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads=n_heads) for _ in range(n_layers)])

        # This normalisation is standard before output is passed to last linear layer which decodes into logits for each possible next token
        self.ln_f = nn.LayerNorm(n_embd)

        # This linear layer takes in the embedding of a single token and outputs the logits for each possible next token (higher the logit, the more likely to pick that token as next)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # For all batches and sequences, retrieves the embeddings for every token, hence C = embedding size. The 32 embeddings of each token are like how much royal, gender etc. each token is
        # Adds the embeddings for all tokens
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)

        # For each token in the sequence it retrieves the positional embedding for that token. Each batch has same positional embedding, e.g. 1st token in 1st batch has same positional embedding as 1st token in 2nd batch
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T, C)

        # Broadcasting happens so: (B, T, C) + (T, C) -> (B, T, C) + (1, T, C) -> (B, T, C) + (B, T, C) -> (B, T, C)
        x = tok_emb + pos_emb

        x = self.blocks(x)

        # Takes in all batches and sequences and for each token, generates the logits for each possible next token
        # (B, T, V) where V = vocab_size as logits are for each possible next character
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crops idx to the last block_size tokens so positional embeddings can work (only goes up to block_size)
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# CREATE MODEL INSTANCE
model = BigramLanguageModel()
m = model.to(device)
# Number of parameters = ~10 million
# Number of characters trained on drake dataset = 2641878
print(sum(p.numel() for p in m.parameters()))

# CREATE OPTIMISER
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# TRAIN MODEL
for iter in range(max_iters):

    # Should you evaluate model on this iteration
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
torch.save(model.state_dict(), 'model1.pth')
