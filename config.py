import torch
# batch_size = 64
# block_size = 256 # previous character context size when sampling
# learning_rate = 3e-4
# train_split_portion = 0.9
# max_iters = 5000
# eval_interval = 500
# device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available
# eval_iters = 200
# n_embd = 384
# n_heads = 6
# n_layers = 6
# dropout = 0.2 # dropout prob (increase for greater regularization)


batch_size = 1
block_size = 1 # previous character context size when sampling
learning_rate = 3e-4
train_split_portion = 0.9
max_iters = 1000
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available
eval_iters = 200
n_embd = 36
n_heads = 2
n_layers = 2
dropout = 0.2 # dropout prob (increase for greater regularization)