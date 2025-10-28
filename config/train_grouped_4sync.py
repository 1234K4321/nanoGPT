# Training configuration for Grouped model (4 syncs)
# Group 3 layers together, 12 layers / 3 = 4 groups = 4 syncs

# Model architecture
sync_strategy = 'grouped'
attn_sync_layers = None  # Not used for grouped
layers_per_group = 3  # 3 layers per group -> 4 groups -> 4 syncs

# I/O
out_dir = 'out-grouped-4sync'
eval_interval = 500
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# AdamW optimizer
learning_rate = 6e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 20000
min_lr = 6e-5

# DDP settings
backend = 'nccl'

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
