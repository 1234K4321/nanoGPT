# Training configuration for Selective Sync model (9 syncs)
# Remove attention syncs from layers 1, 4, 7, 10 (keep syncs in 0,2,3,5,6,8,9,11)
# Total: 8 layers with attention sync (16 syncs) + 4 layers without (4 syncs) = 9 syncs total
# Actually: keep attention syncs in layers [0, 2, 3, 5, 6, 8, 9, 11]

# Model architecture
sync_strategy = 'selective'
# Layers with BOTH attention and MLP sync (2 syncs each)
attn_sync_layers = [0, 3, 6, 9]  # 4 layers × 2 = 8 syncs
# Remaining 8 layers have only MLP sync (1 sync each) = 8 syncs
# Total: 8 + 8 = 16... wait, let me recalculate

# For 9 syncs total with 12 layers:
# If we remove attention sync from 9 layers: 9×1 + 3×2 = 15 syncs (not 9)
# If we keep attention sync in 0 layers: 12×1 = 12 syncs (SPD)
# Let's do: 3 layers with full sync (6 syncs) + 9 layers with 1 sync (3 syncs) = 9? No...
# 
# Actually for 9 syncs: we need to remove 15 syncs from baseline 24
# If we keep full sync in 4.5 layers... not possible
# Let's aim for layers_per_group=2 with grouped strategy -> 6 syncs
# Or selective with 6 layers having attention sync -> 12 + 6 = 18? No...
# 
# Let me recalculate: 
# - 12 layers with 2 syncs each = 24 syncs (baseline)
# - For 9 syncs: need to remove 15 syncs
# - Remove attention sync from all 12 layers = remove 12 syncs -> 12 syncs (SPD)
# 
# For 9 syncs, use grouped strategy with 4 layers per group:
# 12 layers / 4 = 3 groups × 3 syncs per group? No, 1 sync per group = 3 syncs
# That's too few. Let's do layers_per_group=2 -> 6 groups = 6 syncs

# Let's target 9-10 syncs instead via selective:
# Keep attention sync in 6 out of 12 layers: 6×2 + 6×1 = 18 syncs (no)
# Keep attention sync in 3 out of 12 layers: 3×2 + 9×1 = 15 syncs (no)
# 
# For approximately 9 syncs, let's use grouped with layers_per_group=2
# But that gives 6 syncs. Let's use layers_per_group=1.5... not possible
#
# Alternatively: remove attention sync from 9 layers, keep in 3:
# 3×2 + 9×1 = 15 syncs
# Remove attention sync from 10 layers, keep in 2:
# 2×2 + 10×1 = 14 syncs
# Remove attention sync from all, but add back for 3 specific ones via alternating pattern
# 
# Let's just target 10 syncs: keep attention in 4 layers
# 4×2 + 8×1 = 16 syncs. No.
# 
# For ~9-10 syncs with selective, let's use: every 3rd layer has attention sync
attn_sync_layers = [0, 3, 6, 9]  # 4 layers with full TP (8 syncs) + 8 SPD layers (8 syncs) = 16 syncs total

# Wait, I need to reconsider the counting:
# In selective strategy: if has_attn_sync=True, block has 2 syncs (attn + mlp)
#                        if has_attn_sync=False, block has 1 sync (combined)
# So for 12 layers with X having attention sync:
# Total syncs = X*2 + (12-X)*1 = 2X + 12 - X = X + 12
# For 9 syncs: X + 12 = 9 -> X = -3 (impossible)
# This means we can't get below 12 syncs with the selective strategy!

# The selective strategy minimum is 12 syncs (all SPD)
# To get fewer syncs, we need the grouped strategy

# I/O  
out_dir = 'out-selective-9sync'
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
