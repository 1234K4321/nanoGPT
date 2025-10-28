"""
Tensor Parallel version of GPT model (simulated for 2 nodes).
This version demonstrates how tensor parallelism would work with 2 GPUs
by splitting computations and adding explicit synchronization points (all-reduce),
all simulated within a single process.

Based on Megatron-LM tensor parallelism:
- MLP: First linear split by columns, second by rows, all-reduce at end
- Attention: Q/K/V split by columns (heads), output projection split by rows, all-reduce at end
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class TensorParallelCausalSelfAttention(nn.Module):
    """
    Tensor Parallel Self-Attention simulated for 2 nodes.
    
    Partitioning strategy (Megatron-LM):
    - Q, K, V weight matrices split by columns (split attention heads across GPUs)
    - Output projection split by rows
    - All-reduce after output projection
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % 2 == 0, "Number of heads must be divisible by 2 for 2-way parallelism"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Split heads across 2 "GPUs"
        self.n_head_per_partition = config.n_head // 2
        
        # GPU 1: First half of heads
        # Q, K, V projections split by columns (each GPU handles subset of heads)
        self.c_attn_1 = nn.Linear(config.n_embd, 3 * config.n_embd // 2, bias=config.bias)
        self.c_proj_1 = nn.Linear(config.n_embd // 2, config.n_embd, bias=config.bias)
        
        # GPU 2: Second half of heads
        self.c_attn_2 = nn.Linear(config.n_embd, 3 * config.n_embd // 2, bias=config.bias)
        self.c_proj_2 = nn.Linear(config.n_embd // 2, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # ========== GPU 1: First half of heads ==========
        # Calculate Q, K, V for first half of heads
        qkv_1 = self.c_attn_1(x)  # (B, T, 3 * n_embd // 2)
        q_1, k_1, v_1 = qkv_1.split(self.n_embd // 2, dim=2)
        
        # Reshape for attention computation
        k_1 = k_1.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        q_1 = q_1.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        v_1 = v_1.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        
        # Compute attention for first partition
        if self.flash:
            y_1 = torch.nn.functional.scaled_dot_product_attention(
                q_1, k_1, v_1, attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            att_1 = (q_1 @ k_1.transpose(-2, -1)) * (1.0 / math.sqrt(k_1.size(-1)))
            att_1 = att_1.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att_1 = F.softmax(att_1, dim=-1)
            att_1 = self.attn_dropout(att_1)
            y_1 = att_1 @ v_1
        
        y_1 = y_1.transpose(1, 2).contiguous().view(B, T, C // 2)
        
        # Output projection for first partition (split by rows)
        z_1 = self.c_proj_1(y_1)  # (B, T, C)
        
        # ========== GPU 2: Second half of heads ==========
        # Calculate Q, K, V for second half of heads
        qkv_2 = self.c_attn_2(x)  # (B, T, 3 * n_embd // 2)
        q_2, k_2, v_2 = qkv_2.split(self.n_embd // 2, dim=2)
        
        # Reshape for attention computation
        k_2 = k_2.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        q_2 = q_2.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        v_2 = v_2.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        
        # Compute attention for second partition
        if self.flash:
            y_2 = torch.nn.functional.scaled_dot_product_attention(
                q_2, k_2, v_2, attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            att_2 = (q_2 @ k_2.transpose(-2, -1)) * (1.0 / math.sqrt(k_2.size(-1)))
            att_2 = att_2.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att_2 = F.softmax(att_2, dim=-1)
            att_2 = self.attn_dropout(att_2)
            y_2 = att_2 @ v_2
        
        y_2 = y_2.transpose(1, 2).contiguous().view(B, T, C // 2)
        
        # Output projection for second partition (split by rows)
        z_2 = self.c_proj_2(y_2)  # (B, T, C)
        
        # ========== ALL-REDUCE: Sum results from both GPUs ==========
        y = z_1 + z_2
        
        # Apply dropout after all-reduce
        y = self.resid_dropout(y)
        
        return y


class TensorParallelMLP(nn.Module):
    """
    Tensor Parallel MLP simulated for 2 nodes.
    
    Partitioning strategy (Megatron-LM):
    - First linear layer (c_fc) split by columns
    - Second linear layer (c_proj) split by rows
    - All-reduce after second linear layer
    """

    def __init__(self, config):
        super().__init__()
        
        # GPU 1: First partition
        # First layer split by columns (output dimension split)
        self.c_fc_1 = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.gelu_1 = nn.GELU()
        # Second layer split by rows (input dimension split)
        self.c_proj_1 = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)
        
        # GPU 2: Second partition
        self.c_fc_2 = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.gelu_2 = nn.GELU()
        self.c_proj_2 = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # ========== GPU 1: First partition ==========
        # First linear + GeLU (no communication needed - identity operation f)
        y_1 = self.c_fc_1(x)
        y_1 = self.gelu_1(y_1)
        # Second linear (produces partial result)
        z_1 = self.c_proj_1(y_1)
        
        # ========== GPU 2: Second partition ==========
        # First linear + GeLU (no communication needed - identity operation f)
        y_2 = self.c_fc_2(x)
        y_2 = self.gelu_2(y_2)
        # Second linear (produces partial result)
        z_2 = self.c_proj_2(y_2)
        
        # ========== ALL-REDUCE: Sum results from both GPUs ==========
        z = z_1 + z_2
        
        # Apply dropout after all-reduce
        z = self.dropout(z)
        
        return z


class Block(nn.Module):
    """
    Transformer block with tensor parallel attention and MLP.
    
    Synchronization points per block:
    - 1 all-reduce after attention output projection
    - 1 all-reduce after MLP second linear layer
    Total: 2 all-reduces per block
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = TensorParallelCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = TensorParallelMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 1 all-reduce inside attn
        x = x + self.mlp(self.ln_2(x))   # 1 all-reduce inside mlp
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('c_proj_1.weight') or pn.endswith('c_proj_2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        print(f"Total synchronization points per forward pass: {config.n_layer * 2}")
        print(f"  - {config.n_layer} all-reduces in attention layers")
        print(f"  - {config.n_layer} all-reduces in MLP layers")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
