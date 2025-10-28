"""
Reduced Synchronization Tensor Parallel GPT Models.
Based on SPD (Sync-Point Drop) paper and tensor parallelism experiments.

This file contains multiple architecture variants that reduce the number of 
synchronization points while maintaining similar parameter counts.

For inference with tensor parallelism, fewer sync points = lower latency.
Goal: Train from scratch and find which architectures achieve ~3.28 val loss at 20k steps.

Variants:
- SPD_12sync: Remove attention syncs, keep MLP syncs (12 syncs total)
- SPD_6sync: Remove syncs from half the layers (6 syncs total)
- SPD_Grouped: Group layers to share syncs
- SPD_Adaptive: Smart placement of syncs
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


# ============================================================================
# VARIANT 1: SPD-Style (1 sync per block) - 12 syncs total
# ============================================================================

class SPD_CausalSelfAttention(nn.Module):
    """
    SPD-style attention: No sync after attention.
    Each partition's attention output goes directly to its MLP.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % 2 == 0, "Number of heads must be divisible by 2"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_head_per_partition = config.n_head // 2
        
        # Two partitions for Q,K,V
        self.c_attn_1 = nn.Linear(config.n_embd, 3 * config.n_embd // 2, bias=config.bias)
        self.c_attn_2 = nn.Linear(config.n_embd, 3 * config.n_embd // 2, bias=config.bias)
        
        # NO output projection here - will be fused with MLP
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        
        # GPU 1: First half of heads
        qkv_1 = self.c_attn_1(x)
        q_1, k_1, v_1 = qkv_1.split(self.n_embd // 2, dim=2)
        k_1 = k_1.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        q_1 = q_1.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        v_1 = v_1.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        
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
        
        # GPU 2: Second half of heads
        qkv_2 = self.c_attn_2(x)
        q_2, k_2, v_2 = qkv_2.split(self.n_embd // 2, dim=2)
        k_2 = k_2.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        q_2 = q_2.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        v_2 = v_2.view(B, T, self.n_head_per_partition, C // self.n_head).transpose(1, 2)
        
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
        
        # NO SYNC - return both local outputs
        return y_1, y_2


class SPD_MLP(nn.Module):
    """
    SPD-style MLP: Takes local attention output, produces local MLP output.
    Single sync at the end combines all outputs.
    """

    def __init__(self, config):
        super().__init__()
        
        # Attention output projection + MLP first layer (fused)
        # Each partition: (n_embd//2) -> n_embd -> 2*n_embd
        self.c_attn_proj_1 = nn.Linear(config.n_embd // 2, config.n_embd, bias=config.bias)
        self.c_fc_1 = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.gelu_1 = nn.GELU()
        self.c_proj_1 = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.c_attn_proj_2 = nn.Linear(config.n_embd // 2, config.n_embd, bias=config.bias)
        self.c_fc_2 = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.gelu_2 = nn.GELU()
        self.c_proj_2 = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, y_1, y_2):
        # GPU 1: project local attention + MLP
        h_1 = self.c_attn_proj_1(y_1)
        h_1 = self.c_fc_1(h_1)
        h_1 = self.gelu_1(h_1)
        z_1 = self.c_proj_1(h_1)
        
        # GPU 2: project local attention + MLP
        h_2 = self.c_attn_proj_2(y_2)
        h_2 = self.c_fc_2(h_2)
        h_2 = self.gelu_2(h_2)
        z_2 = self.c_proj_2(h_2)
        
        # SINGLE SYNC: combine both attention and MLP outputs
        z = z_1 + z_2
        z = self.dropout(z)
        
        return z


class SPD_Block(nn.Module):
    """
    SPD Block: 1 sync point per block (at the very end).
    Attention outputs go directly to their local MLPs.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SPD_CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = SPD_MLP(config)

    def forward(self, x):
        # Attention (no sync)
        attn_in = self.ln_1(x)
        y_1, y_2 = self.attn(attn_in)
        
        # MLP takes local attention outputs (with residual)
        # Note: In SPD paper, they do X + Y_i as input to MLP_i
        # Here we simplify by passing y_i directly and handling residual at the end
        mlp_out = self.mlp(y_1, y_2)  # This includes the sync
        
        # Residual connection
        x = x + mlp_out
        
        return x


# ============================================================================
# VARIANT 2: Selective Sync (6-12 syncs, configurable)
# ============================================================================

class SelectiveSyncBlock(nn.Module):
    """
    Block that can optionally have sync after attention.
    Used to experiment with different sync patterns.
    """

    def __init__(self, config, has_attn_sync=True):
        super().__init__()
        self.has_attn_sync = has_attn_sync
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        
        if has_attn_sync:
            # Standard tensor parallel attention with sync
            from tensor_parallel_model import TensorParallelCausalSelfAttention, TensorParallelMLP
            self.attn = TensorParallelCausalSelfAttention(config)
            self.mlp = TensorParallelMLP(config)
        else:
            # SPD-style: no attention sync
            self.attn = SPD_CausalSelfAttention(config)
            self.mlp = SPD_MLP(config)

    def forward(self, x):
        if self.has_attn_sync:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            attn_in = self.ln_1(x)
            y_1, y_2 = self.attn(attn_in)
            mlp_out = self.mlp(y_1, y_2)
            x = x + mlp_out
        return x


# ============================================================================
# VARIANT 3: Grouped Layers (fewer syncs via layer grouping)
# ============================================================================

class GroupedBlock(nn.Module):
    """
    Group multiple transformer layers together with single sync at the end.
    Each group has N layers that share local computations.
    """

    def __init__(self, config, num_layers_in_group=2):
        super().__init__()
        self.num_layers = num_layers_in_group
        
        # Create sub-blocks without individual syncs
        self.layers_1 = nn.ModuleList([
            self._make_local_layer(config) for _ in range(num_layers_in_group)
        ])
        self.layers_2 = nn.ModuleList([
            self._make_local_layer(config) for _ in range(num_layers_in_group)
        ])
        
    def _make_local_layer(self, config):
        """Single layer without sync"""
        return nn.ModuleDict({
            'ln_1': LayerNorm(config.n_embd, bias=config.bias),
            'attn': SPD_CausalSelfAttention(config),
            'ln_2': LayerNorm(config.n_embd, bias=config.bias),
            'mlp_fc': nn.Linear(config.n_embd // 2, 2 * config.n_embd, bias=config.bias),
            'mlp_proj': nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias),
            'gelu': nn.GELU(),
            'dropout': nn.Dropout(config.dropout),
        })
    
    def forward(self, x):
        # Process GPU 1's path through all layers
        x_1 = x
        for layer in self.layers_1:
            attn_in = layer['ln_1'](x_1)
            y_1, _ = layer['attn'](attn_in)
            mlp_in = y_1
            mlp_out = layer['mlp_fc'](mlp_in)
            mlp_out = layer['gelu'](mlp_out)
            mlp_out = layer['mlp_proj'](mlp_out)
            x_1 = x_1 + mlp_out
        
        # Process GPU 2's path through all layers
        x_2 = x
        for layer in self.layers_2:
            attn_in = layer['ln_1'](x_2)
            _, y_2 = layer['attn'](attn_in)
            mlp_in = y_2
            mlp_out = layer['mlp_fc'](mlp_in)
            mlp_out = layer['gelu'](mlp_out)
            mlp_out = layer['mlp_proj'](mlp_out)
            x_2 = x_2 + mlp_out
        
        # SINGLE SYNC for entire group
        return x_1 + x_2


# ============================================================================
# GPT Model Configurations
# ============================================================================

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # Architecture variant
    sync_strategy: str = 'spd'  # 'spd', 'selective', 'grouped'
    # For selective strategy: which layers have attention sync
    attn_sync_layers: list = None  # e.g., [0, 2, 4, 6, 8, 10] for 6 syncs
    # For grouped strategy: how many layers per group
    layers_per_group: int = 2


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
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Build blocks based on strategy
        if config.sync_strategy == 'spd':
            # SPD: 1 sync per block = 12 syncs total
            self.transformer.h = nn.ModuleList([SPD_Block(config) for _ in range(config.n_layer)])
            self.num_syncs = config.n_layer
            
        elif config.sync_strategy == 'selective':
            # Selective: some blocks have attention sync, some don't
            attn_sync_layers = config.attn_sync_layers or []
            self.transformer.h = nn.ModuleList([
                SelectiveSyncBlock(config, has_attn_sync=(i in attn_sync_layers))
                for i in range(config.n_layer)
            ])
            self.num_syncs = len(attn_sync_layers) * 2 + (config.n_layer - len(attn_sync_layers))
            
        elif config.sync_strategy == 'grouped':
            # Grouped: layers grouped together with single sync per group
            assert config.n_layer % config.layers_per_group == 0
            num_groups = config.n_layer // config.layers_per_group
            self.transformer.h = nn.ModuleList([
                GroupedBlock(config, config.layers_per_group) 
                for _ in range(num_groups)
            ])
            self.num_syncs = num_groups
            
        else:
            raise ValueError(f"Unknown sync strategy: {config.sync_strategy}")
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if 'c_proj' in pn and 'weight' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        print(f"Sync strategy: {config.sync_strategy}")
        print(f"Total synchronization points per forward pass: {self.num_syncs}")

    def get_num_params(self, non_embedding=True):
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
