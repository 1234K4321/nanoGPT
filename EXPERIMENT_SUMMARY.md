# Reduced Synchronization Architecture Experiments

## Overview

This directory contains implementations of GPT architectures with reduced synchronization points for more efficient tensor-parallel inference. Based on the SPD (Sync-Point Drop) paper, we explore how few synchronization points can achieve acceptable validation loss when training from scratch.

## Files Created

### Model Implementation
- **`reduced_sync_model.py`** - Main implementation with 3 architecture variants:
  - **SPD**: 1 sync per block (12 syncs total)
  - **Selective**: Configurable syncs per layer
  - **Grouped**: Multiple layers share single sync point

### Training
- **`train_reduced_sync.py`** - Modified training script for reduced sync models
- **`test_all_architectures.py`** - Verification script (all architectures ✓ PASS)

### Configurations (in `config/` directory)
- `train_spd_12sync.py` - SPD architecture (12 syncs)
- `train_grouped_6sync.py` - Grouped: 2 layers/group (6 syncs)
- `train_grouped_4sync.py` - Grouped: 3 layers/group (4 syncs)
- `train_grouped_3sync.py` - Grouped: 4 layers/group (3 syncs)
- `train_selective_9sync.py` - Selective sync placement

### Documentation
- **`TRAINING_INSTRUCTIONS.md`** - Complete guide for running experiments
- **`EXPERIMENT_SUMMARY.md`** - This file

## Architecture Variants Tested

| Variant | Sync Points | Reduction | Strategy |
|---------|-------------|-----------|----------|
| Baseline TP | 24 | 0% | Standard tensor parallelism |
| SPD-12 | 12 | 50% | Remove attention syncs |
| Grouped-6 | 6 | 75% | Group 2 layers together |
| Grouped-4 | 4 | 83% | Group 3 layers together |
| Grouped-3 | 3 | 87.5% | Group 4 layers together |
| Grouped-2 | 2 | 91.7% | Group 6 layers together |

## Key Implementation Details

### SPD Architecture (12 syncs)
```
For each block:
  1. Attention computes local outputs (no sync)
  2. MLP takes local attention outputs
  3. Single sync at end combines attention + MLP outputs
```

### Grouped Architecture (6/4/3 syncs)
```
For each group of N layers:
  GPU 1: Processes all N layers with local attention outputs
  GPU 2: Processes all N layers with local attention outputs
  Single sync at end of group
```

### Parameter Count
All variants maintain ~124M parameters (same as baseline GPT-2 small).

## Experimental Setup

### Hardware Requirements
- 8× H100 GPUs (SXM)
- ~80GB VRAM per GPU for batch training
- High-bandwidth interconnect (NVLink/InfiniBand)

### Training Configuration
- Dataset: OpenWebText (~54GB)
- Training steps: 20,000
- Batch size: 12 per GPU
- Gradient accumulation: 5 × 8 = 40
- Effective batch size: 12 × 8 × 40 = 3,840 tokens
- Learning rate: 6e-4 (cosine decay)
- Expected time: ~45 minutes per experiment

### Success Criteria
Achieve validation loss ≤ 3.35 at 20k steps (within 2% of baseline 3.28).

## Running Experiments

### Prerequisites
```bash
# Prepare dataset (do this FIRST, takes 1-2 hours)
python data/openwebtext/prepare.py

# Verify all architectures work
python test_all_architectures.py
```

### Run Experiments
```bash
# SPD (12 syncs) - Expected: ~3.29 val loss
torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/train_spd_12sync.py

# Grouped-6 (6 syncs) - Expected: ~3.35 val loss
torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/train_grouped_6sync.py

# Grouped-4 (4 syncs) - Expected: ~3.5 val loss
torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/train_grouped_4sync.py

# Grouped-3 (3 syncs) - Experimental
torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/train_grouped_3sync.py
```

### Parallel Execution (Optional)
Run multiple experiments on different GPU subsets:
```bash
# Terminal 1: SPD on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_reduced_sync.py config/train_spd_12sync.py

# Terminal 2: Grouped-6 on GPUs 4-7  
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
    train_reduced_sync.py config/train_grouped_6sync.py
```

## Monitoring Progress

```bash
# Watch training progress
tail -f out-spd-12sync/log.txt

# Check validation losses
grep "val loss" out-*/log.txt

# Compare results at 20k steps
for dir in out-*sync; do
    echo -n "$dir: "
    grep "step 20000" $dir/log.txt 2>/dev/null || echo "Not done yet"
done
```

## Expected Results

Based on the SPD paper findings (for pre-trained models) and our training from scratch:

### Conservative Estimate
- **12 syncs (SPD)**: ✓ Should achieve ~3.29 val loss (within 0.3%)
- **6 syncs (Grouped)**: ✓ Likely achieves ~3.35 val loss (within 2%)
- **4 syncs (Grouped)**: ? May achieve ~3.5 val loss (within 7%)
- **3 syncs (Grouped)**: ? Uncertain, possibly 3.6-3.8 val loss

### Optimistic Estimate (if SPD paper insights apply to training from scratch)
- **6 syncs**: ✓ Achieves ≤3.35 val loss
- **4 syncs**: ✓ Achieves ~3.4 val loss
- **3 syncs**: ? Achieves ~3.5 val loss

## Key Research Questions

1. **How does sync reduction affect training vs inference?**
   - SPD paper removes syncs from pre-trained models
   - We're training from scratch with reduced syncs
   - Does the model learn to compensate?

2. **What's the minimum viable sync count?**
   - Baseline: 24 syncs
   - Conservative goal: 6 syncs (75% reduction)
   - Stretch goal: 3-4 syncs (83-87% reduction)

3. **Does layer grouping work as well as SPD?**
   - SPD: Selective removal per layer
   - Grouping: More regular pattern
   - Trade-off: Simplicity vs optimal placement

## Theoretical Basis

### Why Reducing Syncs Works

1. **Local Information Sufficiency**: MLPs can partially process local attention outputs without full synchronization
2. **Residual Connections**: Carry forward information across layers even with delayed syncs
3. **Learned Compensation**: During training, model adapts to architecture constraints

### SPD Paper Insights
- Can drop ~50% of syncs in pre-trained models (zero-shot)
- Sensitive layers identified via perplexity drop
- Block-to-block distillation helps recovery
- Attention head grouping improves stability

### Our Approach (Training from Scratch)
- No pre-trained model to start from
- Model learns optimal weight distribution given sync constraints
- May naturally develop robustness to delayed synchronization
- Simpler than SPD's sensitivity-based approach

## Architecture Comparison

### Standard Tensor Parallelism (24 syncs)
```
Block 1: Attn (sync) → MLP (sync)
Block 2: Attn (sync) → MLP (sync)
...
Block 12: Attn (sync) → MLP (sync)
Total: 12 blocks × 2 syncs = 24 syncs
```

### SPD (12 syncs)
```
Block 1: Attn (no sync) → MLP (sync)
Block 2: Attn (no sync) → MLP (sync)
...
Block 12: Attn (no sync) → MLP (sync)
Total: 12 blocks × 1 sync = 12 syncs
```

### Grouped (6 syncs, 2 layers/group)
```
Group 1: [Block 1-2] → Sync
Group 2: [Block 3-4] → Sync
...
Group 6: [Block 11-12] → Sync
Total: 6 groups × 1 sync = 6 syncs
```

## Next Steps

1. **Run core experiments** (SPD-12, Grouped-6, Grouped-4)
2. **Analyze results** at 5k, 10k, 15k, 20k steps
3. **Determine minimum viable sync count**
4. **Optional**: Test intermediate configs based on results
5. **Document findings** for paper/report

## References

- **SPD Paper**: arXiv:2502.20727 - Sync-Point Drop for tensor parallelism
- **Megatron-LM**: arXiv:1909.08053 - Original tensor parallelism paper
- **nanoGPT**: github.com/karpathy/nanoGPT - Base implementation

## Success Criteria Summary

**Goal**: Find minimum syncs where `val_loss ≤ 3.35` at 20k steps

**Baseline**: 3.28 val loss with 24 syncs (standard TP)

**Targets**:
- 12 syncs: Should work (50% reduction)
- 6 syncs: Likely works (75% reduction) ← Primary target
- 4 syncs: Uncertain (83% reduction) ← Stretch goal
- 3 syncs: Very uncertain (87% reduction)

**Answer to research question**:
> "How few synchronization points can we achieve while maintaining ~3.28 validation loss?"

**Expected answer**: 6-9 synchronization points (75-83% reduction from baseline)

---

## Quick Start Checklist

- [ ] SSH into 8-GPU machine
- [ ] Activate venv: `source venv/bin/activate`
- [ ] Prepare dataset: `python data/openwebtext/prepare.py` (1-2 hours)
- [ ] Verify architectures: `python test_all_architectures.py` (should see all ✓ PASS)
- [ ] Run SPD experiment: `torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/train_spd_12sync.py`
- [ ] Run Grouped-6: `torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/train_grouped_6sync.py`
- [ ] Monitor: `tail -f out-*/log.txt`
- [ ] Collect results at 20k steps
- [ ] Analyze: Which has best loss vs sync trade-off?

Total time: ~2-4 hours (including dataset prep + training)
