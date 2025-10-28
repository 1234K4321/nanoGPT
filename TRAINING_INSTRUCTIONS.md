# Training Instructions: Reduced Synchronization Experiments

## Objective
Find the minimum number of synchronization points that still achieves ~3.28 validation loss after 20k steps, matching the baseline 125M GPT performance.

## Setup (Run these commands on the 8-GPU machine)

### 1. Prepare the environment
```bash
cd /path/to/nanoGPT
source venv/bin/activate  # or create new venv if needed

# Install dependencies if not already done
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### 2. Prepare the dataset (OpenWebText)
```bash
# This downloads and tokenizes OpenWebText (~54GB download)
python data/openwebtext/prepare.py
```
This will create `data/openwebtext/train.bin` and `data/openwebtext/val.bin`.

**Important**: This step takes ~1-2 hours. Do this FIRST before starting experiments.

---

## Experiment Plan

We'll train 4-5 models with different sync strategies to find the optimal trade-off:

| Experiment | Sync Strategy | # Syncs | Config File | Expected Time |
|------------|---------------|---------|-------------|---------------|
| Baseline   | Standard TP   | 24      | N/A (reference) | ~45 min |
| Exp 1      | SPD (1/block) | 12      | `train_spd_12sync.py` | ~45 min |
| Exp 2      | Grouped (2)   | 6       | `train_grouped_6sync.py` | ~45 min |
| Exp 3      | Grouped (3)   | 4       | `train_grouped_4sync.py` | ~45 min |
| Exp 4      | Grouped (4)   | 3       | (create if needed) | ~45 min |

### Baseline Reference
The standard nanoGPT 125M model achieves **val loss ~3.28** at 20k steps with 24 sync points (tensor parallel inference).

---

## Running Experiments

### General Command Structure
```bash
torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/<config_file>.py
```

### Experiment 1: SPD - 12 Syncs (1 per block)
```bash
torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/train_spd_12sync.py
```
- **Expectation**: Should perform close to baseline (~3.28 val loss)
- **Why**: Removes only attention syncs; still synchronizes after each block
- **Output**: `out-spd-12sync/`

### Experiment 2: Grouped - 6 Syncs (2 layers per group)
```bash
torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/train_grouped_6sync.py
```
- **Expectation**: Likely 3.3-3.5 val loss (slight degradation acceptable)
- **Why**: Groups 2 layers together, reducing syncs by 50%
- **Output**: `out-grouped-6sync/`

### Experiment 3: Grouped - 4 Syncs (3 layers per group)
```bash
torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/train_grouped_4sync.py
```
- **Expectation**: May see 3.5-3.8 val loss (monitor closely)
- **Why**: Aggressive grouping - 67% reduction in syncs
- **Output**: `out-grouped-4sync/`

### Experiment 4 (Optional): Grouped - 3 Syncs (4 layers per group)
If Exp 3 still works well, create config for 3 syncs:
```bash
# Create config/train_grouped_3sync.py with layers_per_group=4
torchrun --standalone --nproc_per_node=8 train_reduced_sync.py config/train_grouped_3sync.py
```

---

## Quick Start Strategy (Optimal Path)

If you want to find the answer quickly:

### Step 1: Run experiments in parallel
You can run multiple experiments simultaneously on different subsets of GPUs:

```bash
# Terminal 1: SPD (12 syncs) on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_reduced_sync.py config/train_spd_12sync.py

# Terminal 2: Grouped-6 on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train_reduced_sync.py config/train_grouped_6sync.py
```

**Note**: Adjust `gradient_accumulation_steps` in configs for 4 GPUs:
- Original: `gradient_accumulation_steps = 5 * 8 = 40`
- For 4 GPUs: Set to `5 * 8 = 40` (will be divided by 4 automatically in training)

### Step 2: Early stopping decision (at 5k steps)
Check validation loss at 5000 steps:
```bash
# Check logs
tail -f out-spd-12sync/log.txt
tail -f out-grouped-6sync/log.txt
```

If at 5k steps:
- SPD-12: val loss ~3.5 → Continue
- Grouped-6: val loss ~3.6 → Continue  
- If Grouped-6 > 3.8 → Stop and focus on SPD-12

### Step 3: Binary search on sync count
Based on early results, run the promising middle ground:
- If both SPD and Grouped-6 work well → Test Grouped-4
- If only SPD works → Test something between 12 and 6 (e.g., selective with 9 syncs)

---

## Monitoring Progress

### Check training progress
```bash
# Watch real-time logs
tail -f out-spd-12sync/log.txt

# Check validation losses
grep "val loss" out-spd-12sync/log.txt
```

### Key metrics to track
- **Iteration 5000**: Early indicator (~25% done)
- **Iteration 10000**: Mid-point check (~50% done)
- **Iteration 15000**: Late stage (~75% done)
- **Iteration 20000**: Final result

### Expected timeline
- Each experiment: ~45 minutes on 8× H100 GPUs
- With parallel execution: Can test 2 configs simultaneously in ~45 minutes
- Total for all 3 main experiments (sequential): ~2.5 hours
- Total with parallel execution: ~1.5 hours

---

## Results Analysis

### Success Criteria
Find the minimum sync count where `val_loss @ 20k ≤ 3.35` (within 2% of baseline 3.28)

### Create results summary
```bash
# Extract final validation losses
echo "=== Final Results (20k steps) ===" > results_summary.txt
for dir in out-spd-12sync out-grouped-6sync out-grouped-4sync; do
    echo -n "$dir: " >> results_summary.txt
    grep "step 20000" $dir/log.txt >> results_summary.txt
done
cat results_summary.txt
```

### Example expected results
```
Baseline (24 syncs):   val_loss = 3.28
SPD (12 syncs):        val_loss = 3.29 ✓ (within 0.3% of baseline)
Grouped-6 (6 syncs):   val_loss = 3.35 ✓ (within 2% of baseline) 
Grouped-4 (4 syncs):   val_loss = 3.52 ✗ (7% degradation)
```
**Conclusion**: Can reduce to 6 syncs (75% reduction!) with minimal performance loss.

---

## Troubleshooting

### Out of Memory
If you get OOM errors:
```python
# In config file, reduce batch_size
batch_size = 8  # instead of 12
gradient_accumulation_steps = 7 * 8  # adjust to maintain same effective batch
```

### Training too slow
```python
compile = False  # Disable torch.compile for faster startup (slightly slower training)
```

### Check model parameters
```bash
python -c "from reduced_sync_model import GPTConfig, GPT; 
config = GPTConfig(sync_strategy='spd', n_layer=12, n_head=12, n_embd=768);
model = GPT(config); 
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')"
```
Should print ~124-125M parameters for all variants.

---

## Advanced: Creating Additional Configurations

### For 3 syncs (4 layers per group)
```bash
cat > config/train_grouped_3sync.py << 'EOF'
# 3 syncs: 12 layers / 4 = 3 groups
sync_strategy = 'grouped'
attn_sync_layers = None
layers_per_group = 4
out_dir = 'out-grouped-3sync'
# ... (copy rest from train_grouped_6sync.py)
EOF
```

### For 2 syncs (6 layers per group)
```bash
cat > config/train_grouped_2sync.py << 'EOF'
# 2 syncs: 12 layers / 6 = 2 groups
sync_strategy = 'grouped'
attn_sync_layers = None
layers_per_group = 6
out_dir = 'out-grouped-2sync'
# ... (copy rest from train_grouped_6sync.py)
EOF
```

---

## Final Recommendation

**Priority order for experiments:**

1. **Run SPD (12 syncs) first** - This should definitely work and gives you a solid baseline with 50% reduction.

2. **Run Grouped-6 (6 syncs)** - If this works, you've achieved 75% reduction, which is excellent!

3. **Run Grouped-4 (4 syncs)** - Only if Grouped-6 shows promising results (< 3.4 val loss at 10k steps).

4. **Binary search between working and non-working** - If Grouped-6 works but Grouped-4 doesn't, try Grouped-5 (not directly possible, but you could use selective strategy).

**Most likely outcome**: You'll find that **6-9 synchronization points** provide the best trade-off between communication efficiency and model performance, achieving a **75-83% reduction** in sync points while maintaining val loss within 2-3% of baseline.

Good luck with the experiments! 🚀
