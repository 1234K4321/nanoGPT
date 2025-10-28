#!/bin/bash

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         Training Dashboard - SPD 12-Sync                       ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Training status
    if pgrep -f "train_reduced_sync.py.*spd" > /dev/null; then
        echo "✅ Status: RUNNING"
    else
        echo "⏸️  Status: STOPPED"
    fi
    echo ""
    
    # Checkpoint info
    if [ -f out-spd-12sync/ckpt.pt ]; then
        echo "📊 Training Progress:"
        python3 -c "
import torch
ckpt = torch.load('out-spd-12sync/ckpt.pt', map_location='cpu')
iter_num = ckpt['iter_num']
max_iters = ckpt['config']['max_iters']
best_val = ckpt['best_val_loss']
progress = (iter_num / max_iters) * 100

print(f'   Iteration: {iter_num:,} / {max_iters:,}')
print(f'   Progress: {progress:.1f}%')
print(f'   Best Val Loss: {best_val:.4f}')
print(f'   Target: 3.28')
print(f'   Gap: {best_val - 3.28:+.4f}')

# Estimate time remaining
if iter_num > 0:
    import os
    from datetime import datetime
    ckpt_time = os.path.getmtime('out-spd-12sync/ckpt.pt')
    elapsed = ckpt_time - os.path.getctime('out-spd-12sync/ckpt.pt')
    time_per_iter = elapsed / iter_num if iter_num > 0 else 0
    remaining_iters = max_iters - iter_num
    remaining_secs = remaining_iters * time_per_iter
    hours = int(remaining_secs // 3600)
    mins = int((remaining_secs % 3600) // 60)
    print(f'   Est. Time Remaining: ~{hours}h {mins}m')
" 2>/dev/null || echo "   Error reading checkpoint"
    else
        echo "📊 No checkpoint yet"
    fi
    
    echo ""
    echo "🖥️  GPU Usage (Top 8):"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -8 | awk '{printf "   GPU %s: %s%% | %s/%s MB\n", $1, $2, $3, $4}' || echo "   Unable to query GPUs"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Press Ctrl+C to stop monitoring | Updates every 10s"
    
    sleep 10
done
