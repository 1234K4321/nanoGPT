"""
Quick test script to verify all model architectures load correctly
and have the right parameter count before starting long training runs.
"""

import torch
from reduced_sync_model import GPTConfig, GPT

def test_model(sync_strategy, layers_per_group=None, attn_sync_layers=None, name=""):
    """Test a model configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    config = GPTConfig(
        block_size=256,  # Small for testing
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False,
        sync_strategy=sync_strategy,
        attn_sync_layers=attn_sync_layers,
        layers_per_group=layers_per_group,
    )
    
    try:
        model = GPT(config)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {n_params/1e6:.2f}M")
        print(f"  Sync points: {model.num_syncs}")
        
        # Test forward pass
        x = torch.randint(0, config.vocab_size, (2, 64))
        with torch.no_grad():
            logits, loss = model(x, x)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("REDUCED SYNC MODEL VERIFICATION")
    print("="*60)
    print("\nTesting all model variants before training...")
    
    results = {}
    
    # Test SPD (12 syncs)
    results['SPD-12'] = test_model(
        sync_strategy='spd',
        name="SPD (12 syncs - 1 per block)"
    )
    
    # Test Grouped-6 (6 syncs)
    results['Grouped-6'] = test_model(
        sync_strategy='grouped',
        layers_per_group=2,
        name="Grouped (6 syncs - 2 layers per group)"
    )
    
    # Test Grouped-4 (4 syncs)
    results['Grouped-4'] = test_model(
        sync_strategy='grouped',
        layers_per_group=3,
        name="Grouped (4 syncs - 3 layers per group)"
    )
    
    # Test Grouped-3 (3 syncs)
    results['Grouped-3'] = test_model(
        sync_strategy='grouped',
        layers_per_group=4,
        name="Grouped (3 syncs - 4 layers per group)"
    )
    
    # Test Grouped-2 (2 syncs) - very aggressive
    results['Grouped-2'] = test_model(
        sync_strategy='grouped',
        layers_per_group=6,
        name="Grouped (2 syncs - 6 layers per group)"
    )
    
    # Test Selective (mixed strategy)
    results['Selective'] = test_model(
        sync_strategy='selective',
        attn_sync_layers=[0, 3, 6, 9],
        name="Selective (16 syncs - attention sync in 4 layers)"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All models verified successfully!")
        print(f"  Ready to start training experiments.")
    else:
        print(f"\n✗ Some models failed. Fix errors before training.")
        return 1
    
    print(f"\n{'='*60}")
    print("RECOMMENDED TRAINING ORDER:")
    print(f"{'='*60}")
    print("1. SPD-12 (baseline with reduced syncs)")
    print("2. Grouped-6 (50% reduction)")
    print("3. Grouped-4 (67% reduction)")
    print("4. Grouped-3 (75% reduction) - if Grouped-4 works well")
    print("\nUse commands from TRAINING_INSTRUCTIONS.md")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
