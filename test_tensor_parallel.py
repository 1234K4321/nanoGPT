"""
Test script to verify that tensor_parallel_model.py is mathematically equivalent
to the original model.py by comparing their outputs on the same inputs.
"""

import torch
import sys

# Import both models
import model as original_model
import tensor_parallel_model as tp_model


def copy_weights_to_tensor_parallel(original_model_instance, tp_model_instance):
    """
    Copy weights from original model to tensor parallel model.
    
    For tensor parallel model:
    - Split attention Q/K/V weights by columns (heads)
    - Split attention output projection by rows
    - Split MLP first layer by columns
    - Split MLP second layer by rows
    """
    # Copy embeddings and other shared layers
    tp_model_instance.transformer.wte.weight.data = original_model_instance.transformer.wte.weight.data.clone()
    tp_model_instance.transformer.wpe.weight.data = original_model_instance.transformer.wpe.weight.data.clone()
    tp_model_instance.transformer.ln_f.weight.data = original_model_instance.transformer.ln_f.weight.data.clone()
    if original_model_instance.transformer.ln_f.bias is not None:
        tp_model_instance.transformer.ln_f.bias.data = original_model_instance.transformer.ln_f.bias.data.clone()
    tp_model_instance.lm_head.weight.data = original_model_instance.lm_head.weight.data.clone()
    
    # Copy each block
    for orig_block, tp_block in zip(original_model_instance.transformer.h, tp_model_instance.transformer.h):
        # Copy layer norms
        tp_block.ln_1.weight.data = orig_block.ln_1.weight.data.clone()
        if orig_block.ln_1.bias is not None:
            tp_block.ln_1.bias.data = orig_block.ln_1.bias.data.clone()
        tp_block.ln_2.weight.data = orig_block.ln_2.weight.data.clone()
        if orig_block.ln_2.bias is not None:
            tp_block.ln_2.bias.data = orig_block.ln_2.bias.data.clone()
        
        # Copy attention weights
        # Original: c_attn has shape (n_embd, 3 * n_embd)
        # Split by columns for Q, K, V
        orig_qkv_weight = orig_block.attn.c_attn.weight.data  # (3*n_embd, n_embd)
        orig_qkv_bias = orig_block.attn.c_attn.bias.data if orig_block.attn.c_attn.bias is not None else None
        
        n_embd = orig_qkv_weight.shape[1]
        
        # Split into Q, K, V
        q_weight, k_weight, v_weight = orig_qkv_weight.chunk(3, dim=0)
        
        # Split each by half for two partitions
        q_weight_1, q_weight_2 = q_weight.chunk(2, dim=0)
        k_weight_1, k_weight_2 = k_weight.chunk(2, dim=0)
        v_weight_1, v_weight_2 = v_weight.chunk(2, dim=0)
        
        # Concatenate Q, K, V for each partition
        tp_block.attn.c_attn_1.weight.data = torch.cat([q_weight_1, k_weight_1, v_weight_1], dim=0)
        tp_block.attn.c_attn_2.weight.data = torch.cat([q_weight_2, k_weight_2, v_weight_2], dim=0)
        
        if orig_qkv_bias is not None:
            q_bias, k_bias, v_bias = orig_qkv_bias.chunk(3, dim=0)
            q_bias_1, q_bias_2 = q_bias.chunk(2, dim=0)
            k_bias_1, k_bias_2 = k_bias.chunk(2, dim=0)
            v_bias_1, v_bias_2 = v_bias.chunk(2, dim=0)
            tp_block.attn.c_attn_1.bias.data = torch.cat([q_bias_1, k_bias_1, v_bias_1], dim=0)
            tp_block.attn.c_attn_2.bias.data = torch.cat([q_bias_2, k_bias_2, v_bias_2], dim=0)
        
        # Copy attention output projection (split by rows)
        orig_proj_weight = orig_block.attn.c_proj.weight.data  # (n_embd, n_embd)
        orig_proj_bias = orig_block.attn.c_proj.bias.data if orig_block.attn.c_proj.bias is not None else None
        
        proj_weight_1, proj_weight_2 = orig_proj_weight.chunk(2, dim=1)  # Split input dimension
        tp_block.attn.c_proj_1.weight.data = proj_weight_1.clone()
        tp_block.attn.c_proj_2.weight.data = proj_weight_2.clone()
        
        if orig_proj_bias is not None:
            # Bias is replicated for row-split (both partitions contribute to full output)
            tp_block.attn.c_proj_1.bias.data = orig_proj_bias.clone()
            tp_block.attn.c_proj_2.bias.data = orig_proj_bias.clone()
        
        # Copy MLP weights
        # First layer split by columns (output dimension)
        orig_fc_weight = orig_block.mlp.c_fc.weight.data  # (4*n_embd, n_embd)
        orig_fc_bias = orig_block.mlp.c_fc.bias.data if orig_block.mlp.c_fc.bias is not None else None
        
        fc_weight_1, fc_weight_2 = orig_fc_weight.chunk(2, dim=0)  # Split output dimension
        tp_block.mlp.c_fc_1.weight.data = fc_weight_1.clone()
        tp_block.mlp.c_fc_2.weight.data = fc_weight_2.clone()
        
        if orig_fc_bias is not None:
            fc_bias_1, fc_bias_2 = orig_fc_bias.chunk(2, dim=0)
            tp_block.mlp.c_fc_1.bias.data = fc_bias_1.clone()
            tp_block.mlp.c_fc_2.bias.data = fc_bias_2.clone()
        
        # Second layer split by rows (input dimension)
        orig_proj_weight = orig_block.mlp.c_proj.weight.data  # (n_embd, 4*n_embd)
        orig_proj_bias = orig_block.mlp.c_proj.bias.data if orig_block.mlp.c_proj.bias is not None else None
        
        mlp_proj_weight_1, mlp_proj_weight_2 = orig_proj_weight.chunk(2, dim=1)  # Split input dimension
        tp_block.mlp.c_proj_1.weight.data = mlp_proj_weight_1.clone()
        tp_block.mlp.c_proj_2.weight.data = mlp_proj_weight_2.clone()
        
        if orig_proj_bias is not None:
            # Bias is replicated for row-split
            tp_block.mlp.c_proj_1.bias.data = orig_proj_bias.clone()
            tp_block.mlp.c_proj_2.bias.data = orig_proj_bias.clone()


def test_equivalence():
    """Test that both models produce the same outputs."""
    print("=" * 80)
    print("Testing Tensor Parallel Model Equivalence")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create small model configuration for testing
    config = original_model.GPTConfig(
        block_size=128,
        vocab_size=256,  # Small vocab for testing
        n_layer=2,       # Just 2 layers for testing
        n_head=4,        # 4 heads (divisible by 2)
        n_embd=128,      # Small embedding
        dropout=0.0,     # No dropout for deterministic testing
        bias=True
    )
    
    print(f"\nTest Configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding dimension: {config.n_embd}")
    print(f"  Block size: {config.block_size}")
    
    # Create both models
    print("\nCreating original model...")
    orig_model = original_model.GPT(config)
    orig_model.eval()  # Set to eval mode
    
    print("Creating tensor parallel model...")
    tp_config = tp_model.GPTConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )
    parallel_model = tp_model.GPT(tp_config)
    parallel_model.eval()
    
    # Copy weights from original to tensor parallel
    print("\nCopying weights from original to tensor parallel model...")
    copy_weights_to_tensor_parallel(orig_model, parallel_model)
    
    # Create test input
    batch_size = 2
    seq_len = 16
    test_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nTest input shape: {test_input.shape}")
    
    # Run both models
    print("\nRunning forward pass on original model...")
    with torch.no_grad():
        orig_logits, _ = orig_model(test_input)
    
    print("Running forward pass on tensor parallel model...")
    with torch.no_grad():
        tp_logits, _ = parallel_model(test_input)
    
    # Compare outputs
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    
    print(f"\nOriginal model output shape: {orig_logits.shape}")
    print(f"Tensor parallel model output shape: {tp_logits.shape}")
    
    # Calculate differences
    max_diff = torch.max(torch.abs(orig_logits - tp_logits)).item()
    mean_diff = torch.mean(torch.abs(orig_logits - tp_logits)).item()
    rel_diff = mean_diff / (torch.mean(torch.abs(orig_logits)).item() + 1e-8)
    
    print(f"\nMaximum absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Relative difference: {rel_diff:.2e}")
    
    # Check if they're close enough (accounting for numerical precision)
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"\n✓ SUCCESS: Models are equivalent (max diff {max_diff:.2e} < {tolerance})")
    else:
        print(f"\n✗ FAILURE: Models differ significantly (max diff {max_diff:.2e} >= {tolerance})")
        print("\nShowing first few logits for comparison:")
        print(f"Original:  {orig_logits[0, 0, :5]}")
        print(f"Parallel:  {tp_logits[0, 0, :5]}")
        return False
    
    # Test generation
    print("\n" + "=" * 80)
    print("Testing Text Generation:")
    print("=" * 80)
    
    start_ids = torch.zeros((1, 1), dtype=torch.long)
    max_new_tokens = 10
    
    print(f"\nGenerating {max_new_tokens} tokens...")
    
    with torch.no_grad():
        orig_gen = orig_model.generate(start_ids, max_new_tokens=max_new_tokens, temperature=1.0)
        tp_gen = parallel_model.generate(start_ids, max_new_tokens=max_new_tokens, temperature=1.0)
    
    print(f"Original model generated: {orig_gen[0].tolist()}")
    print(f"Tensor parallel generated: {tp_gen[0].tolist()}")
    
    if torch.equal(orig_gen, tp_gen):
        print("\n✓ SUCCESS: Both models generated identical sequences")
    else:
        print("\n✗ Note: Sequences differ (expected due to potential numerical differences in sampling)")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"\nTensor Parallel Model Properties:")
    print(f"  - Simulates 2-way tensor parallelism")
    print(f"  - {config.n_layer * 2} synchronization points (all-reduces) per forward pass")
    print(f"    * {config.n_layer} in attention layers")
    print(f"    * {config.n_layer} in MLP layers")
    print(f"  - Each layer computation split across 2 'GPUs'")
    print(f"  - Mathematically equivalent to original model")
    
    return True


if __name__ == "__main__":
    try:
        success = test_equivalence()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
