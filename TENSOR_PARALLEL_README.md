# Tensor Parallel nanoGPT

This repository contains a simulated tensor-parallel implementation of GPT based on the Megatron-LM tensor parallelism approach.

## Overview

The goal of this implementation is to demonstrate how tensor parallelism can reduce the number of synchronization points compared to naive pipeline parallelism, while maintaining mathematical equivalence to the original model.

### Key Concepts

**Tensor Parallelism** (from Megatron-LM) splits individual layer computations across multiple GPUs:
- **MLP layers**: First linear layer split by columns, second by rows
- **Attention layers**: Q/K/V projections split by columns (attention heads), output projection split by rows
- **Synchronization**: All-reduce operation after each layer's final computation

### Synchronization Points

For a model with `N` transformer blocks:
- **Total synchronization points**: `2N` (vs. `~240` for models like Llama-400B)
  - `N` all-reduces in attention layers
  - `N` all-reduces in MLP layers

## Files

- `model.py` - Original nanoGPT model (unchanged)
- `tensor_parallel_model.py` - Tensor parallel version simulating 2-way parallelism
- `test_tensor_parallel.py` - Test script to verify mathematical equivalence
- `requirements.txt` - Python dependencies

## Implementation Details

### Tensor Parallel MLP

```
Standard MLP: Z = GeLU(X·A)·B

Tensor Parallel (2 GPUs):
GPU 1: Y₁ = GeLU(X·A₁), Z₁ = Y₁·B₁
GPU 2: Y₂ = GeLU(X·A₂), Z₂ = Y₂·B₂
All-Reduce: Z = Z₁ + Z₂
```

- **A** split by columns: `A = [A₁, A₂]`
- **B** split by rows: `B = [B₁; B₂]`
- **Synchronization**: 1 all-reduce after second linear layer

### Tensor Parallel Attention

```
Standard Attention: Z = Attention(X·Wq, X·Wk, X·Wv)·Wo

Tensor Parallel (2 GPUs):
GPU 1: Q₁ = X·Wq₁, K₁ = X·Wk₁, V₁ = X·Wv₁
       Head₁ = Attention(Q₁, K₁, V₁)
       Z₁ = Head₁·Wo₁
GPU 2: Q₂ = X·Wq₂, K₂ = X·Wk₂, V₂ = X·Wv₂
       Head₂ = Attention(Q₂, K₂, V₂)
       Z₂ = Head₂·Wo₂
All-Reduce: Z = Z₁ + Z₂
```

- **Wq, Wk, Wv** split by columns (split attention heads across GPUs)
- **Wo** split by rows
- **Synchronization**: 1 all-reduce after output projection

## Installation and Usage

### 1. Clone and setup

```bash
cd nanoGPT
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the test

```bash
python test_tensor_parallel.py
```

This will:
1. Create both original and tensor parallel models
2. Copy weights from original to tensor parallel version
3. Run forward passes and compare outputs
4. Verify mathematical equivalence (should be within 1e-4 tolerance)
5. Test text generation

### Expected Output

```
================================================================================
Testing Tensor Parallel Model Equivalence
================================================================================

Test Configuration:
  Layers: 2
  Heads: 4
  Embedding dimension: 128
  Block size: 128

Creating original model...
number of parameters: 0.43M
Creating tensor parallel model...
number of parameters: 0.43M
Total synchronization points per forward pass: 4
  - 2 all-reduces in attention layers
  - 2 all-reduces in MLP layers

✓ SUCCESS: Models are equivalent (max diff < 0.0001)
```

## Architecture Modifications for Fewer Communications

The tensor parallel implementation demonstrates a key insight: **By splitting computations within layers rather than across layers, we reduce synchronization points from O(sequence_length × layers) to O(layers).**

### Comparison: Pipeline vs. Tensor Parallelism

| Approach | Synchronization Points | Latency Impact |
|----------|------------------------|----------------|
| Naive Pipeline | ~1 per token per layer | Very High |
| Tensor Parallel | 2 per layer (total) | Moderate |
| Hybrid | Reduced further | Low |

### Key Benefits

1. **Fewer All-Reduces**: For a 12-layer model, only 24 synchronization points vs. hundreds in pipeline parallelism
2. **Parallel Computation**: Each layer's computation happens simultaneously across GPUs
3. **Load Balancing**: Work is evenly distributed across devices

### Potential Further Optimizations

1. **Sequence Parallelism**: Split along sequence dimension to reduce activation memory
2. **Fused Operations**: Combine multiple operations to reduce intermediate synchronizations
3. **Hybrid Parallelism**: Combine tensor, pipeline, and data parallelism for large models
4. **Expert Parallelism**: For MoE models, assign different experts to different GPUs

## References

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Original implementation by Andrej Karpathy

## Technical Notes

### Simulation vs. Actual Distribution

This implementation simulates tensor parallelism within a single process:
- Real implementation would use `torch.distributed` with NCCL backend
- All-reduce would be actual network communication
- Each "GPU" would have its own memory space
- Our simulation runs sequentially but demonstrates the mathematical structure

### Weight Splitting Strategy

The key insight is matching input/output dimensions across split layers:

```
First layer output dimension split: [n_embd] → [n_embd/2, n_embd/2]
Second layer input dimension split: [n_embd] → [n_embd/2, n_embd/2]
```

This ensures partial results from each GPU combine correctly via all-reduce.

## License

MIT (following nanoGPT's license)
