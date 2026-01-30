#!/usr/bin/env python3
"""Debug Marlin SM70 by testing with known simple values."""

import torch
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    awq_marlin_quantize,
)

torch.manual_seed(42)

# Match the failing test case exactly
size_m, size_k, size_n = 1, 128, 256
group_size = 128  # 
dtype = torch.float16
b_type = scalar_types.uint4

print(f"Testing Marlin SM70 with M={size_m}, K={size_k}, N={size_n}")
print(f"Group size: {group_size}, dtype: {dtype}, b_type: {b_type}")

# Create simple input: all ones
a_input = torch.ones((size_m, size_k), dtype=dtype, device="cuda")
# Create simple weights: all ones (will be quantized)
b_weight = torch.ones((size_k, size_n), dtype=dtype, device="cuda")

# Quantize with AWQ style (has zero point)
w_ref, marlin_q_w, marlin_s, marlin_zp = awq_marlin_quantize(
    b_weight, b_type, group_size, input_dtype=dtype
)

print(f"w_ref shape: {w_ref.shape}, marlin_q_w shape: {marlin_q_w.shape}")
print(f"marlin_s shape: {marlin_s.shape}, marlin_zp shape: {marlin_zp.shape}")
print(f"w_ref mean: {w_ref.mean().item():.4f}")

workspace = marlin_make_workspace_new(a_input.device)
output = torch.empty((size_m, size_n), dtype=dtype, device=a_input.device)

# Run Marlin GEMM
output = ops.marlin_gemm(
    a_input,
    output,
    marlin_q_w,
    None,  # bias
    marlin_s,
    None,  # a_scales
    None,  # s2
    marlin_zp,
    None,  # g_idx
    None,  # sort_indices
    workspace,
    b_type,
    a_input.shape[0],
    b_weight.shape[1],
    a_input.shape[1],
    is_k_full=True,
    use_atomic_add=False,
    use_fp32_reduce=True,
    is_zp_float=False,
)

# Reference computation
output_ref = torch.matmul(a_input.float(), w_ref.float()).to(dtype)

print(f"\nResults:")
print(f"Output mean: {output.mean().item():.4f}")
print(f"Reference mean: {output_ref.mean().item():.4f}")
print(f"Expected (K * w_ref_mean): {size_k * w_ref.mean().item():.4f}")

# Check diff
max_diff = (output - output_ref).abs().max().item()
mean_diff = (output - output_ref).abs().mean().item()
print(f"\nMax diff: {max_diff:.4f}")
print(f"Mean diff: {mean_diff:.4f}")
print(f"Relative max diff: {max_diff / output_ref.abs().mean().item():.4f}")

# Check if output is some fraction of expected
ratio = output.mean().item() / output_ref.mean().item()
print(f"\nRatio (output/ref): {ratio:.4f}")

# Print some actual values
print(f"\nFirst row of output: {output[0, :8].tolist()}")
print(f"First row of ref:    {output_ref[0, :8].tolist()}")
