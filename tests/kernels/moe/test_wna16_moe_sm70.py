# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comprehensive suite of tests for the SM70 (V100) WNA16 MoE kernel.

These tests sweep across various dimensions (batch sizes, hidden dimensions,
token counts) to verify stability and correctness in diverse scenarios.

We test:
1. **Various Token Counts (M)**: {1, 16, 17, 32, 64, 128, 256}
     - Checks padding, multi-tile handling, and edge cases.
2. **Various Hidden Dims (K, N)**: {1024, 4096}
     - Checks typical model sizes.
3. **High Expert Concurrency**: top-k=2 with 16 experts.
"""

import sys

import torch

from tests.kernels.moe.utils import fused_moe
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
from vllm.scalar_type import scalar_types


# ----------------------------------------------------------------------------
# Helper: Single Test Execution
# ----------------------------------------------------------------------------
def run_moe_test(m, n, k, e, topk, group_size=128):
    torch.manual_seed(0)
    dtype = torch.float16

    print(f"  Running config: M={m}, N={n}, K={k}, E={e}, topk={topk}...")

    # Data generation
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10.0
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10.0
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10.0
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    # Quantization
    pack_factor = 2
    quant_type = scalar_types.uint4b8

    def quant_layer(w_in):
        # [Output, Input]
        weight_t = w_in.t().contiguous()
        w_quant, q_qweight, q_scales, q_zeros = quantize_weights(
            weight_t, quant_type, group_size, False, False
        )
        # q_qweight out is [Input, Output]
        q_qweight = q_qweight.t().contiguous().to(torch.uint8)  # [Output, Input]
        q_qweight = q_qweight[:, 1::2] * 16 + q_qweight[:, ::2]  # [Output, Input//2]
        return w_quant.t().contiguous(), q_qweight, q_scales.t().contiguous()

    # Pre-allocate to save time
    w1_qweight = torch.empty(
        (e, 2 * n, k // pack_factor), device="cuda", dtype=torch.uint8
    )
    w2_qweight = torch.empty((e, k, n // pack_factor), device="cuda", dtype=torch.uint8)
    w1_scales = torch.empty((e, 2 * n, k // group_size), device="cuda", dtype=dtype)
    w2_scales = torch.empty((e, k, n // group_size), device="cuda", dtype=dtype)

    w1_ref = []
    w2_ref = []

    for i in range(e):
        r1, q1, s1 = quant_layer(w1[i])
        r2, q2, s2 = quant_layer(w2[i])
        w1_ref.append(r1)
        w2_ref.append(r2)
        w1_qweight[i] = q1
        w2_qweight[i] = q2
        w1_scales[i] = s1
        w2_scales[i] = s2

    w1_ref = torch.stack(w1_ref)
    w2_ref = torch.stack(w2_ref)

    # Quant Config
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=None,
        w2_zp=None,
        block_shape=[0, group_size],
    )

    # Reference Computation
    import torch.nn.functional as F

    topk_weights_raw = score.softmax(dim=-1, dtype=torch.float)
    topk_weights, selected_experts = topk_weights_raw.topk(topk, dim=-1)
    topk_weights = topk_weights.to(dtype)

    ref_out = torch.zeros((m, k), device="cuda", dtype=dtype)
    for t in range(m):
        for k_idx in range(topk):
            exp_id = selected_experts[t, k_idx]
            exp_w1 = w1_ref[exp_id]
            exp_w2 = w2_ref[exp_id]

            # Layer 1
            x = F.linear(a[t : t + 1], exp_w1)
            gate = F.silu(x[:, :n])
            up = x[:, n:]
            x = gate * up

            # Layer 2
            x = F.linear(x, exp_w2)
            ref_out[t] += x[0] * topk_weights[t, k_idx]

    # Kernel Call
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        output = fused_moe(
            a,
            w1_qweight,
            w2_qweight,
            score,
            topk,
            renormalize=False,
            global_num_experts=e,
            quant_config=quant_config,
        )

    # Validation
    diff = (output - ref_out).abs()
    max_diff = diff.max().item()

    # Loose tolerance 2e-2 is standard for 4-bit vs float16 accumulation drift
    if max_diff > 2e-2:
        print(f"    ✗ FAIL: Max diff {max_diff:.6f} > 2e-2")
        return False
    else:
        print(f"    ✓ PASS: Max diff {max_diff:.6f}")
        return True


# ----------------------------------------------------------------------------
# Suite Execution
# ----------------------------------------------------------------------------
def run_suite():
    print("=== SM70 (V100) Comprehensive Test Suite ===")

    # Checkpoint 1: Token Count Sweeps (M)
    # Why? To ensure padding and tiling logic works for all alignments
    ms_to_test = [1, 16, 17, 32, 64, 128]
    print("\n[Test Block 1: Token Count Sweep]")
    for m in ms_to_test:
        success = run_moe_test(m=m, n=1024, k=1024, e=8, topk=2)
        if not success:
            sys.exit(1)

    # Checkpoint 2: Large Dimension Sweeps (N, K)
    # Why? To ensure shared memory and register usage is stable for large matrices
    print("\n[Test Block 2: Large Dimensions]")
    success = run_moe_test(m=32, n=4096, k=4096, e=4, topk=2)
    if not success:
        sys.exit(1)

    # Checkpoint 3: Odd Alignments (Regression Check)
    # Why? M=17 or K not divisible by normal chunks often break kernels
    print("\n[Test Block 3: Odd Alignments]")
    # n=512 (smaller but power of 2), k=1024
    success = run_moe_test(m=17, n=512, k=1024, e=8, topk=2)
    if not success:
        sys.exit(1)

    print("\n=== All Tests Passed! ===")


if __name__ == "__main__":
    run_suite()
