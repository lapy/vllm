import torch
import pytest
import time
from vllm import _custom_ops as ops

def run_benchmark(num_experts, num_tokens, hidden_size, intermediate_size, group_size, top_k):
    print(f"Benchmarking: E={num_experts}, T={num_tokens}, H={hidden_size}, I={intermediate_size}, G={group_size}, K={top_k}")
    
    dtype = torch.float16
    device = torch.device("cuda")

    # Inputs
    a = torch.randn((num_tokens, hidden_size), device=device, dtype=dtype)
    
    # qweight: [E, K/8, N]
    
    K = hidden_size
    N = intermediate_size
    E = num_experts
    
    qweight = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (E, K // 8, N), device=device, dtype=torch.int32)
    scales = torch.randn((E, K // group_size, N), device=device, dtype=dtype)
    qzeros = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (E, K // group_size, N // 8), device=device, dtype=torch.int32)
    
    topk_weights = torch.rand((num_tokens, top_k), device=device, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    
    # Fake expert assignment
    sorted_token_ids = torch.arange(num_tokens * top_k, device=device, dtype=torch.int32)
    expert_ids = torch.randint(0, E, (num_tokens * top_k // 32 + 1,), device=device, dtype=torch.int32).repeat_interleave(32)[:num_tokens * top_k]
    # Actually expert_ids in kernel is per-block (BLOCK_SIZE_M)
    # Let's just use a simple setup where we process everything
    
    # Need to setup expert_ids properly for the kernel
    # kernel arg: expert_ids [num_blocks]
    BLOCK_SIZE_M = 32
    num_blocks = (num_tokens * top_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    expert_ids = torch.randint(0, E, (num_blocks,), device=device, dtype=torch.int32)
    
    num_tokens_post_pad = torch.tensor([num_tokens * top_k], device=device, dtype=torch.int32)
    
    output = torch.zeros((num_tokens, top_k, N), device=device, dtype=dtype)
    
    # Warmup
    for _ in range(5):
        ops.moe_w4a16_gptq_gemm(
            a, output, qweight, scales, qzeros, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_pad,
            top_k, 32, 32, 128, None, False, False
        )
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(20):
        ops.moe_w4a16_gptq_gemm(
            a, output, qweight, scales, qzeros, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_pad,
            top_k, 32, 32, 128, None, False, False
        )
    end_event.record()
    torch.cuda.synchronize()
    
    avg_latency = start_event.elapsed_time(end_event) / 20
    print(f"Latency: {avg_latency:.3f} ms")


if __name__ == "__main__":
    run_benchmark(8, 128, 4096, 11008, 128, 2)
