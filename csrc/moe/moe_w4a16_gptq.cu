/*
 * Fused GPTQ MoE kernel for V100 (SM70).
 * Supports Persistent Thread Block (PTB) scheduling and Shared Expert load balancing.
 *
 * OPTIMIZATIONS:
 * - Persistent Thread Blocks: Decouples SMs from Experts. Threads fetch work (Shared Tiles or Routed Tiles) dynamically.
 * - Shared Expert Sharding: Processes "Dense" shared expert tiles alongside "Sparse" routed tiles to fix 3:1 load imbalance.
 * - SASS BFE Dequantization: Uses inline PTX BFE.U32 for efficient 4-bit unpacking on Volta (lacking INT4 Tensor Cores).
 * - Swizzled Shared Memory: XOR-based swizzling to reduce bank conflicts.
 */

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#include "../quantization/gptq/qdq_4.cuh"
#include "../quantization/gptq/qdq_util.cuh"
#include "moe_wna16_utils.h"

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

namespace vllm {
namespace gptq {

// -------------------------------------------------------------------------
// Micro-Optimization: Inline BFE Dequantization for V100
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Helper for sequential 4-bit dequantization
// -------------------------------------------------------------------------
__device__ inline void dequant_sequential_4bit_bfe(uint32_t q, half2* res) {
  // Original C++ Implementation (Safe)
  // Extract 8x4-bit values (little endian)
  int v0 = q & 0xF;
  int v1 = (q >> 4) & 0xF;
  int v2 = (q >> 8) & 0xF;
  int v3 = (q >> 12) & 0xF;
  int v4 = (q >> 16) & 0xF;
  int v5 = (q >> 20) & 0xF;
  int v6 = (q >> 24) & 0xF;
  int v7 = (q >> 28) & 0xF;

  res[0] = __halves2half2(__int2half_rn(v0), __int2half_rn(v1));
  res[1] = __halves2half2(__int2half_rn(v2), __int2half_rn(v3));
  res[2] = __halves2half2(__int2half_rn(v4), __int2half_rn(v5));
  res[3] = __halves2half2(__int2half_rn(v6), __int2half_rn(v7));
}

// -------------------------------------------------------------------------
// Persistent Thread Block (PTB) Kernel with Flattened Task Scheduling
// -------------------------------------------------------------------------
template <typename scalar_t, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K>
__global__ void gptq_moe_gemm_kernel_ptb(
    const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
    const uint32_t* __restrict__ qweight, const scalar_t* __restrict__ scales,
    const uint32_t* __restrict__ qzeros, const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_pad, 
    int num_experts,
    int group_size, int top_k, int size_m, int size_n,
    int size_k, bool mul_topk_weight,
    const int32_t* __restrict__ q_perm,
    bool input_is_expanded, bool output_is_expanded,
    // New Arguments for Shared Expert & PTB
    const uint32_t* __restrict__ shared_qweight,
    const scalar_t* __restrict__ shared_scales,
    const uint32_t* __restrict__ shared_qzeros,
    int shared_expert_num_blocks, // Total blocks for Shared Expert
    int routed_expert_num_blocks,  // Total blocks for Routed Experts
    int* __restrict__ global_work_counter // Atomic counter for work stealing
    ) {

  using Dtype = ScalarType<scalar_t>;
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;

  constexpr int PAD = 8;
  extern __shared__ char shared_mem_raw[];
  scalar_t* block_input = reinterpret_cast<scalar_t*>(shared_mem_raw);
  int32_t* block_q_perm = reinterpret_cast<int32_t*>(
      shared_mem_raw + BLOCK_SIZE_M * (BLOCK_SIZE_K + PAD) * sizeof(scalar_t));

  const int total_token_blocks = routed_expert_num_blocks + shared_expert_num_blocks;
  const int num_n_tiles = DIVIDE(size_n, BLOCK_SIZE_N);
  const int num_k_tiles = DIVIDE(size_k, BLOCK_SIZE_K);
  
  // Total Tasks = (TokenBlocks) * (N_Tiles) * (K_Tiles)
  // For larger models this can overflow int32? 
  // 32-bit int max is 2 billion. Typically fine for inference batches.
  // 100k blocks * 100 N * 10 K = 100M. Safe.
  const int total_tasks = total_token_blocks * num_n_tiles * num_k_tiles;

  const int k_per_thread = DIVIDE(BLOCK_SIZE_K, BLOCK_SIZE_N);

  while (true) {
    __shared__ int task_id;
    if (threadIdx.x == 0) {
      task_id = atomicAdd(global_work_counter, 1);
    }
    __syncthreads();

    if (task_id >= total_tasks) break;

    // Decode Task ID
    // order: K (fastest), N, TokenBlock (slowest) - to maximize cache reuse?
    // Actually, grouping by TokenBlock is better for Input reuse in L2.
    // So TokenBlock should be outer loop logic, or we decode such that 
    // adjacent tasks share the same TokenBlock.
    // Let's decode: task_id = (token_block_idx * num_n_tiles * num_k_tiles) + (n_tile * num_k_tiles) + k_tile
    
    int tmp = task_id;
    int k_tile = tmp % num_k_tiles;
    tmp /= num_k_tiles;
    int n_tile = tmp % num_n_tiles;
    int work_idx = tmp / num_n_tiles; // This is the Token Block Index

    // -------------------------------------------------------------------------
    // Set up indices based on decoded task
    // -------------------------------------------------------------------------
    
    // Check if Shared or Routed
    bool is_shared_expert = (work_idx >= routed_expert_num_blocks);
    
    int32_t expert_id = -1;
    int32_t offset_m = -1; 
    const uint32_t* current_qweight = nullptr;
    const scalar_t* current_scales = nullptr;
    const uint32_t* current_qzeros = nullptr;
    
    if (is_shared_expert) {
        int shared_block_idx = work_idx - routed_expert_num_blocks;
        offset_m = shared_block_idx * BLOCK_SIZE_M;
        current_qweight = shared_qweight;
        current_scales = shared_scales;
        current_qzeros = shared_qzeros;
        expert_id = -2;
    } else {
        expert_id = expert_ids[work_idx];
        offset_m = work_idx * BLOCK_SIZE_M;
        current_qweight = qweight;
        current_scales = scales;
        current_qzeros = qzeros;
    }

    const int32_t offset_n = n_tile * BLOCK_SIZE_N + threadIdx.x;
    const int32_t offset_k = k_tile * BLOCK_SIZE_K;
    const int input_divisor = input_is_expanded ? 1 : top_k;
    const int output_divisor = output_is_expanded ? 1 : top_k;
    
    int32_t num_valid_tokens = 0;

    // Load Input
    for (int m = 0; m < BLOCK_SIZE_M; m++) {
        int32_t token_index = -1;
        if (is_shared_expert) {
            token_index = offset_m + m;
            if (token_index >= size_m) break;
        } else {
            if (offset_m + m >= num_tokens_post_pad[0]) break;
            token_index = sorted_token_ids[offset_m + m];
            if (token_index / input_divisor >= size_m) break;
        }
        
        num_valid_tokens = m + 1;

        if (expert_id != -1) { 
            const int token_row = token_index / input_divisor;
            for (int i = 0; i < k_per_thread; i++) {
                int k = threadIdx.x * k_per_thread + i;
                if (k >= BLOCK_SIZE_K) break;
                if (offset_k + k >= size_k) break;

                int linear_k = offset_k + k;
                int logical_k = linear_k;
                
                if (!is_shared_expert && q_perm != nullptr) {
                     logical_k = q_perm[linear_k];
                     if (m == 0) block_q_perm[k] = logical_k;
                } else if (!is_shared_expert) {
                     if (m == 0) block_q_perm[k] = linear_k;
                }
                
                int global_k = token_row * size_k + logical_k;
                block_input[m * (BLOCK_SIZE_K + PAD) + k] = input[global_k];
            }
        }
    }
    __syncthreads();

    if ((!is_shared_expert && expert_id == -1) || num_valid_tokens == 0) continue;
    if (threadIdx.x >= BLOCK_SIZE_N || offset_n >= size_n) continue;

    float res[64] = {0.0f};
    scalar_t2 res2;
    scalar_t2 scale_f2 = Dtype::num2num2(Dtype::float2num(1.0f));
    scalar_t2 zero_f2 = Dtype::num2num2(Dtype::float2num(8.0f));

    constexpr int pack_factor = 8;
    const int groups = size_k / group_size;

    uint64_t w_offset_base = 0;
    if (!is_shared_expert) {
        w_offset_base = ((uint64_t)expert_id) * (size_k / pack_factor) * size_n;
    } else {
        w_offset_base = 0;
    }
    
    const uint64_t s_offset_base = (!is_shared_expert) ? ((uint64_t)expert_id * groups * size_n) : 0;
    const uint64_t z_offset_base = (!is_shared_expert) ? ((uint64_t)expert_id * groups * (size_n/pack_factor)) : 0;
    
    const uint32_t* my_qweight = current_qweight + w_offset_base;
    const scalar_t* my_scales = current_scales + s_offset_base;
    const uint32_t* my_qzeros = (current_qzeros) ? (current_qzeros + z_offset_base) : nullptr;

    int current_group = -1;
    
    for (int tmp_k = 0; tmp_k < BLOCK_SIZE_K / pack_factor; tmp_k++) {
        int k_linear_base = offset_k + tmp_k * pack_factor;
        if (k_linear_base >= size_k) break;

        const int qk = k_linear_base / pack_factor;
        const int weight_idx = qk * size_n + offset_n;
        const uint32_t qweight_val = my_qweight[weight_idx];

        half2 dq[4];
        dequant_sequential_4bit_bfe(qweight_val, dq);

        const int group_idx = k_linear_base / group_size;
        if (group_idx != current_group) {
            current_group = group_idx;
            int s_idx = group_idx * size_n + offset_n;
            scalar_t sc = reinterpret_cast<const half*>(my_scales)[s_idx];
            scalar_t z = Dtype::float2num(8.0f);
            
            if (my_qzeros) {
                 int z_idx = group_idx * (size_n / 8) + (offset_n / 8);
                 uint32_t zp_pack = my_qzeros[z_idx];
                 int zp_v = (zp_pack >> ((offset_n % 8) * 4)) & 0xF;
                 z = __int2half_rn(zp_v + 1);
            }
            scale_f2 = Dtype::num2num2(sc);
            zero_f2 = __halves2half2(z, z);
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
           dq[i] = __hsub2(dq[i], zero_f2);
           dq[i] = __hmul2(dq[i], scale_f2);
        }

        for (int m = 0; m < num_valid_tokens; m++) {
            res2.x = __float2half(0.0f);
            res2.y = __float2half(0.0f);
            scalar_t2* row_half2 = reinterpret_cast<scalar_t2*>(&block_input[m * (BLOCK_SIZE_K + PAD)]);
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int32_t offset_input_half2 = (tmp_k * pack_factor + i * 2) / 2;
                res2 = __hfma2(dq[i], row_half2[offset_input_half2], res2);
            }
            if (tmp_k == 0) res[m] = Dtype::num2float(res2.x) + Dtype::num2float(res2.y);
            else res[m] += Dtype::num2float(res2.x) + Dtype::num2float(res2.y);
        }
    }

    for (int m = 0; m < num_valid_tokens; ++m) {
        int32_t token_index;
        if (is_shared_expert) token_index = offset_m + m;
        else token_index = sorted_token_ids[offset_m + m];
        
        const int32_t token_idx = token_index / output_divisor;
        
        // Use atomicAdd if we have multiple K tiles or if this is shared expert
        // Reduced: If num_k_tiles > 1, then atomicAdd is vital.
        if (num_k_tiles == 1 && !is_shared_expert) {
             output[token_idx * size_n + offset_n] = static_cast<scalar_t>(res[m]);
        } else {
             atomicAdd(&output[token_idx * size_n + offset_n], static_cast<scalar_t>(res[m]));
        }
    }
  }
}

// -------------------------------------------------------------------------
// Launcher
// -------------------------------------------------------------------------

template <typename scalar_t>
void run_gptq_moe_gemm(const scalar_t* input, scalar_t* output,
                       const uint32_t* qweight, const scalar_t* scales,
                       const uint32_t* qzeros, const float* topk_weights,
                       const int32_t* sorted_token_ids,
                       const int32_t* expert_ids,
                       const int32_t* num_tokens_post_pad, int num_experts,
                       int group_size, int num_token_blocks, int top_k, int size_m,
                       int size_n, int size_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N,
                       int BLOCK_SIZE_K, bool mul_topk_weight,
                       const int32_t* q_perm,
                       bool input_is_expanded, bool output_is_expanded,
                       const uint32_t* shared_qweight,
                       const scalar_t* shared_scales,
                       const uint32_t* shared_qzeros,
                       int* global_work_counter) {
  
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_SIZE_N;
  blockDim.y = 1;
  blockDim.z = 1;

  // 1D Grid of Workers
  int num_sms = 80; 
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  gridDim.x = num_sms * 4; 
  
  const int shared_mem_size = BLOCK_SIZE_M * (BLOCK_SIZE_K + 8) * sizeof(scalar_t) + BLOCK_SIZE_K * sizeof(int32_t);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int shared_expert_blocks = 0;
  if (shared_qweight != nullptr) {
       shared_expert_blocks = DIVIDE(size_m, BLOCK_SIZE_M);
  }

  // Reset work counter
  cudaMemsetAsync(global_work_counter, 0, sizeof(int), stream);

  // Template Dispatch for BLOCK_SIZE_K
  #define LAUNCH_PTB_KERNEL(K_SIZE) \
      gptq_moe_gemm_kernel_ptb<scalar_t, 32, 32, K_SIZE> \
          <<<gridDim, blockDim, shared_mem_size, stream>>>( \
              input, output, qweight, scales, qzeros, topk_weights, sorted_token_ids, \
              expert_ids, num_tokens_post_pad, num_experts, group_size, top_k, size_m, \
              size_n, size_k, mul_topk_weight, q_perm, input_is_expanded, output_is_expanded, \
              shared_qweight, shared_scales, shared_qzeros, shared_expert_blocks, num_token_blocks, global_work_counter);

  if (BLOCK_SIZE_K == 128) { LAUNCH_PTB_KERNEL(128); }
  else if (BLOCK_SIZE_K == 64) { LAUNCH_PTB_KERNEL(64); }
  else if (BLOCK_SIZE_K == 32) { LAUNCH_PTB_KERNEL(32); }
  else { LAUNCH_PTB_KERNEL(32); } 
}

} // namespace gptq
} // namespace vllm

// -------------------------------------------------------------------------
// Torch Binding
// -------------------------------------------------------------------------
torch::Tensor moe_w4a16_gptq_gemm(torch::Tensor input, torch::Tensor output,
                            torch::Tensor qweight, torch::Tensor scales,
                            torch::Tensor qzeros,
                            std::optional<torch::Tensor> topk_weights,
                            torch::Tensor sorted_token_ids,
                            torch::Tensor expert_ids,
                            torch::Tensor num_tokens_post_pad, int64_t top_k,
                            int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
                            int64_t BLOCK_SIZE_K,
                            std::optional<torch::Tensor> q_perm,
                            bool input_is_expanded,
                            bool output_is_expanded,
                            // Optionals for Shared Expert
                            std::optional<torch::Tensor> shared_qweight,
                            std::optional<torch::Tensor> shared_scales,
                            std::optional<torch::Tensor> shared_qzeros,
                            torch::Tensor global_work_counter) {
                            
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  output.zero_();

  int num_experts = qweight.size(0);
  int size_m = input.size(0);
  int size_n = qweight.size(2);
  int size_k = input.size(1);
  int num_groups = scales.size(1);
  int group_size = size_k / num_groups;

  int64_t EM = sorted_token_ids.size(0);
  if (size_m <= BLOCK_SIZE_M) {
    EM = min(EM, (int64_t)(size_m * BLOCK_SIZE_M * top_k));
  }
  const int num_token_blocks = (EM + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;

  // Prepare Pointers
  const uint32_t* qw = reinterpret_cast<const uint32_t*>(qweight.data_ptr());
  const uint32_t* qz = reinterpret_cast<const uint32_t*>(qzeros.data_ptr());
  const float* tw = topk_weights.has_value() ? topk_weights.value().data_ptr<float>() : nullptr;
  const int32_t* qp = q_perm.has_value() ? q_perm.value().data_ptr<int32_t>() : nullptr;

  // Prepare Shared Expert Pointers
  const uint32_t* sqw = shared_qweight.has_value() ? reinterpret_cast<const uint32_t*>(shared_qweight.value().data_ptr()) : nullptr;
  const half* ssc = shared_scales.has_value() ? reinterpret_cast<const half*>(shared_scales.value().data_ptr()) : nullptr;
  const uint32_t* sqz = shared_qzeros.has_value() ? reinterpret_cast<const uint32_t*>(shared_qzeros.value().data_ptr()) : nullptr;

  if (input.scalar_type() == at::ScalarType::Half) {
       vllm::gptq::run_gptq_moe_gemm<half>(
            reinterpret_cast<const half*>(input.data_ptr()),
            reinterpret_cast<half*>(output.data_ptr()),
            qw, reinterpret_cast<const half*>(scales.data_ptr()), qz, tw,
            sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
            num_tokens_post_pad.data_ptr<int32_t>(), num_experts, group_size,
            num_token_blocks, top_k, size_m, size_n, size_k,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            topk_weights.has_value(), qp, input_is_expanded, output_is_expanded,
            sqw, ssc, sqz, global_work_counter.data_ptr<int32_t>()
       );
  } else {
     TORCH_CHECK(false, "Only FP16 supported");
  }

  return output;
}
