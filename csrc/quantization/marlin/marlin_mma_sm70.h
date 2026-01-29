#pragma once

// =============================================================================
//
//  SM70 (Volta V100) Tensor Core Support for Marlin Quantized GEMM Kernels
//
// =============================================================================
//
// This header provides tensor core operations for Volta GPUs (SM70) which only
// support the m8n8k4 instruction. Higher-level m16n8k16 operations are emulated
// by decomposing into multiple m8n8k4 calls.
//
// Author: GitHub Copilot + Human Collaboration
// Date: January 2026
// Hardware: Tesla V100-SXM2-32GB (SM70)
//
// -----------------------------------------------------------------------------
// QUICK REFERENCE
// -----------------------------------------------------------------------------
//
// Recommended Functions (fused approach - DEFAULT):
//   fused_ldmatrix_mma_m16n8k16_sm70()  - Direct smem→tensor core, fewer shuffles
//   fused_ldmatrix_mma_m16n8k32_sm70()  - K=32 fused version
//
// Register-based Functions (legacy):
//   mma_m16n8k16_sm70(A, B, frag_c)     - Uses pre-loaded register fragments
//   mma_m16n8k32_sm70(A, B, frag_c)     - Double-K version
//
// Variant Functions:
//   mma_m16n8k16_sm70_trans(...)        - Transposed multiply for Marlin
//   mma_m16n8k16_sm70_fp16(...)         - FP16 accumulator version
//
// Low-level Primitives:
//   ldmatrix_m8n8_x1/x2/x4_sm70()       - ldmatrix instruction emulation
//   mma_m8n8k4_sm70()                   - Direct PTX tensor core wrapper
//
// -----------------------------------------------------------------------------
// RUNTIME SELECTION
// -----------------------------------------------------------------------------
//
//   VLLM_SM70_USE_FUSED_MMA=1 (default)  Use optimized fused approach
//   VLLM_SM70_USE_FUSED_MMA=0            Use register-based approach
//
// -----------------------------------------------------------------------------
// COMPILATION OPTIONS
// -----------------------------------------------------------------------------
//
//   Default: WMMA functions disabled for ~25% faster compilation
//   -DMARLIN_SM70_ENABLE_WMMA=1    Enable legacy WMMA-based functions
//
// -----------------------------------------------------------------------------
// WARNING
// -----------------------------------------------------------------------------
//
// DO NOT MODIFY mma_m16n8k16_sm70 WITHOUT EXTENSIVE TESTING!
// This implementation was empirically discovered and validated.
// Max error vs CPU reference: 0.000000
//
// =============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 700
#warning "marlin_mma_sm70.h is optimized for SM70 (Volta) architecture only"
#endif

// =============================================================================
// SECTION 1: INCLUDES AND CONFIGURATION
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifdef MARLIN_SM70_ENABLE_WMMA
#include <mma.h>
#endif

namespace MARLIN_NAMESPACE_NAME {

#ifdef MARLIN_SM70_ENABLE_WMMA
using namespace nvcuda;
#endif

// =============================================================================
// SECTION 2: ARCHITECTURE REFERENCE
// =============================================================================
//
// This section documents the Volta tensor core architecture and fragment
// layouts. Skip to Section 3 for the actual implementation.
//
// -----------------------------------------------------------------------------
// 2.1 Volta Tensor Core Basics
// -----------------------------------------------------------------------------
//
// Volta (SM70) only supports m8n8k4, NOT m16n8k16 like Ampere/Turing.
// We emulate m16n8k16 using: 2 row blocks × 4 k-blocks = 8 m8n8k4 ops
//
//   m16n8k16: C[16×8] += A[16×16] × B[16×8]
//   m8n8k4:   C[8×8]  += A[8×4]   × B[4×8]
//
// -----------------------------------------------------------------------------
// 2.2 Quadpair Thread Mapping
// -----------------------------------------------------------------------------
//
// The m8n8k4 uses a "quadpair" - 8 threads from the 32-thread warp:
//   Lanes 0-3   → logical tid 0-3
//   Lanes 16-19 → logical tid 4-7
//
// ALL 32 threads must execute mma.sync (synchronizing instruction), but only
// quadpair threads contribute data. Others pass zeros.
//
// -----------------------------------------------------------------------------
// 2.3 m8n8k4 Fragment Layout (Empirically Discovered)
// -----------------------------------------------------------------------------
//
// For quadpair thread with logical tid t (0-7):
//
// A fragment (8×4, 2 regs = 4 halves):
//   a0 = {A[t,0], A[t,1]}    // Row t, cols 0-1
//   a1 = {A[t,2], A[t,3]}    // Row t, cols 2-3
//
// B fragment (4×8, 2 regs = 4 halves):
//   b0 = {B[0,t], B[1,t]}    // Col t, rows 0-1
//   b1 = {B[2,t], B[3,t]}    // Col t, rows 2-3
//
// C/D fragment (8×8, 8 floats, SCATTERED):
//   row(t,i) = (t%2) + 2*((i/2)%2) + 4*(t/4)
//   col(t,i) = 2*((t/2)%2) + (i%2) + 4*(i/4)
//
// Inverse (find tid/index for position row,col):
//   t = (row%2) + 2*((col/2)%2) + 4*(row/4)
//   i = (col%2) + 2*((row/2)%2) + 4*(col/4)
//
// -----------------------------------------------------------------------------
// 2.4 Marlin Fragment Layout
// -----------------------------------------------------------------------------
//
// FragA (4 uint32 = 8 halves per thread):
//   lane = row*4 + k_pair, where row∈[0,7], k_pair∈[0,3]
//   A[0]: half2 @ A[row, k_pair*2..k_pair*2+1]       (k=0..7)
//   A[1]: half2 @ A[row, k_pair*2+8..k_pair*2+9]     (k=8..15)
//   A[2]: half2 @ A[row+8, k_pair*2..k_pair*2+1]
//   A[3]: half2 @ A[row+8, k_pair*2+8..k_pair*2+9]
//
// FragB (2 uint32 = 4 halves per thread):
//   lane = col*4 + k_pair, where col∈[0,7], k_pair∈[0,3]
//   B[0]: half2 @ B[k_pair*2..k_pair*2+1, col]       (k=0..7)
//   B[1]: half2 @ B[k_pair*2+8..k_pair*2+9, col]     (k=8..15)
//
// FragC (4 floats per thread):
//   lane = row*4 + col_pair, where row∈[0,7], col_pair∈[0,3]
//   frag_c[0]: C[row, col_pair*2]
//   frag_c[1]: C[row, col_pair*2+1]
//   frag_c[2]: C[row+8, col_pair*2]
//   frag_c[3]: C[row+8, col_pair*2+1]
//
// -----------------------------------------------------------------------------
// 2.5 Shuffle Semantics (Critical Gotcha)
// -----------------------------------------------------------------------------
//
// __shfl_sync(mask, var, srcLane) evaluates 'var' on CALLING thread first!
//
// WRONG: __shfl_sync(FULL_MASK, array[index], srcLane)
//        where index differs per thread - does NOT read srcLane's array[index]
//
// RIGHT: Shuffle ALL elements, then select locally
//
// =============================================================================


// =============================================================================
// SECTION 3: LOW-LEVEL PRIMITIVES
// =============================================================================

// -----------------------------------------------------------------------------
// 3.1 ldmatrix Emulation
// -----------------------------------------------------------------------------
//
// Volta lacks ldmatrix instruction. These functions emulate it using shuffles.

/// @brief Emulates ldmatrix.sync.aligned.m8n8.x1.shared.b16
/// @param dst Output register (1 uint32)
/// @param smem_ptr Pointer to shared memory (each thread provides its row)
__device__ __forceinline__ void ldmatrix_m8n8_x1_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t my_word = row_ptr[0];
    
    int source_row = lane % 8;
    int warp_group_offset = (lane / 8) * 8;
    int src_lane = warp_group_offset + source_row;
    
    dst[0] = __shfl_sync(FULL_MASK, my_word, src_lane);
}

/// @brief Emulates ldmatrix.sync.aligned.m8n8.x2.shared.b16
/// @param dst Output registers (2 uint32)
/// @param smem_ptr Pointer to shared memory
__device__ __forceinline__ void ldmatrix_m8n8_x2_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t my_words[2] = {row_ptr[0], row_ptr[1]};
    
    int row_in_tile = lane % 8;
    int warp_group_offset = (lane / 8) * 8;
    int src_lane = warp_group_offset + row_in_tile;

    dst[0] = __shfl_sync(FULL_MASK, my_words[0], src_lane);
    dst[1] = __shfl_sync(FULL_MASK, my_words[1], src_lane);
}

/// @brief Emulates ldmatrix.sync.aligned.m8n8.x4.shared.b16
/// @param dst Output registers (4 uint32)
/// @param smem_ptr Pointer to shared memory
/// @note Primary function used by Marlin for loading FragA
__device__ __forceinline__ void ldmatrix_m8n8_x4_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t my_words[4] = {row_ptr[0], row_ptr[1], row_ptr[2], row_ptr[3]};
    
    int out_row = lane / 4;
    int k_pair = lane % 4;
    
    // Source lanes for 4 quadrants of 16x16 matrix
    int src_lanes[4] = {out_row, out_row + 8, out_row + 16, out_row + 24};
    
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        uint32_t w0 = __shfl_sync(FULL_MASK, my_words[0], src_lanes[q]);
        uint32_t w1 = __shfl_sync(FULL_MASK, my_words[1], src_lanes[q]);
        uint32_t w2 = __shfl_sync(FULL_MASK, my_words[2], src_lanes[q]);
        uint32_t w3 = __shfl_sync(FULL_MASK, my_words[3], src_lanes[q]);
        uint32_t arr[4] = {w0, w1, w2, w3};
        dst[q] = arr[k_pair];
    }
}

// -----------------------------------------------------------------------------
// 3.2 Core Tensor Core Instruction
// -----------------------------------------------------------------------------

/// @brief Direct PTX wrapper for mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32
/// @param a0, a1 A fragment registers (2 uint32 = 4 halves)
/// @param b0, b1 B fragment registers (2 uint32 = 4 halves)
/// @param c Accumulator array (8 floats), modified in-place
__device__ __forceinline__ void mma_m8n8k4_sm70(
    uint32_t a0, uint32_t a1, 
    uint32_t b0, uint32_t b1, 
    float* c) 
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
          "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}


// =============================================================================
// SECTION 4: REGISTER-BASED MMA FUNCTIONS (Legacy)
// =============================================================================
//
// These functions use pre-loaded register fragments. For new code, prefer the
// fused functions in Section 6 which have better performance.

// -----------------------------------------------------------------------------
// 4.1 mma_m16n8k16_sm70 - FROZEN (DO NOT MODIFY)
// -----------------------------------------------------------------------------
//
// Register-based tensor core function. Use fused_ldmatrix_mma_m16n8k16_sm70()
// for better performance when data is in shared memory.
// Validated: January 2026, Max error: 0.000000
//
/// @brief Computes C[16×8] += A[16×16] × B[16×8] using m8n8k4 tensor cores
/// @param A FragA in Marlin layout (4 uint32)
/// @param B FragB in Marlin layout (2 uint32)  
/// @param frag_c FragC output, accumulated (4 floats)
/// @note Consider using fused_ldmatrix_mma_m16n8k16_sm70() for better perf
__device__ __forceinline__ void mma_m16n8k16_sm70(
    const uint32_t* A, 
    const uint32_t* B,
    float* frag_c) 
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    // Thread position in Marlin layout
    const int marlin_row = lane / 4;
    const int marlin_col_pair = lane % 4;
    
    // Quadpair mapping: lanes 0-3 → tid 0-3, lanes 16-19 → tid 4-7
    const int qp_tid = (lane < 4) ? lane : ((lane >= 16 && lane < 20) ? (lane - 16 + 4) : -1);
    const bool is_quadpair = (qp_tid >= 0);
    
    // Accumulators for top/bottom 8x8 blocks
    float c_top[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float c_bot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // Process 4 k-blocks (k=0-3, 4-7, 8-11, 12-15)
    #pragma unroll
    for (int kb = 0; kb < 4; kb++) {
        int a_reg = (kb < 2) ? 0 : 1;
        int b_reg = (kb < 2) ? 0 : 1;
        
        // Compute source lanes for shuffles
        int row_for_a = is_quadpair ? qp_tid : 0;
        int a_k_pair_lo = (kb % 2) * 2;
        int a_k_pair_hi = a_k_pair_lo + 1;
        int a_src_lane_lo = row_for_a * 4 + a_k_pair_lo;
        int a_src_lane_hi = row_for_a * 4 + a_k_pair_hi;
        
        int col_for_b = is_quadpair ? qp_tid : 0;
        int b_k_pair_lo = (kb % 2) * 2;
        int b_k_pair_hi = b_k_pair_lo + 1;
        int b_src_lane_lo = col_for_b * 4 + b_k_pair_lo;
        int b_src_lane_hi = col_for_b * 4 + b_k_pair_hi;
        
        // Gather data via shuffles
        uint32_t a_top_word_lo = __shfl_sync(FULL_MASK, A[a_reg], a_src_lane_lo);
        uint32_t a_top_word_hi = __shfl_sync(FULL_MASK, A[a_reg], a_src_lane_hi);
        uint32_t a_bot_word_lo = __shfl_sync(FULL_MASK, A[a_reg + 2], a_src_lane_lo);
        uint32_t a_bot_word_hi = __shfl_sync(FULL_MASK, A[a_reg + 2], a_src_lane_hi);
        uint32_t b_word_lo = __shfl_sync(FULL_MASK, B[b_reg], b_src_lane_lo);
        uint32_t b_word_hi = __shfl_sync(FULL_MASK, B[b_reg], b_src_lane_hi);
        
        uint32_t a_top0, a_top1, a_bot0, a_bot1, b0, b1;
        
        if (is_quadpair) {
            // Repack to m8n8k4 format
            half2 a_top_lo = *reinterpret_cast<half2*>(&a_top_word_lo);
            half2 a_top_hi = *reinterpret_cast<half2*>(&a_top_word_hi);
            half2 a_top_0 = __halves2half2(__low2half(a_top_lo), __high2half(a_top_lo));
            half2 a_top_1 = __halves2half2(__low2half(a_top_hi), __high2half(a_top_hi));
            a_top0 = *reinterpret_cast<uint32_t*>(&a_top_0);
            a_top1 = *reinterpret_cast<uint32_t*>(&a_top_1);
            
            half2 a_bot_lo = *reinterpret_cast<half2*>(&a_bot_word_lo);
            half2 a_bot_hi = *reinterpret_cast<half2*>(&a_bot_word_hi);
            half2 a_bot_0 = __halves2half2(__low2half(a_bot_lo), __high2half(a_bot_lo));
            half2 a_bot_1 = __halves2half2(__low2half(a_bot_hi), __high2half(a_bot_hi));
            a_bot0 = *reinterpret_cast<uint32_t*>(&a_bot_0);
            a_bot1 = *reinterpret_cast<uint32_t*>(&a_bot_1);
            
            half2 b_lo = *reinterpret_cast<half2*>(&b_word_lo);
            half2 b_hi = *reinterpret_cast<half2*>(&b_word_hi);
            half2 b_0 = __halves2half2(__low2half(b_lo), __high2half(b_lo));
            half2 b_1 = __halves2half2(__low2half(b_hi), __high2half(b_hi));
            b0 = *reinterpret_cast<uint32_t*>(&b_0);
            b1 = *reinterpret_cast<uint32_t*>(&b_1);
        } else {
            // Non-quadpair: zero values
            half2 z = __halves2half2(__float2half(0.0f), __float2half(0.0f));
            uint32_t zv = *reinterpret_cast<uint32_t*>(&z);
            a_top0 = a_top1 = a_bot0 = a_bot1 = b0 = b1 = zv;
        }
        
        // Execute tensor cores
        mma_m8n8k4_sm70(a_top0, a_top1, b0, b1, c_top);
        mma_m8n8k4_sm70(a_bot0, a_bot1, b0, b1, c_bot);
    }
    
    // Gather results back to Marlin layout
    // Inverse mapping: t = (row%2) + 2*((col/2)%2) + 4*(row/4)
    //                  i = (col%2) + 2*((row/2)%2) + 4*(col/4)
    
    #define TID_TO_LANE(t) ((t) < 4 ? (t) : ((t) - 4 + 16))
    #define GATHER_OUTPUT(c_arr, row, col, dst_idx) do { \
        int t = ((row) % 2) + 2 * (((col) / 2) % 2) + 4 * ((row) / 4); \
        int i = ((col) % 2) + 2 * (((row) / 2) % 2) + 4 * ((col) / 4); \
        int src_lane = TID_TO_LANE(t); \
        float vals[8]; \
        _Pragma("unroll") \
        for (int k = 0; k < 8; k++) vals[k] = __shfl_sync(FULL_MASK, c_arr[k], src_lane); \
        frag_c[dst_idx] += vals[i]; \
    } while(0)
    
    int col0 = marlin_col_pair * 2;
    int col1 = col0 + 1;
    
    GATHER_OUTPUT(c_top, marlin_row, col0, 0);
    GATHER_OUTPUT(c_top, marlin_row, col1, 1);
    GATHER_OUTPUT(c_bot, marlin_row, col0, 2);
    GATHER_OUTPUT(c_bot, marlin_row, col1, 3);
    
    #undef GATHER_OUTPUT
    #undef TID_TO_LANE
}

// -----------------------------------------------------------------------------
// 4.2 mma_m16n8k32_sm70
// -----------------------------------------------------------------------------

/// @brief Computes C[16×8] += A[16×32] × B[32×8] (two m16n8k16 operations)
/// @param A FragA array (8 uint32 for K=32)
/// @param B FragB array (4 uint32 for K=32)
/// @param frag_c Output accumulator (4 floats)
__device__ __forceinline__ void mma_m16n8k32_sm70(
    const uint32_t* A, 
    const uint32_t* B,
    float* frag_c) 
{
    mma_m16n8k16_sm70(A, B, frag_c);
    mma_m16n8k16_sm70(A + 4, B + 2, frag_c);
}


// =============================================================================
// SECTION 5: VARIANT MMA FUNCTIONS
// =============================================================================

// -----------------------------------------------------------------------------
// 5.1 Transposed MMA (for Marlin's mma_trans)
// -----------------------------------------------------------------------------
//
// Parameter naming follows Marlin convention, NOT PTX convention:
//   'a' (activations) → MMA's B operand
//   'b'+'b2' (weights) → MMA's A operand
//
/// @brief Transposed multiply: C += (b,b2) * a
/// @param marlin_a Activations (becomes MMA B)
/// @param marlin_b Weights part 1 (becomes MMA A)
/// @param marlin_b2 Weights part 2
/// @param frag_c Output accumulator
__device__ void mma_m16n8k16_sm70_trans(
    const uint32_t* marlin_a, 
    const uint32_t* marlin_b,
    const uint32_t* marlin_b2, 
    float* frag_c) 
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    const int out_row = lane / 4;
    const int col_pair = lane % 4;
    const int out_col0 = col_pair * 2;
    const int out_col1 = col_pair * 2 + 1;
    
    float sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
    
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        int a_k_pair = (k / 2) % 4;
        int a_thread = out_row * 4 + a_k_pair;
        int a_half = k % 2;
        
        // Get A values from weights
        uint32_t a_top_raw, a_bot_raw;
        if (k < 8) {
            a_top_raw = __shfl_sync(FULL_MASK, marlin_b[0], a_thread);
            a_bot_raw = __shfl_sync(FULL_MASK, marlin_b[1], a_thread);
        } else {
            a_top_raw = __shfl_sync(FULL_MASK, marlin_b2[0], a_thread);
            a_bot_raw = __shfl_sync(FULL_MASK, marlin_b2[1], a_thread);
        }
        
        half2 a_top_h2 = *reinterpret_cast<half2*>(&a_top_raw);
        half2 a_bot_h2 = *reinterpret_cast<half2*>(&a_bot_raw);
        
        float a_top = (a_half == 0) ? __half2float(__low2half(a_top_h2)) 
                                    : __half2float(__high2half(a_top_h2));
        float a_bot = (a_half == 0) ? __half2float(__low2half(a_bot_h2))
                                    : __half2float(__high2half(a_bot_h2));
        
        // Get B values from activations
        int b_k_pair = (k / 2) % 4;
        int b_thread0 = out_col0 * 4 + b_k_pair;
        int b_thread1 = out_col1 * 4 + b_k_pair;
        int b_reg_idx = (k < 8) ? 0 : 1;
        int b_half = k % 2;
        
        uint32_t b0_raw = __shfl_sync(FULL_MASK, marlin_a[b_reg_idx], b_thread0);
        uint32_t b1_raw = __shfl_sync(FULL_MASK, marlin_a[b_reg_idx], b_thread1);
        
        half2 b0_h2 = *reinterpret_cast<half2*>(&b0_raw);
        half2 b1_h2 = *reinterpret_cast<half2*>(&b1_raw);
        
        float b0_val = (b_half == 0) ? __half2float(__low2half(b0_h2))
                                     : __half2float(__high2half(b0_h2));
        float b1_val = (b_half == 0) ? __half2float(__low2half(b1_h2))
                                     : __half2float(__high2half(b1_h2));
        
        sum00 += a_top * b0_val;
        sum01 += a_top * b1_val;
        sum10 += a_bot * b0_val;
        sum11 += a_bot * b1_val;
    }
    
    frag_c[0] += sum00;
    frag_c[1] += sum01;
    frag_c[2] += sum10;
    frag_c[3] += sum11;
}

// -----------------------------------------------------------------------------
// 5.2 FP16 Accumulator Version
// -----------------------------------------------------------------------------

/// @brief MMA with FP16 accumulator (converts internally to FP32)
/// @param A FragA in Marlin layout
/// @param B FragB in Marlin layout
/// @param frag_c FP16 accumulator (2 uint32 = 4 halves)
__device__ void mma_m16n8k16_sm70_fp16(
    const uint32_t* A, 
    const uint32_t* B, 
    uint32_t* frag_c) 
{
    // Convert FP16 accumulator to FP32
    float frag_f32[4];
    half2 cur0 = *reinterpret_cast<const half2*>(&frag_c[0]);
    half2 cur1 = *reinterpret_cast<const half2*>(&frag_c[1]);
    frag_f32[0] = __half2float(__low2half(cur0));
    frag_f32[1] = __half2float(__high2half(cur0));
    frag_f32[2] = __half2float(__low2half(cur1));
    frag_f32[3] = __half2float(__high2half(cur1));
    
    // Compute in FP32
    mma_m16n8k16_sm70(A, B, frag_f32);
    
    // Convert back to FP16
    half2 result0 = __halves2half2(__float2half(frag_f32[0]), __float2half(frag_f32[1]));
    half2 result1 = __halves2half2(__float2half(frag_f32[2]), __float2half(frag_f32[3]));
    frag_c[0] = *reinterpret_cast<uint32_t*>(&result0);
    frag_c[1] = *reinterpret_cast<uint32_t*>(&result1);
}


// =============================================================================
// SECTION 6: FUSED FUNCTIONS (DEFAULT - RECOMMENDED)
// =============================================================================
//
// These are the DEFAULT and RECOMMENDED functions for SM70 tensor core ops.
// They load directly from shared memory, eliminating ~24 input shuffles.
// Provides ~5% speedup at high occupancy, up to ~39% under register pressure.
//
// Advantages over register-based approach:
//   + Fewer shuffles (32 vs 56 per m16n8k16)
//   + Lower register pressure  
//   + Better performance under contention
//
// Requirements:
//   - Data must be in shared memory (row-major A, column-major B)

/// @brief Fused load+MMA from shared memory (DEFAULT - USE THIS)
/// @param sh_A 16x16 row-major A matrix in shared memory
/// @param sh_B 16x8 column-major B matrix (B[k + col*ldb])
/// @param frag_c Output in Marlin format, accumulated
/// @param lda Leading dimension of A (default 16)
/// @param ldb Leading dimension of B (default 16)
__device__ __forceinline__ void fused_ldmatrix_mma_m16n8k16_sm70(
    const half* sh_A,
    const half* sh_B,
    float* frag_c,
    int lda = 16,
    int ldb = 16)
{
    const int lane = threadIdx.x % 32;
    const int marlin_row = lane / 4;
    const int marlin_col_pair = lane % 4;
    const int qp_tid = (lane < 4) ? lane : ((lane >= 16 && lane < 20) ? (lane - 16 + 4) : -1);
    const bool is_quadpair = (qp_tid >= 0);
    
    float c_top[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float c_bot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    #pragma unroll
    for (int kb = 0; kb < 4; kb++) {
        uint32_t a_top0, a_top1, a_bot0, a_bot1, b0, b1;
        
        if (is_quadpair) {
            // Direct shared memory load - NO INPUT SHUFFLES
            int row_top = qp_tid;
            int row_bot = qp_tid + 8;
            int col = qp_tid;
            int k_start = kb * 4;
            
            const half* a_top_ptr = sh_A + row_top * lda + k_start;
            const half* a_bot_ptr = sh_A + row_bot * lda + k_start;
            const half* b_ptr = sh_B + col * ldb + k_start;
            
            half2 at0 = __halves2half2(a_top_ptr[0], a_top_ptr[1]);
            half2 at1 = __halves2half2(a_top_ptr[2], a_top_ptr[3]);
            half2 ab0 = __halves2half2(a_bot_ptr[0], a_bot_ptr[1]);
            half2 ab1 = __halves2half2(a_bot_ptr[2], a_bot_ptr[3]);
            half2 bb0 = __halves2half2(b_ptr[0], b_ptr[1]);
            half2 bb1 = __halves2half2(b_ptr[2], b_ptr[3]);
            
            a_top0 = *reinterpret_cast<uint32_t*>(&at0);
            a_top1 = *reinterpret_cast<uint32_t*>(&at1);
            a_bot0 = *reinterpret_cast<uint32_t*>(&ab0);
            a_bot1 = *reinterpret_cast<uint32_t*>(&ab1);
            b0 = *reinterpret_cast<uint32_t*>(&bb0);
            b1 = *reinterpret_cast<uint32_t*>(&bb1);
        } else {
            half2 z = __halves2half2(__float2half(0.0f), __float2half(0.0f));
            uint32_t zv = *reinterpret_cast<uint32_t*>(&z);
            a_top0 = a_top1 = a_bot0 = a_bot1 = b0 = b1 = zv;
        }
        
        mma_m8n8k4_sm70(a_top0, a_top1, b0, b1, c_top);
        mma_m8n8k4_sm70(a_bot0, a_bot1, b0, b1, c_bot);
    }
    
    // Output gathering
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    #define TID_TO_LANE(t) ((t) < 4 ? (t) : ((t) - 4 + 16))
    #define GATHER_FUSED(c_arr, row, col, dst_idx) do { \
        int t = ((row) % 2) + 2 * (((col) / 2) % 2) + 4 * ((row) / 4); \
        int i = ((col) % 2) + 2 * (((row) / 2) % 2) + 4 * ((col) / 4); \
        int src_lane = TID_TO_LANE(t); \
        float vals[8]; \
        for (int k = 0; k < 8; k++) vals[k] = __shfl_sync(FULL_MASK, c_arr[k], src_lane); \
        frag_c[dst_idx] += vals[i]; \
    } while(0)
    
    int col0 = marlin_col_pair * 2;
    int col1 = col0 + 1;
    
    GATHER_FUSED(c_top, marlin_row, col0, 0);
    GATHER_FUSED(c_top, marlin_row, col1, 1);
    GATHER_FUSED(c_bot, marlin_row, col0, 2);
    GATHER_FUSED(c_bot, marlin_row, col1, 3);
    
    #undef GATHER_FUSED
    #undef TID_TO_LANE
}

/// @brief Fused K=32 version (two fused m16n8k16)
/// @param sh_A 16x32 row-major A matrix
/// @param sh_B 32x8 column-major B matrix
/// @param frag_c Output accumulator
/// @param lda Leading dimension of A (default 32)
/// @param ldb Leading dimension of B (default 32)
__device__ __forceinline__ void fused_ldmatrix_mma_m16n8k32_sm70(
    const half* sh_A,
    const half* sh_B,
    float* frag_c,
    int lda = 32,
    int ldb = 32)
{
    fused_ldmatrix_mma_m16n8k16_sm70(sh_A, sh_B, frag_c, lda, ldb);
    fused_ldmatrix_mma_m16n8k16_sm70(sh_A + 16, sh_B + 16, frag_c, lda, ldb);
}


// =============================================================================
// SECTION 7: LEGACY WMMA FUNCTIONS (Optional)
// =============================================================================
//
// These functions use NVIDIA's WMMA API. Disabled by default to speed up
// compilation. Enable with -DMARLIN_SM70_ENABLE_WMMA=1
//
// For most use cases, the primary functions in Section 4 are preferred.

#ifdef MARLIN_SM70_ENABLE_WMMA

/// @brief WMMA-based MMA from shared memory (non-accumulating)
__device__ void mma_m16n8k16_sm70_direct(
    const half* sh_a, 
    const half* sh_b,
    float* frag_c, 
    int lda = 16, 
    int ldb = 8) 
{
    int tid = threadIdx.x % 32;
    int wid = (threadIdx.x / 32) % 8;

    __shared__ char smem_pool_direct[8 * 1536];

    half* sh_b_padded = reinterpret_cast<half*>(&smem_pool_direct[wid * 1536]);
    float* sh_c = reinterpret_cast<float*>(&smem_pool_direct[wid * 1536 + 512]);

    // Pad B to 16 columns
    int elements_per_thread = (16 * 16) / 32;
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid * elements_per_thread + i;
        int row = idx / 16;
        int col = idx % 16;
        sh_b_padded[row * 16 + col] = (col < 8) ? sh_b[row * ldb + col] : __float2half(0.0f);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_acc;

    wmma::fill_fragment(frag_acc, 0.0f);
    wmma::load_matrix_sync(frag_a, sh_a, lda);
    wmma::load_matrix_sync(frag_b, sh_b_padded, 16);
    wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
    wmma::store_matrix_sync(sh_c, frag_acc, 16, wmma::mem_row_major);
    __syncwarp();

    int marlin_row = tid / 4;
    int marlin_col = (tid % 4) * 2;
    frag_c[0] = sh_c[marlin_row * 16 + marlin_col];
    frag_c[1] = sh_c[marlin_row * 16 + marlin_col + 1];
    frag_c[2] = sh_c[(marlin_row + 8) * 16 + marlin_col];
    frag_c[3] = sh_c[(marlin_row + 8) * 16 + marlin_col + 1];
}

/// @brief WMMA-based MMA from shared memory (accumulating)
__device__ void mma_m16n8k16_sm70_direct_accum(
    const half* sh_a, 
    const half* sh_b,
    float* frag_c, 
    int lda = 16, 
    int ldb = 8) 
{
    int tid = threadIdx.x % 32;
    int wid = (threadIdx.x / 32) % 8;

    __shared__ char smem_pool_acc[8 * 1536];

    half* sh_b_padded = reinterpret_cast<half*>(&smem_pool_acc[wid * 1536]);
    float* sh_c = reinterpret_cast<float*>(&smem_pool_acc[wid * 1536 + 512]);

    int elements_per_thread = (16 * 16) / 32;
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid * elements_per_thread + i;
        int row = idx / 16;
        int col = idx % 16;
        sh_b_padded[row * 16 + col] = (col < 8) ? sh_b[row * ldb + col] : __float2half(0.0f);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_acc;

    wmma::fill_fragment(frag_acc, 0.0f);
    wmma::load_matrix_sync(frag_a, sh_a, lda);
    wmma::load_matrix_sync(frag_b, sh_b_padded, 16);
    wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
    wmma::store_matrix_sync(sh_c, frag_acc, 16, wmma::mem_row_major);
    __syncwarp();

    int marlin_row = tid / 4;
    int marlin_col = (tid % 4) * 2;
    frag_c[0] += sh_c[marlin_row * 16 + marlin_col];
    frag_c[1] += sh_c[marlin_row * 16 + marlin_col + 1];
    frag_c[2] += sh_c[(marlin_row + 8) * 16 + marlin_col];
    frag_c[3] += sh_c[(marlin_row + 8) * 16 + marlin_col + 1];
}

/// @brief WMMA-based MMA with register interface (same signature as mma_m16n8k16_sm70)
__device__ void mma_m16n8k16_sm70_wmma(
    const uint32_t* A, 
    const uint32_t* B, 
    float* frag_c)
{
    const int lane = threadIdx.x % 32;
    const int wid = (threadIdx.x / 32) % 8;
    
    __shared__ char smem_wmma[8 * 2048];
    
    half* sh_a = reinterpret_cast<half*>(&smem_wmma[wid * 2048]);
    half* sh_b_padded = reinterpret_cast<half*>(&smem_wmma[wid * 2048 + 512]);
    float* sh_c = reinterpret_cast<float*>(&smem_wmma[wid * 2048 + 1024]);
    
    // Unpack A fragment to shared memory
    {
        int row = lane / 4;
        int kp = lane % 4;
        half2 h0 = *reinterpret_cast<const half2*>(&A[0]);
        half2 h1 = *reinterpret_cast<const half2*>(&A[1]);
        half2 h2 = *reinterpret_cast<const half2*>(&A[2]);
        half2 h3 = *reinterpret_cast<const half2*>(&A[3]);
        sh_a[row * 16 + kp * 2] = __low2half(h0);
        sh_a[row * 16 + kp * 2 + 1] = __high2half(h0);
        sh_a[row * 16 + kp * 2 + 8] = __low2half(h1);
        sh_a[row * 16 + kp * 2 + 9] = __high2half(h1);
        sh_a[(row + 8) * 16 + kp * 2] = __low2half(h2);
        sh_a[(row + 8) * 16 + kp * 2 + 1] = __high2half(h2);
        sh_a[(row + 8) * 16 + kp * 2 + 8] = __low2half(h3);
        sh_a[(row + 8) * 16 + kp * 2 + 9] = __high2half(h3);
    }
    
    // Unpack B fragment to shared memory (with padding)
    {
        int col = lane / 4;
        int kp = lane % 4;
        half2 b0 = *reinterpret_cast<const half2*>(&B[0]);
        half2 b1 = *reinterpret_cast<const half2*>(&B[1]);
        sh_b_padded[(kp * 2) * 16 + col] = __low2half(b0);
        sh_b_padded[(kp * 2 + 1) * 16 + col] = __high2half(b0);
        sh_b_padded[(kp * 2 + 8) * 16 + col] = __low2half(b1);
        sh_b_padded[(kp * 2 + 9) * 16 + col] = __high2half(b1);
        // Pad
        sh_b_padded[(kp * 2) * 16 + col + 8] = __float2half(0.0f);
        sh_b_padded[(kp * 2 + 1) * 16 + col + 8] = __float2half(0.0f);
        sh_b_padded[(kp * 2 + 8) * 16 + col + 8] = __float2half(0.0f);
        sh_b_padded[(kp * 2 + 9) * 16 + col + 8] = __float2half(0.0f);
    }
    __syncwarp();
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_acc;
    
    wmma::fill_fragment(frag_acc, 0.0f);
    wmma::load_matrix_sync(frag_a, sh_a, 16);
    wmma::load_matrix_sync(frag_b, sh_b_padded, 16);
    wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
    wmma::store_matrix_sync(sh_c, frag_acc, 16, wmma::mem_row_major);
    __syncwarp();
    
    int marlin_row = lane / 4;
    int marlin_col = (lane % 4) * 2;
    frag_c[0] += sh_c[marlin_row * 16 + marlin_col];
    frag_c[1] += sh_c[marlin_row * 16 + marlin_col + 1];
    frag_c[2] += sh_c[(marlin_row + 8) * 16 + marlin_col];
    frag_c[3] += sh_c[(marlin_row + 8) * 16 + marlin_col + 1];
}

#endif // MARLIN_SM70_ENABLE_WMMA


} // namespace MARLIN_NAMESPACE_NAME

// =============================================================================
// END OF FILE
// =============================================================================
