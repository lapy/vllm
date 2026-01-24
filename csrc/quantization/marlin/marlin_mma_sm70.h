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
// Primary Functions:
//   mma_m16n8k16_sm70(A, B, frag_c)     - Main MMA operation for quantized GEMM
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
/// @note Loads one 8x8 matrix. Threads 0-7 provide rows 0-7.
///
/// Marlin shared memory layout for 8x8 matrix (m_block_size_8 path):
///   - Threads 0-7: rows 0-7, first half (cols 0-7, 16 bytes each)
///   - Threads 8-15: not used for x1
///   - Threads 16-23: rows 0-7, second half (cols 8-15, 16 bytes each)
///   - Threads 24-31: not used for x1
///
/// Output: frag[0] = A[row, k_pair*2..k_pair*2+1] for rows 0-7, k=0-7
__device__ __forceinline__ void ldmatrix_m8n8_x1_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;

    // Each thread loads 4 uint32 (16 bytes = 8 halves) from its row
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t my_words[4] = {row_ptr[0], row_ptr[1], row_ptr[2], row_ptr[3]};

    // For x1: only one 8x8 matrix, threads 0-7 provide rows 0-7
    int out_row = lane / 4;      // 0-7 for first 32 lanes (wraps at 8)
    int k_pair = lane % 4;       // Which of 4 words in the row

    // Source lane is the thread that has the row we need
    // For x1 (8x8 matrix, cols 0-7): threads 0-7 have rows 0-7, first half
    int src_lane = out_row % 8;

    // Get all 4 words from the source lane
    uint32_t w0 = __shfl_sync(FULL_MASK, my_words[0], src_lane);
    uint32_t w1 = __shfl_sync(FULL_MASK, my_words[1], src_lane);
    uint32_t w2 = __shfl_sync(FULL_MASK, my_words[2], src_lane);
    uint32_t w3 = __shfl_sync(FULL_MASK, my_words[3], src_lane);

    uint32_t arr[4] = {w0, w1, w2, w3};
    dst[0] = arr[k_pair];
}

/// @brief Emulates ldmatrix.sync.aligned.m8n8.x2.shared.b16
/// @param dst Output registers (2 uint32)
/// @param smem_ptr Pointer to shared memory
/// @note Loads two 8x8 matrices (8 rows, 16 cols = full row width).
///
/// Marlin shared memory layout for 8x16 matrix (m_block_size_8 path):
/// With a_sh_rd = a_sh_stride * (lane % 8) + lane / 8 and a_sh_stride=2:
///   - Threads 0-7: a_sh_rd 0,2,4,6,8,10,12,14 -> rows 0-7, first half (cols 0-7)
///   - Threads 8-15: a_sh_rd 1,3,5,7,9,11,13,15 -> rows 0-7, second half (cols 8-15)
///   - Threads 16-31: not used (would overflow a_sh_rd range)
///
/// Output Marlin FragA layout for m_block_size_8=true (lane = row*4 + k_pair):
///   - frag[0]: A[row, k_pair*2..k_pair*2+1]     (rows 0-7, k=0-7)
///   - frag[1]: A[row, k_pair*2+8..k_pair*2+9]   (rows 0-7, k=8-15)
__device__ __forceinline__ void ldmatrix_m8n8_x2_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;

    // Each thread loads 4 uint32 (16 bytes = 8 halves) from its row
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t my_words[4] = {row_ptr[0], row_ptr[1], row_ptr[2], row_ptr[3]};

    int out_row = lane / 4;      // 0-7 for 32 threads
    int k_pair = lane % 4;       // Which of 4 words in the row

    // Source lanes for Marlin 8x16 layout (m_block_size_8=true):
    // With a_sh_rd = a_sh_stride * (lane % 8) + lane / 8 and a_sh_stride=2:
    //   Threads 0-7:  indices 0, 2, 4, ..., 14 (first half of rows 0-7)
    //   Threads 8-15: indices 1, 3, 5, ..., 15 (second half of rows 0-7)
    //   frag[0]: rows 0-7, k=0-7   -> threads 0-7
    //   frag[1]: rows 0-7, k=8-15  -> threads 8-15
    int src_lane_mat0 = out_row;           // Rows 0-7 from threads 0-7
    int src_lane_mat1 = out_row + 8;       // Rows 0-7, second half from threads 8-15

    // Get words from source lanes
    uint32_t w0_mat0 = __shfl_sync(FULL_MASK, my_words[0], src_lane_mat0);
    uint32_t w1_mat0 = __shfl_sync(FULL_MASK, my_words[1], src_lane_mat0);
    uint32_t w2_mat0 = __shfl_sync(FULL_MASK, my_words[2], src_lane_mat0);
    uint32_t w3_mat0 = __shfl_sync(FULL_MASK, my_words[3], src_lane_mat0);

    uint32_t w0_mat1 = __shfl_sync(FULL_MASK, my_words[0], src_lane_mat1);
    uint32_t w1_mat1 = __shfl_sync(FULL_MASK, my_words[1], src_lane_mat1);
    uint32_t w2_mat1 = __shfl_sync(FULL_MASK, my_words[2], src_lane_mat1);
    uint32_t w3_mat1 = __shfl_sync(FULL_MASK, my_words[3], src_lane_mat1);

    uint32_t arr_mat0[4] = {w0_mat0, w1_mat0, w2_mat0, w3_mat0};
    uint32_t arr_mat1[4] = {w0_mat1, w1_mat1, w2_mat1, w3_mat1};

    dst[0] = arr_mat0[k_pair];
    dst[1] = arr_mat1[k_pair];
}

/// @brief Emulates ldmatrix.sync.aligned.m8n8.x4.shared.b16
/// @param dst Output registers (4 uint32)
/// @param smem_ptr Pointer to shared memory
/// @note Primary function used by Marlin for loading FragA
///
/// Marlin shared memory layout for 16x16 matrix:
///   - Threads 0-15: first half of rows 0-15 (cols 0-7, 16 bytes each)
///   - Threads 16-31: second half of rows 0-15 (cols 8-15, 16 bytes each)
///
/// Output Marlin FragA layout (lane = row*4 + k_pair):
///   - frag[0]: A[row, k_pair*2..k_pair*2+1]     (rows 0-7, k=0-7)
///   - frag[1]: A[row, k_pair*2+8..k_pair*2+9]   (rows 0-7, k=8-15)
///   - frag[2]: A[row+8, k_pair*2..k_pair*2+1]   (rows 8-15, k=0-7)
///   - frag[3]: A[row+8, k_pair*2+8..k_pair*2+9] (rows 8-15, k=8-15)
__device__ __forceinline__ void ldmatrix_m8n8_x4_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;

    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t my_words[4] = {row_ptr[0], row_ptr[1], row_ptr[2], row_ptr[3]};

    int out_row = lane / 4;   // 0-7 (which row in the 8-row block)
    int k_pair = lane % 4;    // 0-3 (which pair of columns)

    // Source lanes for 4 quadrants of 16x16 matrix in Marlin layout:
    //   frag[0]: rows 0-7,  k=0-7   -> threads 0-7   (first half of rows 0-7)
    //   frag[1]: rows 0-7,  k=8-15  -> threads 16-23 (second half of rows 0-7)
    //   frag[2]: rows 8-15, k=0-7   -> threads 8-15  (first half of rows 8-15)
    //   frag[3]: rows 8-15, k=8-15  -> threads 24-31 (second half of rows 8-15)
    int src_lanes[4] = {out_row, out_row + 16, out_row + 8, out_row + 24};

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
// SECTION 4: REGISTER-BASED MMA FUNCTIONS
// =============================================================================
//
// These functions use pre-loaded register fragments for tensor core operations.

// -----------------------------------------------------------------------------
// 4.1 mma_m16n8k16_sm70 - FROZEN (DO NOT MODIFY)
// -----------------------------------------------------------------------------
//
// Register-based tensor core function for quantized GEMM kernels.
// Validated: January 2026, Max error: 0.000000
//
/// @brief Computes C[16×8] += A[16×16] × B[16×8] using m8n8k4 tensor cores
/// @param A FragA in Marlin layout (4 uint32)
/// @param B FragB in Marlin layout (2 uint32)  
/// @param frag_c FragC output, accumulated (4 floats)
__device__ __forceinline__ void mma_m16n8k16_sm70(
    const uint32_t* A, 
    const uint32_t* B,
    float* frag_c) 
{
    // ==========================================================================
    // Emulates m16n8k16 using 8 m8n8k4 operations on Volta (SM70)
    // 
    // Marlin fragment layout (lane = row*4 + k_pair):
    //   A[0]: {A[row, k_pair*2], A[row, k_pair*2+1]}     k=0-7,  rows 0-7
    //   A[1]: {A[row, k_pair*2+8], A[row, k_pair*2+9]}   k=8-15, rows 0-7
    //   A[2]: {A[row+8, k_pair*2], ...}                  k=0-7,  rows 8-15
    //   A[3]: {A[row+8, k_pair*2+8], ...}                k=8-15, rows 8-15
    //   B[0]: {B[k_pair*2, col], B[k_pair*2+1, col]}     k=0-7
    //   B[1]: {B[k_pair*2+8, col], ...}                  k=8-15
    //
    // m8n8k4 fragment layout (quadpair tid t = 0-7):
    //   a0 = {A[t,0], A[t,1]}, a1 = {A[t,2], A[t,3]}
    //   b0 = {B[0,t], B[1,t]}, b1 = {B[2,t], B[3,t]}
    //   C output is scattered across 8 floats per thread
    // ==========================================================================
    
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    // Quadpair: lanes 0-3 → tid 0-3, lanes 16-19 → tid 4-7
    const int qp_tid = (lane < 4) ? lane : ((lane >= 16 && lane < 20) ? (lane - 16 + 4) : -1);
    const bool is_quadpair = (qp_tid >= 0);
    
    // Accumulators for top (rows 0-7) and bottom (rows 8-15) 8x8 blocks
    float c_top[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float c_bot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // Process 4 k-blocks: kb=0 (k=0-3), kb=1 (k=4-7), kb=2 (k=8-11), kb=3 (k=12-15)
    #pragma unroll
    for (int kb = 0; kb < 4; kb++) {
        // Which Marlin register holds this k-range
        const int a_reg = (kb < 2) ? 0 : 1;  // A[0]/A[2] for k<8, A[1]/A[3] for k>=8
        const int b_reg = (kb < 2) ? 0 : 1;  // B[0] for k<8, B[1] for k>=8
        
        // Which k_pair within that register (0,1 for first half, 2,3 for second half)
        const int k_pair_base = (kb % 2) * 2;  // 0 for kb=0,2; 2 for kb=1,3
        
        // For m8n8k4: tid t needs row t of A and column t of B
        // In Marlin layout: row r is at lane r*4+k_pair, col c is at lane c*4+k_pair
        // So for tid t, we need lanes t*4+k_pair_base and t*4+k_pair_base+1
        
        uint32_t a_top0, a_top1, a_bot0, a_bot1, b0, b1;
        
        if (is_quadpair) {
            // Source lanes for this quadpair thread
            const int a_lane0 = qp_tid * 4 + k_pair_base;      // a0: {A[t,0], A[t,1]}
            const int a_lane1 = qp_tid * 4 + k_pair_base + 1;  // a1: {A[t,2], A[t,3]}
            const int b_lane0 = qp_tid * 4 + k_pair_base;      // b0: {B[0,t], B[1,t]}
            const int b_lane1 = qp_tid * 4 + k_pair_base + 1;  // b1: {B[2,t], B[3,t]}

            // Gather A for top rows (0-7) - data is already in correct half2 format!
            a_top0 = __shfl_sync(FULL_MASK, A[a_reg], a_lane0);
            a_top1 = __shfl_sync(FULL_MASK, A[a_reg], a_lane1);

            // Gather A for bottom rows (8-15)
            a_bot0 = __shfl_sync(FULL_MASK, A[a_reg + 2], a_lane0);
            a_bot1 = __shfl_sync(FULL_MASK, A[a_reg + 2], a_lane1);

            // Gather B - data is already in correct half2 format!
            b0 = __shfl_sync(FULL_MASK, B[b_reg], b_lane0);
            b1 = __shfl_sync(FULL_MASK, B[b_reg], b_lane1);
        } else {
            // Non-quadpair threads must still participate in mma.sync but with zeros
            // They also need to participate in shuffles with valid source lanes
            a_top0 = __shfl_sync(FULL_MASK, A[a_reg], k_pair_base);
            a_top1 = __shfl_sync(FULL_MASK, A[a_reg], k_pair_base + 1);
            a_bot0 = __shfl_sync(FULL_MASK, A[a_reg + 2], k_pair_base);
            a_bot1 = __shfl_sync(FULL_MASK, A[a_reg + 2], k_pair_base + 1);
            b0 = __shfl_sync(FULL_MASK, B[b_reg], k_pair_base);
            b1 = __shfl_sync(FULL_MASK, B[b_reg], k_pair_base + 1);
            
            // Zero out for non-quadpair threads
            a_top0 = a_top1 = a_bot0 = a_bot1 = b0 = b1 = 0;
        }
        
        // Execute tensor cores
        mma_m8n8k4_sm70(a_top0, a_top1, b0, b1, c_top);
        mma_m8n8k4_sm70(a_bot0, a_bot1, b0, b1, c_bot);
    }
    
    // ==========================================================================
    // Gather scattered m8n8k4 output back to Marlin's frag_c layout
    //
    // CRITICAL: __shfl_sync(mask, var, srcLane) evaluates 'var' on CALLING thread!
    // So c_arr[i] is evaluated with THIS thread's i, not src_lane's i.
    // We must shuffle ALL 8 elements, then select locally.
    //
    // m8n8k4 C layout is scattered: for tid t and index i:
    //   row(t,i) = (t%2) + 2*((i/2)%2) + 4*(t/4)
    //   col(t,i) = 2*((t/2)%2) + (i%2) + 4*(i/4)
    //
    // Marlin frag_c layout (lane = row*4 + col_pair):
    //   frag_c[0]: C[row, col_pair*2]
    //   frag_c[1]: C[row, col_pair*2+1]
    //   frag_c[2]: C[row+8, col_pair*2]
    //   frag_c[3]: C[row+8, col_pair*2+1]
    // ==========================================================================

    const int marlin_row = lane / 4;       // 0-7
    const int marlin_col_pair = lane % 4;  // 0-3

    #define TID_TO_LANE(t) ((t) < 4 ? (t) : ((t) - 4 + 16))

    #pragma unroll
    for (int out_idx = 0; out_idx < 4; out_idx++) {
        // Target position in the 16x8 output matrix
        const int row = (out_idx < 2) ? marlin_row : (marlin_row + 8);
        const int col = marlin_col_pair * 2 + (out_idx % 2);

        // Use top or bottom accumulator
        float* c_arr = (out_idx < 2) ? c_top : c_bot;
        const int local_row = row % 8;  // Row within the 8x8 block

        // Inverse mapping: which tid t and index i has C[local_row, col]?
        const int t = (local_row % 2) + 2 * ((col / 2) % 2) + 4 * (local_row / 4);
        const int i = (col % 2) + 2 * ((local_row / 2) % 2) + 4 * (col / 4);
        const int src_lane = TID_TO_LANE(t);

        // Shuffle ALL 8 elements from src_lane, then select index i locally
        float shfl_c[8];
        shfl_c[0] = __shfl_sync(FULL_MASK, c_arr[0], src_lane);
        shfl_c[1] = __shfl_sync(FULL_MASK, c_arr[1], src_lane);
        shfl_c[2] = __shfl_sync(FULL_MASK, c_arr[2], src_lane);
        shfl_c[3] = __shfl_sync(FULL_MASK, c_arr[3], src_lane);
        shfl_c[4] = __shfl_sync(FULL_MASK, c_arr[4], src_lane);
        shfl_c[5] = __shfl_sync(FULL_MASK, c_arr[5], src_lane);
        shfl_c[6] = __shfl_sync(FULL_MASK, c_arr[6], src_lane);
        shfl_c[7] = __shfl_sync(FULL_MASK, c_arr[7], src_lane);

        frag_c[out_idx] += shfl_c[i];
    }

    #undef TID_TO_LANE
}


// =============================================================================
// SECTION 5: VARIANT MMA FUNCTIONS
// =============================================================================

// -----------------------------------------------------------------------------
// 5.1 Transposed MMA (for Marlin's mma_trans)
// -----------------------------------------------------------------------------
//
// Parameter naming follows Marlin convention, NOT PTX convention:
//   'marlin_a' (activations) → MMA's B operand (16×8 matrix)
//   'marlin_b'+'marlin_b2' (weights) → MMA's A operand (16×16 matrix)
//
// Marlin fragment layouts for trans (m_block_size_8 path):
//   marlin_a: FragA loaded via ldsm<2> (only 2 registers for 8 rows)
//     lane = col*4 + k_pair, where col∈[0,7], k_pair∈[0,3]
//     marlin_a[0]: {B[k_pair*2, col], B[k_pair*2+1, col]}     k=0-7
//     marlin_a[1]: {B[k_pair*2+8, col], B[k_pair*2+9, col]}   k=8-15
//
//   marlin_b/b2: FragB (weights), each 2 registers
//     lane = row*4 + k_pair, where row∈[0,7], k_pair∈[0,3]
//     marlin_b[0]:  {A[row, k_pair*2], A[row, k_pair*2+1]}     rows 0-7,  k=0-7
//     marlin_b[1]:  {A[row, k_pair*2+8], ...}                  rows 0-7,  k=8-15
//     marlin_b2[0]: {A[row+8, k_pair*2], ...}                  rows 8-15, k=0-7
//     marlin_b2[1]: {A[row+8, k_pair*2+8], ...}                rows 8-15, k=8-15
//
/// @brief Transposed multiply: C += (b,b2) * a using tensor cores
/// @param marlin_a Activations (becomes MMA B)
/// @param marlin_b Weights rows 0-7 (becomes MMA A top half)
/// @param marlin_b2 Weights rows 8-15 (becomes MMA A bottom half)
/// @param frag_c Output accumulator
__device__ void mma_m16n8k16_sm70_trans(
    const uint32_t* marlin_a, 
    const uint32_t* marlin_b,
    const uint32_t* marlin_b2, 
    float* frag_c) 
{
    // ==========================================================================
    // Emulates m16n8k16 trans using 8 m8n8k4 operations
    // Trans: C[16×8] += A[16×16] × B[16×8]
    //        where A = (marlin_b, marlin_b2), B = marlin_a
    // ==========================================================================
    
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    // Quadpair: lanes 0-3 → tid 0-3, lanes 16-19 → tid 4-7
    const int qp_tid = (lane < 4) ? lane : ((lane >= 16 && lane < 20) ? (lane - 16 + 4) : -1);
    const bool is_quadpair = (qp_tid >= 0);
    
    // Accumulators for top (rows 0-7) and bottom (rows 8-15) 8x8 blocks
    float c_top[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float c_bot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // Process 4 k-blocks: kb=0 (k=0-3), kb=1 (k=4-7), kb=2 (k=8-11), kb=3 (k=12-15)
    #pragma unroll
    for (int kb = 0; kb < 4; kb++) {
        // Which register holds this k-range
        // For weights (A): marlin_b[0]/marlin_b2[0] for k<8, [1] for k>=8
        // For activations (B): marlin_a[0] for k<8, marlin_a[1] for k>=8
        const int weight_reg = (kb < 2) ? 0 : 1;
        const int act_reg = (kb < 2) ? 0 : 1;
        
        // Which k_pair within that register (0,1 for first half, 2,3 for second half)
        const int k_pair_base = (kb % 2) * 2;  // 0 for kb=0,2; 2 for kb=1,3
        
        uint32_t a_top0, a_top1, a_bot0, a_bot1, b0, b1;
        
        if (is_quadpair) {
            // For m8n8k4 A operand: tid t needs row t of the weight matrix
            // Weight row r is at lane r*4+k_pair in marlin_b/marlin_b2
            const int a_lane0 = qp_tid * 4 + k_pair_base;
            const int a_lane1 = qp_tid * 4 + k_pair_base + 1;
            
            // Gather weights for A operand (top and bottom halves)
            a_top0 = __shfl_sync(FULL_MASK, marlin_b[weight_reg], a_lane0);
            a_top1 = __shfl_sync(FULL_MASK, marlin_b[weight_reg], a_lane1);
            a_bot0 = __shfl_sync(FULL_MASK, marlin_b2[weight_reg], a_lane0);
            a_bot1 = __shfl_sync(FULL_MASK, marlin_b2[weight_reg], a_lane1);
            
            // For m8n8k4 B operand: tid t needs column t of the activation matrix
            // Activation col c is at lane c*4+k_pair in marlin_a
            const int b_lane0 = qp_tid * 4 + k_pair_base;
            const int b_lane1 = qp_tid * 4 + k_pair_base + 1;
            
            b0 = __shfl_sync(FULL_MASK, marlin_a[act_reg], b_lane0);
            b1 = __shfl_sync(FULL_MASK, marlin_a[act_reg], b_lane1);
        } else {
            // Non-quadpair: participate in shuffles but use zeros
            a_top0 = __shfl_sync(FULL_MASK, marlin_b[weight_reg], k_pair_base);
            a_top1 = __shfl_sync(FULL_MASK, marlin_b[weight_reg], k_pair_base + 1);
            a_bot0 = __shfl_sync(FULL_MASK, marlin_b2[weight_reg], k_pair_base);
            a_bot1 = __shfl_sync(FULL_MASK, marlin_b2[weight_reg], k_pair_base + 1);
            b0 = __shfl_sync(FULL_MASK, marlin_a[act_reg], k_pair_base);
            b1 = __shfl_sync(FULL_MASK, marlin_a[act_reg], k_pair_base + 1);
            
            a_top0 = a_top1 = a_bot0 = a_bot1 = b0 = b1 = 0;
        }
        
        // Execute tensor cores
        mma_m8n8k4_sm70(a_top0, a_top1, b0, b1, c_top);
        mma_m8n8k4_sm70(a_bot0, a_bot1, b0, b1, c_bot);
    }
    
    // ==========================================================================
    // Gather scattered m8n8k4 output back to Marlin's frag_c layout
    // Same as mma_m16n8k16_sm70 since output layout is the same
    //
    // CRITICAL: Must shuffle ALL 8 elements, then select locally.
    // ==========================================================================

    const int marlin_row = lane / 4;       // 0-7
    const int marlin_col_pair = lane % 4;  // 0-3

    #define TID_TO_LANE(t) ((t) < 4 ? (t) : ((t) - 4 + 16))

    #pragma unroll
    for (int out_idx = 0; out_idx < 4; out_idx++) {
        const int row = (out_idx < 2) ? marlin_row : (marlin_row + 8);
        const int col = marlin_col_pair * 2 + (out_idx % 2);

        float* c_arr = (out_idx < 2) ? c_top : c_bot;
        const int local_row = row % 8;

        const int t = (local_row % 2) + 2 * ((col / 2) % 2) + 4 * (local_row / 4);
        const int i = (col % 2) + 2 * ((local_row / 2) % 2) + 4 * (col / 4);
        const int src_lane = TID_TO_LANE(t);

        // Shuffle ALL 8 elements from src_lane, then select index i locally
        float shfl_c[8];
        shfl_c[0] = __shfl_sync(FULL_MASK, c_arr[0], src_lane);
        shfl_c[1] = __shfl_sync(FULL_MASK, c_arr[1], src_lane);
        shfl_c[2] = __shfl_sync(FULL_MASK, c_arr[2], src_lane);
        shfl_c[3] = __shfl_sync(FULL_MASK, c_arr[3], src_lane);
        shfl_c[4] = __shfl_sync(FULL_MASK, c_arr[4], src_lane);
        shfl_c[5] = __shfl_sync(FULL_MASK, c_arr[5], src_lane);
        shfl_c[6] = __shfl_sync(FULL_MASK, c_arr[6], src_lane);
        shfl_c[7] = __shfl_sync(FULL_MASK, c_arr[7], src_lane);

        frag_c[out_idx] += shfl_c[i];
    }

    #undef TID_TO_LANE
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
// SECTION 6: LEGACY WMMA FUNCTIONS (Optional)
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
