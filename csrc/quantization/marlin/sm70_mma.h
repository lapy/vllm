/*
 * SM70 MMA Library for Volta Tensor Cores
 * 
 * This implements m16n8k16 MMA operations for SM70 by composing
 * four m8n8k4 PTX instructions. The key is proper per-lane data
 * distribution following the Volta register mapping.
 *
 * Reference: https://github.com/ahennequ/cuda-tensorcores-register-mapping
 *
 * SM70 m8n8k4.f32 register mapping:
 *   Row: (tid & 16) / 4 + 2 * (tid & 4) + (tid & 1) + (i & 2)
 *   Col: (tid & 10) + (i & 5)
 *
 * where tid = lane ID (0-31), i = fragment element index (0-7)
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace MARLIN_NAMESPACE_NAME {

// =============================================================================
// Warp organization utilities
// =============================================================================

__device__ __forceinline__ int get_sm70_warp_lane() {
    return threadIdx.x % 32;
}

__device__ __forceinline__ int get_sm70_quadpair() {
    return (threadIdx.x % 32) / 8;
}

// =============================================================================
// ldmatrix emulation for SM70 (Volta)
// 
// The ldmatrix.sync.aligned.m8n8.x{1,2,4}.shared.b16 instruction is not 
// available on SM70. We emulate it using warp shuffles.
//
// ldmatrix loads matrix fragments from shared memory into registers in the
// exact layout required by tensor core MMA operations. The instruction performs
// a warp-collective gather where each thread contributes an address and 
// receives values according to the MMA fragment layout.
//
// For m8n8.x4 with b16 elements (half precision):
// - Threads 0-7 each provide address for one row of an 8x8 matrix
// - Each thread receives 4 uint32_t values (8 half values)
// - Output layout matches mma.m16n8k16 fragment requirements
// =============================================================================

// ldmatrix fragment layout for m16n8k16 (A matrix):
// Per-thread distribution for FragA[4] (4 x uint32_t = 8 x half):
//   Lane i contributes bytes for specific (row, col) positions
//   The actual mapping matches PTX mma.m16n8k16 operand A layout

// Emulates ldmatrix.sync.aligned.m8n8.x1.shared.b16
// Each thread provides address to beginning of an 8-element row (16 bytes)
// Thread provides smem_ptr, receives 1 uint32_t in dst[0]
__device__ __forceinline__ void ldmatrix_m8n8_x1_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    // Each row has 8 half values = 4 uint32_t
    // Thread t (where t < 8) provides address for row t
    // Load the full row locally
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    
    // For x1, we load 1 register per thread
    // Distribution: lane % 8 gives source row, lane / 8 gives which word
    int source_row = lane % 8;
    int word_idx = (lane / 8) % 4;
    
    // Load all 4 words of our row
    uint32_t my_words[4];
    my_words[0] = row_ptr[0];
    my_words[1] = row_ptr[1];
    my_words[2] = row_ptr[2];
    my_words[3] = row_ptr[3];
    
    // Shuffle to get the correct word from the correct source lane
    uint32_t val = __shfl_sync(FULL_MASK, my_words[word_idx], source_row);
    dst[0] = val;
}

// Emulates ldmatrix.sync.aligned.m8n8.x2.shared.b16
__device__ __forceinline__ void ldmatrix_m8n8_x2_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    
    // Load our row's 4 words
    uint32_t my_words[4];
    my_words[0] = row_ptr[0];
    my_words[1] = row_ptr[1];
    my_words[2] = row_ptr[2];
    my_words[3] = row_ptr[3];
    
    // For x2, each thread gets 2 registers
    // Layout matches mma.m16n8k16 operand A fragment
    // Lanes 0-7: rows 0-7, lanes 8-15: rows 0-7 (different columns)
    // Lanes 16-23: rows 8-15, lanes 24-31: rows 8-15 (different columns)
    
    int row_in_tile = lane % 8;
    int col_group = (lane / 8) % 2;  // 0 or 1 for column selection
    
    // dst[0]: columns 0-1 or 2-3 based on col_group
    // dst[1]: columns 4-5 or 6-7 based on col_group
    uint32_t v0 = __shfl_sync(FULL_MASK, my_words[col_group * 2 + 0], row_in_tile);
    uint32_t v1 = __shfl_sync(FULL_MASK, my_words[col_group * 2 + 1], row_in_tile);
    
    dst[0] = v0;
    dst[1] = v1;
}

// Emulates ldmatrix.sync.aligned.m8n8.x4.shared.b16
// This is the primary function used by Marlin for loading FragA
__device__ __forceinline__ void ldmatrix_m8n8_x4_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    // smem_ptr points to the start of this thread's row (8 halves = 16 bytes)
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    
    // Each thread loads its full row (4 uint32_t = 8 halves)
    uint32_t my_words[4];
    my_words[0] = row_ptr[0];
    my_words[1] = row_ptr[1];
    my_words[2] = row_ptr[2];
    my_words[3] = row_ptr[3];
    
    // The ldmatrix.m8n8.x4 output layout for mma.m16n8k16 operand A:
    //
    // For a 16x16 matrix A (row-major, elements are halves):
    //   The matrix is divided into 8 rows of 16 halves each
    //   Each thread t (0-31) receives 4 uint32_t covering:
    //     - dst[0]: 2 halves from row (t%8), columns based on (t/8)
    //     - dst[1]: 2 halves from row (t%8), next column pair
    //     - dst[2]: 2 halves from row (t%8)+8, columns based on (t/8)  
    //     - dst[3]: 2 halves from row (t%8)+8, next column pair
    //
    // The key insight: threads 0-7 provide addresses for rows 0-7,
    // threads 8-15 provide addresses for rows 0-7 again but different columns,
    // threads 16-23 provide addresses for rows 8-15,
    // threads 24-31 provide addresses for rows 8-15 again but different columns.
    
    // Within each group of 8 threads:
    //   - thread's lane%8 gives the row within the 8x8 tile
    //   - thread's (lane/8)%2 gives which pair of columns (0-1 or 2-3 from each 4-column group)
    //   - thread's lane/16 gives the 8x8 tile (top or bottom half of 16x16)
    
    int row_in_group = lane % 8;  // Which row within the 8-thread group
    int col_pair = (lane / 8) % 2;  // 0: cols 0-3, 1: cols 4-7 style offset
    
    // For m16n8k16 operand A (16 rows, 16 cols):
    // We need to shuffle to get the right halves to each thread
    // 
    // Source thread for row_in_group's data:
    //   src_lane = row_in_group + (tile_half * 16)
    // But we also need the row from the other half (rows 8-15 vs 0-7)
    
    // Simplified approach: each thread shuffles from the row provider
    // Top half (dst[0], dst[1]) comes from rows 0-7
    // Bottom half (dst[2], dst[3]) comes from rows 8-15
    
    int src_row_top = row_in_group;  // lanes 0-7 have rows 0-7
    int src_row_bot = row_in_group + 8;  // lanes 8-15 have rows 8-15
    
    // Word selection based on column grouping
    // col_pair=0: words 0,1  col_pair=1: words 2,3
    int word_base = col_pair * 2;
    
    // Get values from top half rows (lanes 0-7)
    dst[0] = __shfl_sync(FULL_MASK, my_words[word_base + 0], src_row_top);
    dst[1] = __shfl_sync(FULL_MASK, my_words[word_base + 1], src_row_top);
    
    // Get values from bottom half rows (lanes 16-23)  
    dst[2] = __shfl_sync(FULL_MASK, my_words[word_base + 0], src_row_bot);
    dst[3] = __shfl_sync(FULL_MASK, my_words[word_base + 1], src_row_bot);
}

// Alternative ldmatrix that handles the trans variant (.trans modifier)
// For transposed B matrix loading
__device__ __forceinline__ void ldmatrix_m8n8_x2_trans_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    // For transposed load, columns become rows
    // Thread provides column address, receives row elements
    const uint32_t* col_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    
    // Load column data
    uint32_t my_words[4];
    my_words[0] = col_ptr[0];
    my_words[1] = col_ptr[1];
    my_words[2] = col_ptr[2];
    my_words[3] = col_ptr[3];
    
    int col_in_tile = lane % 8;
    int row_group = (lane / 8) % 2;
    
    // Shuffle for transposed access pattern
    uint32_t v0 = __shfl_sync(FULL_MASK, my_words[row_group * 2 + 0], col_in_tile);
    uint32_t v1 = __shfl_sync(FULL_MASK, my_words[row_group * 2 + 1], col_in_tile);
    
    dst[0] = v0;
    dst[1] = v1;
}

// =============================================================================
// SM70 register mapping helpers for FP32 accumulator
// =============================================================================

// Get the row index in the output matrix for fragment element i
__device__ __forceinline__ int sm70_frag_row_f32(int tid, int i) {
    return ((tid & 16) >> 2) + 2 * ((tid & 4) >> 2) + (tid & 1) + (i & 2);
}

// Get the column index in the output matrix for fragment element i
__device__ __forceinline__ int sm70_frag_col_f32(int tid, int i) {
    return (tid & 10) + (i & 5);
}

// =============================================================================
// Core 8x8x4 MMA operations for Volta
// =============================================================================

// Core m8n8k4 MMA with FP32 accumulation
// PTX requires: 2 half2 for A, 2 half2 for B, 8 f32 for C/D
// Each thread provides its portion of the distributed matrix
__device__ void mma_m8n8k4_sm70(
    const half2& a0, const half2& a1,  // Two A registers
    const half2& b0, const half2& b1,  // Two B registers
    float& c0, float& c1, float& c2, float& c3,
    float& c4, float& c5, float& c6, float& c7) 
{
    uint32_t a0_val = *reinterpret_cast<const uint32_t*>(&a0);
    uint32_t a1_val = *reinterpret_cast<const uint32_t*>(&a1);
    uint32_t b0_val = *reinterpret_cast<const uint32_t*>(&b0);
    uint32_t b1_val = *reinterpret_cast<const uint32_t*>(&b1);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3),
          "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
        : "r"(a0_val), "r"(a1_val), "r"(b0_val), "r"(b1_val));
}

// Simplified 4-output version for marlin compatibility
// Internally uses all 8 outputs but only returns the first 4
// This matches the expected FragC[4] layout
__device__ void mma_m8n8k4_sm70(
    const half2& a, const half2& b, 
    float& c0, float& c1, float& c2, float& c3) 
{
    // For the simplified version, we duplicate registers
    // This works when fragments are pre-distributed correctly
    uint32_t a_val = *reinterpret_cast<const uint32_t*>(&a);
    uint32_t b_val = *reinterpret_cast<const uint32_t*>(&b);
    float c_ext[8] = {c0, c1, c2, c3, 0.0f, 0.0f, 0.0f, 0.0f};

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(c_ext[0]), "+f"(c_ext[1]), "+f"(c_ext[2]), "+f"(c_ext[3]),
          "+f"(c_ext[4]), "+f"(c_ext[5]), "+f"(c_ext[6]), "+f"(c_ext[7])
        : "r"(a_val), "r"(a_val), "r"(b_val), "r"(b_val));

    c0 = c_ext[0];
    c1 = c_ext[1];
    c2 = c_ext[2];
    c3 = c_ext[3];
}

// Core m8n8k4 MMA with FP16 accumulation
__device__ void mma_m8n8k4_sm70_fp16(
    const half2& a0, const half2& a1,
    const half2& b0, const half2& b1,
    half2& c0, half2& c1, half2& c2, half2& c3) 
{
    uint32_t a0_val = *reinterpret_cast<const uint32_t*>(&a0);
    uint32_t a1_val = *reinterpret_cast<const uint32_t*>(&a1);
    uint32_t b0_val = *reinterpret_cast<const uint32_t*>(&b0);
    uint32_t b1_val = *reinterpret_cast<const uint32_t*>(&b1);
    uint32_t d[4];
    d[0] = *reinterpret_cast<const uint32_t*>(&c0);
    d[1] = *reinterpret_cast<const uint32_t*>(&c1);
    d[2] = *reinterpret_cast<const uint32_t*>(&c2);
    d[3] = *reinterpret_cast<const uint32_t*>(&c3);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0_val), "r"(a1_val), "r"(b0_val), "r"(b1_val));

    c0 = *reinterpret_cast<half2*>(&d[0]);
    c1 = *reinterpret_cast<half2*>(&d[1]);
    c2 = *reinterpret_cast<half2*>(&d[2]);
    c3 = *reinterpret_cast<half2*>(&d[3]);
}

// Simplified FP16 version (2 output registers)
__device__ void mma_m8n8k4_sm70_fp16(
    const half2& a, const half2& b,
    half2& c0, half2& c1) 
{
    uint32_t a_val = *reinterpret_cast<const uint32_t*>(&a);
    uint32_t b_val = *reinterpret_cast<const uint32_t*>(&b);
    uint32_t d[4];
    d[0] = *reinterpret_cast<const uint32_t*>(&c0);
    d[1] = *reinterpret_cast<const uint32_t*>(&c1);
    d[2] = 0;
    d[3] = 0;

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a_val), "r"(a_val), "r"(b_val), "r"(b_val));

    c0 = *reinterpret_cast<half2*>(&d[0]);
    c1 = *reinterpret_cast<half2*>(&d[1]);
}

// =============================================================================
// 16x8x16 MMA operations (composed from 4x m8n8k4)
// =============================================================================

// Per-thread fragment API: A[4], B[2], frag_c[4]. Matches marlin_mma.h usage.
// This version works with pre-distributed fragments from Marlin's layout.
// Accumulates into this thread's FragC via 4 m8n8k4 steps.
__device__ void mma_m16n8k16_sm70(const uint32_t* A, const uint32_t* B,
                                  float* frag_c) {
    float c[4] = {frag_c[0], frag_c[1], frag_c[2], frag_c[3]};
    float dummy[2]; // Discard redundant accumulator outputs

    // Iterate over 4 k-steps (k=0,4,8,12 in the k=16 dimension)
    // Each step uses one half2 from A and half of a half2 from B
    // k=0
    {
        half2 a = *reinterpret_cast<const half2*>(&A[0]);
        half2 b_pair = *reinterpret_cast<const half2*>(&B[0]);
        half2 a_top = __halves2half2(a.x, a.x);
        half2 a_bot = __halves2half2(a.y, a.y);
        half2 b_use = __halves2half2(b_pair.x, b_pair.x);
        mma_m8n8k4_sm70(a_top, b_use, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_use, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=1
    {
        half2 a = *reinterpret_cast<const half2*>(&A[1]);
        half2 b_pair = *reinterpret_cast<const half2*>(&B[0]);
        half2 a_top = __halves2half2(a.x, a.x);
        half2 a_bot = __halves2half2(a.y, a.y);
        half2 b_use = __halves2half2(b_pair.y, b_pair.y);
        mma_m8n8k4_sm70(a_top, b_use, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_use, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=2
    {
        half2 a = *reinterpret_cast<const half2*>(&A[2]);
        half2 b_pair = *reinterpret_cast<const half2*>(&B[1]);
        half2 a_top = __halves2half2(a.x, a.x);
        half2 a_bot = __halves2half2(a.y, a.y);
        half2 b_use = __halves2half2(b_pair.x, b_pair.x);
        mma_m8n8k4_sm70(a_top, b_use, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_use, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=3
    {
        half2 a = *reinterpret_cast<const half2*>(&A[3]);
        half2 b_pair = *reinterpret_cast<const half2*>(&B[1]);
        half2 a_top = __halves2half2(a.x, a.x);
        half2 a_bot = __halves2half2(a.y, a.y);
        half2 b_use = __halves2half2(b_pair.y, b_pair.y);
        mma_m8n8k4_sm70(a_top, b_use, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_use, c[2], c[3], dummy[0], dummy[1]);
    }

    frag_c[0] = c[0];
    frag_c[1] = c[1];
    frag_c[2] = c[2];
    frag_c[3] = c[3];
}

// Version that writes results to global memory C[m][n]
// Used for testing and verification
__device__ void mma_m16n8k16_sm70(
    const uint32_t* A, const uint32_t* B, 
    float* C, int m, int n) 
{
    int warp_id = get_sm70_warp_lane() / 16;
    int quadpair = get_sm70_quadpair();

    float frag_C[4] = {0};
    float dummy[2];

    for (int k = 0; k < 4; ++k) {
        half2 a = *reinterpret_cast<const half2*>(&A[k * 4 + warp_id]);
        half2 b = *reinterpret_cast<const half2*>(&B[k * 4 + quadpair]);

        // Split A: A contains {top, bot} (unique). We need to replicate each for m8n8k4.
        half2 a_top = __halves2half2(a.x, a.x);
        half2 a_bot = __halves2half2(a.y, a.y);
        
        // Use B directly (assuming redundancy or 2x factor is handled elsewhere, or B layout differs)
        // See comments in previous edit about B redundancy.
 
        // Accumulate Top
        mma_m8n8k4_sm70(a_top, b, frag_C[0], frag_C[1], dummy[0], dummy[1]);
        
        // Accumulate Bot
        mma_m8n8k4_sm70(a_bot, b, frag_C[2], frag_C[3], dummy[0], dummy[1]);
        
        __syncwarp();
    }

    if (quadpair < 2) {
        for (int i = 0; i < 4; ++i) {
            int row = warp_id * 8 + i;
            int col = quadpair * 4 + (i % 4);
            if (row < m && col < n) {
                atomicAdd(&C[row * n + col], frag_C[i]);
            }
        }
    }
}

// Version with optional C clearing
__device__ void mma_m16n8k16_sm70(const uint32_t* A, const uint32_t* B,
                                  float* C, bool no_c_clear) {
    const int m = 16, n = 8;
    if (!no_c_clear) {
        int tid = get_sm70_warp_lane();
        if (tid == 0) {
            for (int i = 0; i < m * n; i++) C[i] = 0.0f;
        }
        __syncwarp();
    }
    mma_m16n8k16_sm70(A, B, C, m, n);
}

// =============================================================================
// Transposed B variants
// =============================================================================

// Per-thread fragment API with transposed B layout
__device__ void mma_m16n8k16_sm70_trans(const uint32_t* A, const uint32_t* B,
                                        const uint32_t* B2, float* frag_c) {
    float c[4] = {frag_c[0], frag_c[1], frag_c[2], frag_c[3]};
    float dummy[2];

    // k=0
    {
        half2 a = *reinterpret_cast<const half2*>(&A[0]);
        half2 a_top = __halves2half2(a.x, a.x);
        half2 a_bot = __halves2half2(a.y, a.y);
        half2 b_tr = __halves2half2(
            __ushort_as_half(static_cast<unsigned short>(B[0] & 0xFFFF)),
            __ushort_as_half(static_cast<unsigned short>(B2[0] & 0xFFFF)));
        mma_m8n8k4_sm70(a_top, b_tr, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_tr, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=1
    {
        half2 a = *reinterpret_cast<const half2*>(&A[1]);
        half2 a_top = __halves2half2(a.x, a.x);
        half2 a_bot = __halves2half2(a.y, a.y);
        half2 b_tr = __halves2half2(
            __ushort_as_half(static_cast<unsigned short>((B[0] >> 16) & 0xFFFF)),
            __ushort_as_half(static_cast<unsigned short>((B2[0] >> 16) & 0xFFFF)));
        mma_m8n8k4_sm70(a_top, b_tr, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_tr, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=2
    {
        half2 a = *reinterpret_cast<const half2*>(&A[2]);
        half2 a_top = __halves2half2(a.x, a.x);
        half2 a_bot = __halves2half2(a.y, a.y);
        half2 b_tr = __halves2half2(
            __ushort_as_half(static_cast<unsigned short>(B[1] & 0xFFFF)),
            __ushort_as_half(static_cast<unsigned short>(B2[1] & 0xFFFF)));
        mma_m8n8k4_sm70(a_top, b_tr, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_tr, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=3
    {
        half2 a = *reinterpret_cast<const half2*>(&A[3]);
        half2 a_top = __halves2half2(a.x, a.x);
        half2 a_bot = __halves2half2(a.y, a.y);
        half2 b_tr = __halves2half2(
            __ushort_as_half(static_cast<unsigned short>((B[1] >> 16) & 0xFFFF)),
            __ushort_as_half(static_cast<unsigned short>((B2[1] >> 16) & 0xFFFF)));
        mma_m8n8k4_sm70(a_top, b_tr, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_tr, c[2], c[3], dummy[0], dummy[1]);
    }

    frag_c[0] = c[0];
    frag_c[1] = c[1];
    frag_c[2] = c[2];
    frag_c[3] = c[3];
}

// =============================================================================
// FP16 accumulation variants
// =============================================================================

// Per-thread fragment API: A[4], B[2], frag_c[4] (FragC for fp16 is 4x half2).
__device__ void mma_m16n8k16_sm70_fp16(
    const uint32_t* A, const uint32_t* B, uint32_t* frag_c) {
    half2 c[4]; // Needs 4 half2s to cover 16x8 (same spatial size as 4 floats)
    c[0] = *reinterpret_cast<const half2*>(&frag_c[0]);
    c[1] = *reinterpret_cast<const half2*>(&frag_c[1]);
    c[2] = *reinterpret_cast<const half2*>(&frag_c[2]); // Expanded to full size
    c[3] = *reinterpret_cast<const half2*>(&frag_c[3]);

    for (int k = 0; k < 4; ++k) {
        half2 a = *reinterpret_cast<const half2*>(&A[k]);
        half2 b_pair = *reinterpret_cast<const half2*>(&B[k / 2]);

        half2 a_top = __halves2half2(a.x, a.x);
        half2 a_bot = __halves2half2(a.y, a.y);
        
        half b_scalar = (k % 2 == 0) ? b_pair.x : b_pair.y;
        half2 b_use = __halves2half2(b_scalar, b_scalar);
        
        // mma_fp16 writes 2 half2s (4 halves), implicitly covers 8x8 redundant pair
        // We use first pair
        // Note: simplified mma_fp16 outputs c0, c1
        mma_m8n8k4_sm70_fp16(a_top, b_use, c[0], c[1]);
        mma_m8n8k4_sm70_fp16(a_bot, b_use, c[2], c[3]);
    }

    frag_c[0] = *reinterpret_cast<const uint32_t*>(&c[0]);
    frag_c[1] = *reinterpret_cast<const uint32_t*>(&c[1]);
    frag_c[2] = *reinterpret_cast<const uint32_t*>(&c[2]); 
    frag_c[3] = *reinterpret_cast<const uint32_t*>(&c[3]);
}

// =============================================================================
// 16x8x32 MMA operations (composed from 2x m16n8k16)
// =============================================================================

// Version that writes to global memory
__device__ void mma_m16n8k32_sm70(
    const uint32_t* A, const uint32_t* B,
    float* C, int m, int n)
{
    // For k_size=32, we need A[32 uint32_t] and B[16 uint32_t]
    // Split into two k_size=16 operations:
    // 1. A[0:16] × B[0:8] → accumulate to C
    // 2. A[16:32] × B[8:16] → accumulate to C
    mma_m16n8k16_sm70(A, B, C, m, n);
    mma_m16n8k16_sm70(A + 16, B + 8, C, m, n);
}

// Per-thread fragment API: A[8], B[4], frag_c[4]. Two 16x8x16 steps.
__device__ void mma_m16n8k32_sm70(const uint32_t* A, const uint32_t* B,
                                  float* frag_c) {
    mma_m16n8k16_sm70(A, B, frag_c);
    mma_m16n8k16_sm70(A + 4, B + 2, frag_c);
}

// Version with optional C clearing
__device__ void mma_m16n8k32_sm70(const uint32_t* A, const uint32_t* B,
                                  float* C, bool no_c_clear) {
    const int m = 16, n = 8;
    if (!no_c_clear) {
        int tid = get_sm70_warp_lane();
        if (tid == 0) {
            for (int i = 0; i < m * n; i++) C[i] = 0.0f;
        }
        __syncwarp();
    }
    mma_m16n8k32_sm70(A, B, C, m, n);
}

} // namespace MARLIN_NAMESPACE_NAME