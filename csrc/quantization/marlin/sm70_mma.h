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

    // Iterate over 4 k-steps (k=0,4,8,12 in the k=16 dimension)
    // Each step uses one half2 from A and the corresponding half2 from B
    for (int k = 0; k < 4; ++k) {
        half2 a = *reinterpret_cast<const half2*>(&A[k]);
        half2 b = *reinterpret_cast<const half2*>(&B[k / 2]);
        mma_m8n8k4_sm70(a, b, c[0], c[1], c[2], c[3]);
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

    for (int k = 0; k < 4; ++k) {
        half2 a = *reinterpret_cast<const half2*>(&A[k * 4 + warp_id]);
        half2 b = *reinterpret_cast<const half2*>(&B[k * 4 + quadpair]);

        mma_m8n8k4_sm70(a, b, frag_C[0], frag_C[1], frag_C[2], frag_C[3]);
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

    for (int k = 0; k < 4; ++k) {
        half2 a = *reinterpret_cast<const half2*>(&A[k]);
        int i = k / 2;
        int shift = (k % 2) * 16;
        half2 b_tr = __halves2half2(
            __ushort_as_half(static_cast<unsigned short>((B[i] >> shift) & 0xFFFF)),
            __ushort_as_half(static_cast<unsigned short>((B2[i] >> shift) & 0xFFFF)));
        mma_m8n8k4_sm70(a, b_tr, c[0], c[1], c[2], c[3]);
    }

    frag_c[0] = c[0];
    frag_c[1] = c[1];
    frag_c[2] = c[2];
    frag_c[3] = c[3];
}

// =============================================================================
// FP16 accumulation variants
// =============================================================================

// Per-thread fragment API: A[4], B[2], frag_c[2] (FragC for fp16).
__device__ void mma_m16n8k16_sm70_fp16(
    const uint32_t* A, const uint32_t* B, uint32_t* frag_c) {
    half2 c[2];
    c[0] = *reinterpret_cast<const half2*>(&frag_c[0]);
    c[1] = *reinterpret_cast<const half2*>(&frag_c[1]);

    for (int k = 0; k < 4; ++k) {
        half2 a = *reinterpret_cast<const half2*>(&A[k]);
        half2 b = *reinterpret_cast<const half2*>(&B[k / 2]);
        mma_m8n8k4_sm70_fp16(a, b, c[0], c[1]);
    }

    frag_c[0] = *reinterpret_cast<const uint32_t*>(&c[0]);
    frag_c[1] = *reinterpret_cast<const uint32_t*>(&c[1]);
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