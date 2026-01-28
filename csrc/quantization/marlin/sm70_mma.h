#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 700
#warning "sm70_mma.h is optimized for SM70 (Volta) architecture only"
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace MARLIN_NAMESPACE_NAME {

// Compile-time architecture verification
#if defined(__CUDA_ARCH__)
static_assert(__CUDA_ARCH__ >= 700, 
              "SM70 MMA library requires compute capability 7.0 or higher");
static_assert(__CUDA_ARCH__ < 750, 
              "SM70 MMA library is for Volta only; use native instructions for SM75+");
#endif

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
    
    // Each thread loads 1 uint32_t (2 halves) from its assigned smem address
    // The caller (Marlin) ensures smem_ptr is pre-offset for this thread's row
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t my_word = row_ptr[0];
    
    // Distribution: lane % 8 gives source row
    // Each group of 8 threads needs data from the corresponding group of 8 threads.
    // 0-7 get from 0-7. 8-15 get from 8-15 (who loaded the next column chunk).
    
    int source_row = lane % 8;
    // Calculate source lane: which 8-thread group are we in?
    int warp_group_offset = (lane / 8) * 8;
    int src_lane = warp_group_offset + source_row;
    
    // Shuffle to get the word from the correct source lane
    // Since each thread loaded the word it 'owns' logic-wise (via offset pointer),
    // and we just need to distribute rows within the 8x8 block logic:
    dst[0] = __shfl_sync(FULL_MASK, my_word, src_lane);
}

// Emulates ldmatrix.sync.aligned.m8n8.x2.shared.b16
__device__ __forceinline__ void ldmatrix_m8n8_x2_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    
    // For x2, each thread gets 2 registers
    // We load 2 words (8 bytes) from our pointer
    uint32_t my_words[2];
    my_words[0] = row_ptr[0];
    my_words[1] = row_ptr[1];
    
    // Layout matches mma.m16n8k16 operand A fragment
    // Lanes 0-7: rows 0-7, lanes 8-15: rows 0-7 (different columns)
    // Lanes 16-23: rows 8-15, lanes 24-31: rows 8-15 (different columns)
    
    // src_lane logic:
    // 0-7: Src 0-7.
    // 8-15: Src 8-15 (They loaded the next column chunk).
    
    int row_in_tile = lane % 8;
    int warp_group_offset = (lane / 8) * 8;
    int src_lane = warp_group_offset + row_in_tile;

    // We don't need 'col_group' offset logic for indices anymore because
    // my_words[0] IS the correct data for this group (since pointer was offset).
    uint32_t v0 = __shfl_sync(FULL_MASK, my_words[0], src_lane);
    uint32_t v1 = __shfl_sync(FULL_MASK, my_words[1], src_lane);
    
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
    int src_row_bot = row_in_group + 16;  // lanes 16-23 have rows 8-15
    
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





// =============================================================================
// Core 8x8x4 MMA operations for Volta
// =============================================================================



__device__ void mma_m8n8k4_sm70(
    const half2& a, const half2& b, 
    float& c0, float& c1, float& c2, float& c3) 
{
    uint32_t a_val = *reinterpret_cast<const uint32_t*>(&a);
    uint32_t b_val = *reinterpret_cast<const uint32_t*>(&b);
    
    float c_ext[8];
    c_ext[0] = c0; c_ext[1] = c1; c_ext[2] = c2; c_ext[3] = c3;
    c_ext[4] = 0.0f; c_ext[5] = 0.0f; c_ext[6] = 0.0f; c_ext[7] = 0.0f;

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



// Simplified FP16 version (2 output registers)
__device__ void mma_m8n8k4_sm70_fp16(
    const half2& a, const half2& b,
    half2& c0, half2& c1) 
{
    uint32_t a_val = *reinterpret_cast<const uint32_t*>(&a);
    uint32_t b_val = *reinterpret_cast<const uint32_t*>(&b);
    
    // SM70 f16 accumulation outputs 2 meaningful uint32_t (4 halves total)
    // PTX requires 4 output operands but only first 2 contain meaningful data
    uint32_t d0 = *reinterpret_cast<const uint32_t*>(&c0);
    uint32_t d1 = *reinterpret_cast<const uint32_t*>(&c1);
    uint32_t d2 = 0;
    uint32_t d3 = 0;

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a_val), "r"(a_val), "r"(b_val), "r"(b_val));

    c0 = *reinterpret_cast<half2*>(&d0);
    c1 = *reinterpret_cast<half2*>(&d1);
}

// =============================================================================
// 16x8x16 MMA operations (composed from 4x m8n8k4)
// =============================================================================

// Per-thread fragment API: A[4], B[2], frag_c[4]. Matches marlin_mma.h usage.
// This version works with pre-distributed fragments from Marlin's layout.
// Accumulates into this thread's FragC via 4 m8n8k4 steps.
// Per-thread fragment API: A[4], B[2], frag_c[4]. Matches marlin_mma.h usage.
// This version works with pre-distributed fragments from Marlin's layout.
// Accumulates into this thread's FragC via 4 m8n8k4 steps.
__device__ void mma_m16n8k16_sm70(const uint32_t* A, const uint32_t* B,
                                  float* frag_c) {
    float c[4] = {frag_c[0], frag_c[1], frag_c[2], frag_c[3]};
    float dummy[2];

    const half2* A_h = reinterpret_cast<const half2*>(A);
    const half2* B_h = reinterpret_cast<const half2*>(B);

    // SM70 ldmatrix returns:
    // A[0], A[1] -> Top 8 rows (distributed)
    // A[2], A[3] -> Bottom 8 rows (distributed)
    // We must correctly route these to the Top/Bot accumulators.

    // k=0: Process first k-slice
    {
        half2 a_t = A_h[0]; half2 a_b = A_h[2];
        half2 b_pair = B_h[0];
        half2 a_top = __halves2half2(a_t.x, a_t.x); 
        half2 a_bot = __halves2half2(a_b.x, a_b.x);
        half2 b_use = __halves2half2(b_pair.x, b_pair.x);
        mma_m8n8k4_sm70(a_top, b_use, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_use, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=1
    {
        half2 a_t = A_h[0]; half2 a_b = A_h[2];
        half2 b_pair = B_h[1]; 
        half2 a_top = __halves2half2(a_t.y, a_t.y);
        half2 a_bot = __halves2half2(a_b.y, a_b.y);
        half2 b_use = __halves2half2(b_pair.x, b_pair.x);
        mma_m8n8k4_sm70(a_top, b_use, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_use, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=2
    {
        half2 a_t = A_h[1]; half2 a_b = A_h[3];
        half2 b_pair = B_h[2];
        half2 a_top = __halves2half2(a_t.x, a_t.x);
        half2 a_bot = __halves2half2(a_b.x, a_b.x);
        half2 b_use = __halves2half2(b_pair.x, b_pair.x);
        mma_m8n8k4_sm70(a_top, b_use, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_use, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=3
    {
        half2 a_t = A_h[1]; half2 a_b = A_h[3];
        half2 b_pair = B_h[3];
        half2 a_top = __halves2half2(a_t.y, a_t.y);
        half2 a_bot = __halves2half2(a_b.y, a_b.y);
        half2 b_use = __halves2half2(b_pair.x, b_pair.x);
        mma_m8n8k4_sm70(a_top, b_use, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_use, c[2], c[3], dummy[0], dummy[1]);
    }

    frag_c[0] = c[0];
    frag_c[1] = c[1];
    frag_c[2] = c[2];
    frag_c[3] = c[3];
}



// =============================================================================
// Transposed B variants
// =============================================================================

// Per-thread fragment API with transposed B layout
__device__ void mma_m16n8k16_sm70_trans(const uint32_t* A, const uint32_t* B,
                                        const uint32_t* B2, float* frag_c) {
    float c[4] = {frag_c[0], frag_c[1], frag_c[2], frag_c[3]};
    float dummy[2];

    const half2* A_h = reinterpret_cast<const half2*>(A);

    // k=0
    {
        half2 a_t = A_h[0];
        half2 a_b = A_h[2];
        half2 a_top = __halves2half2(a_t.x, a_t.x);
        half2 a_bot = __halves2half2(a_b.x, a_b.x);
        half2 b_tr = __halves2half2(
            __ushort_as_half(static_cast<unsigned short>(B[0] & 0xFFFF)),
            __ushort_as_half(static_cast<unsigned short>(B2[0] & 0xFFFF)));
        mma_m8n8k4_sm70(a_top, b_tr, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_tr, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=1
    {
        half2 a_t = A_h[0];
        half2 a_b = A_h[2];
        half2 a_top = __halves2half2(a_t.y, a_t.y);
        half2 a_bot = __halves2half2(a_b.y, a_b.y);
        half2 b_tr = __halves2half2(
            __ushort_as_half(static_cast<unsigned short>((B[0] >> 16) & 0xFFFF)),
            __ushort_as_half(static_cast<unsigned short>((B2[0] >> 16) & 0xFFFF)));
        mma_m8n8k4_sm70(a_top, b_tr, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_tr, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=2
    {
        half2 a_t = A_h[1];
        half2 a_b = A_h[3];
        half2 a_top = __halves2half2(a_t.x, a_t.x);
        half2 a_bot = __halves2half2(a_b.x, a_b.x);
        half2 b_tr = __halves2half2(
            __ushort_as_half(static_cast<unsigned short>(B[1] & 0xFFFF)),
            __ushort_as_half(static_cast<unsigned short>(B2[1] & 0xFFFF)));
        mma_m8n8k4_sm70(a_top, b_tr, c[0], c[1], dummy[0], dummy[1]);
        mma_m8n8k4_sm70(a_bot, b_tr, c[2], c[3], dummy[0], dummy[1]);
    }

    // k=3
    {
        half2 a_t = A_h[1];
        half2 a_b = A_h[3];
        half2 a_top = __halves2half2(a_t.y, a_t.y);
        half2 a_bot = __halves2half2(a_b.y, a_b.y);
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

    const half2* A_h = reinterpret_cast<const half2*>(A);
    const half2* B_h = reinterpret_cast<const half2*>(B);

    for (int k = 0; k < 4; ++k) {
        // A Map:
        // k=0 -> T0 (A0.x), B0 (A2.x)
        // k=1 -> T1 (A0.y), B1 (A2.y)
        // k=2 -> T2 (A1.x), B2 (A3.x)
        // k=3 -> T3 (A1.y), B3 (A3.y)
        
        half2 a_t, a_b;
        if (k < 2) {
             half2 t = A_h[0];
             half2 b = A_h[2];
             if (k % 2 == 0) {
                 a_t = __halves2half2(t.x, t.x);
                 a_b = __halves2half2(b.x, b.x);
             } else {
                 a_t = __halves2half2(t.y, t.y);
                 a_b = __halves2half2(b.y, b.y);
             }
        } else {
             half2 t = A_h[1];
             half2 b = A_h[3];
             if (k % 2 == 0) {
                 a_t = __halves2half2(t.x, t.x);
                 a_b = __halves2half2(b.x, b.x);
             } else {
                 a_t = __halves2half2(t.y, t.y);
                 a_b = __halves2half2(b.y, b.y);
             }
        }
        
        // Update to use distinct B fragments B[0]..B[3]
        half2 b_pair = B_h[k]; 
        // Use first half of B[k]
        half b_scalar = b_pair.x;
        half2 b_use = __halves2half2(b_scalar, b_scalar);
        
        // mma_fp16 outputs c0, c1
        mma_m8n8k4_sm70_fp16(a_t, b_use, c[0], c[1]);
        mma_m8n8k4_sm70_fp16(a_b, b_use, c[2], c[3]);
    }

    frag_c[0] = *reinterpret_cast<const uint32_t*>(&c[0]);
    frag_c[1] = *reinterpret_cast<const uint32_t*>(&c[1]);
    frag_c[2] = *reinterpret_cast<const uint32_t*>(&c[2]); 
    frag_c[3] = *reinterpret_cast<const uint32_t*>(&c[3]);
}

// =============================================================================
// 16x8x32 MMA operations (composed from 2x m16n8k16)
// =============================================================================



// Per-thread fragment API: A[8], B[4], frag_c[4]. Two 16x8x16 steps.
__device__ void mma_m16n8k32_sm70(const uint32_t* A, const uint32_t* B,
                                  float* frag_c) {
    mma_m16n8k16_sm70(A, B, frag_c);
    mma_m16n8k16_sm70(A + 4, B + 4, frag_c);
}



} // namespace MARLIN_NAMESPACE_NAME