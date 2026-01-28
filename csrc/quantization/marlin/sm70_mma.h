#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 700
#warning "sm70_mma.h is optimized for SM70 (Volta) architecture only"
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <mma.h>

namespace MARLIN_NAMESPACE_NAME {

using namespace nvcuda;

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
// Emulates ldmatrix.m8n8.x{1,2,4} via warp shuffles.
// For x4: Thread t receives rows (t%8) and (t%8)+8, distributed across 4 registers.
// =============================================================================

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
// This is the primary function used by Marlin for loading FragA.
// A matrix is 16x16 (256 halves). 32 threads. Each thread provides 8 halves (16 bytes).
// We distribute the data such that each thread holds a vertical slice of 4 columns for 2 rows.
__device__ __forceinline__ void ldmatrix_m8n8_x4_sm70(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t my_words[4];
    my_words[0] = row_ptr[0]; my_words[1] = row_ptr[1];
    my_words[2] = row_ptr[2]; my_words[3] = row_ptr[3];
    
    // Distribution Protocol:
    // Src Threads have:
    // 0-7:   Row 0-7 Left (Cols 0-7)
    // 8-15:  Row 0-7 Right (Cols 8-15)
    // 16-23: Row 8-15 Left (Cols 0-7)
    // 24-31: Row 8-15 Right (Cols 8-15)
    
    // We map Dst threads (grp 0..3, row 0..7) to these sources.
    // Grp 0 (0-7):   Gets Cols 0-3 from Left Srcs.   (Words 0,1)
    // Grp 1 (8-15):  Gets Cols 8-11 from Right Srcs. (Words 0,1)
    // Grp 2 (16-23): Gets Cols 4-7 from Left Srcs.   (Words 2,3)
    // Grp 3 (24-31): Gets Cols 12-15 from Right Srcs. (Words 2,3)
    
    int grp = lane / 8;
    int row = lane % 8;
    
    bool from_right = (grp % 2 == 1); // Grp 1,3
    bool high_cols  = (grp >= 2);     // Grp 2,3
    
    int src_top_base = from_right ? 8 : 0;
    int src_bot_base = from_right ? 24 : 16;
    
    int src_top = row + src_top_base;
    int src_bot = row + src_bot_base;
    
    // If high_cols (4-7 or 12-15), we need words 2,3 from source.
    // If low_cols (0-3 or 8-11), we need words 0,1 from source.
    // Note: Source words: 0,1 cover Cols 0-3 (or 8-11). 2,3 cover Cols 4-7 (or 12-15).
    int w0_idx = high_cols ? 2 : 0;
    int w1_idx = high_cols ? 3 : 1;
    
    uint32_t top0 = __shfl_sync(FULL_MASK, my_words[w0_idx], src_top);
    uint32_t top1 = __shfl_sync(FULL_MASK, my_words[w1_idx], src_top);
    
    uint32_t bot0 = __shfl_sync(FULL_MASK, my_words[w0_idx], src_bot);
    uint32_t bot1 = __shfl_sync(FULL_MASK, my_words[w1_idx], src_bot);
    
    dst[0] = top0;
    dst[1] = top1;
    dst[2] = bot0;
    dst[3] = bot1;
}





// =============================================================================
// Core 8x8x4 MMA operations for Volta
// =============================================================================



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



// Simplified FP16 version (2 output registers)
__device__ __forceinline__ void mma_m8n8k4_sm70_fp16(
    uint32_t a0, uint32_t a1, 
    uint32_t b0, uint32_t b1, 
    uint32_t* c) 
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}



// =============================================================================
// Shuffle-based mma_m16n8k16 for Volta (SM70) with FP32 accumulation
// Uses warp shuffles to distribute A and B fragments, avoiding shared memory.
// Composes 4x mma.m8n8k4 to achieve m16n8k16.
//
// On SM70, FragB has the same size as other architectures (Vec<half2, 2> = 4 halves).
// The B data is distributed across threads in the warp via shuffles to provide
// the full k=16 dimension needed for the computation.
// =============================================================================
__device__ void mma_m16n8k16_sm70(const uint32_t* A, const uint32_t* B,
                                  float* frag_c) {
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    int tid = threadIdx.x % 32;
    int col_pair = tid % 4;
    
    // Initialize accumulator to add to existing frag_c values
    float c_local[4] = {frag_c[0], frag_c[1], frag_c[2], frag_c[3]};
    
    // m16n8k16 = 4 x m8n8k4
    // A fragment: 8 uint32 per thread (16x16 matrix distributed across 32 threads)
    // B fragment: 2 uint32 per thread (16x8 matrix, but only 4 halves per thread)
    //
    // For each k-step (0..3), we process k=4 elements:
    //   k=0: columns 0-3, k=1: columns 4-7, k=2: columns 8-11, k=3: columns 12-15
    // The B data is distributed across threads in groups of 8.
    // Threads 0-7 hold k=0-3, threads 8-15 hold k=4-7, etc.
    // We use shuffles to gather the appropriate B slice for each k-step.
    
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int k_part = (k / 2);
        // Shuffle A fragments from the appropriate source lanes
        // A layout: each thread holds portions of rows, need to gather by k-slice
        uint32_t a0_t = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 0], (tid % 8) + k_part * 8);
        uint32_t a1_t = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 1], (tid % 8) + k_part * 8);
        uint32_t a0_b = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 4], (tid % 8) + 16 + k_part * 8);
        uint32_t a1_b = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 5], (tid % 8) + 16 + k_part * 8);
        
        // Shuffle B fragments to get different k-slices from different threads.
        // B layout: B[0] and B[1] each hold 2 halves.
        // For k steps 0,2 we use B[0], for k steps 1,3 we use B[1]
        // Source lane provides data for a different k-slice
        int b_source = (tid % 8) + k_part * 8;
        int b_idx = k % 2;
        uint32_t b_val = __shfl_sync(FULL_MASK, B[b_idx], b_source);
        // For m8n8k4, we need 2 B registers - replicate for broadcast
        uint32_t b0 = b_val;
        uint32_t b1 = b_val;

        // Top 8 rows of output (m = 0..7)
        float c_step_t[8] = {0,0,0,0,0,0,0,0};
        mma_m8n8k4_sm70(a0_t, a1_t, b0, b1, c_step_t);
        c_local[0] += c_step_t[col_pair * 2 + 0];
        c_local[1] += c_step_t[col_pair * 2 + 1];

        // Bottom 8 rows of output (m = 8..15)
        float c_step_b[8] = {0,0,0,0,0,0,0,0};
        mma_m8n8k4_sm70(a0_b, a1_b, b0, b1, c_step_b);
        c_local[2] += c_step_b[col_pair * 2 + 0];
        c_local[3] += c_step_b[col_pair * 2 + 1];
    }
    
    // Write back accumulated results
    frag_c[0] = c_local[0];
    frag_c[1] = c_local[1];
    frag_c[2] = c_local[2];
    frag_c[3] = c_local[3];
}

__device__ void mma_m16n8k16_sm70_trans(const uint32_t* A, const uint32_t* B,
                                        const uint32_t* B2, float* frag_c) {
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    int tid = threadIdx.x % 32;
    int col_pair = tid % 4;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int k_part = (k / 2);
        uint32_t a0_t = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 0], (tid % 8) + k_part * 8);
        uint32_t a1_t = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 1], (tid % 8) + k_part * 8);
        uint32_t a0_b = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 4], (tid % 8) + 16 + k_part * 8);
        uint32_t a1_b = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 5], (tid % 8) + 16 + k_part * 8);
        
        // B is transposed, extract from packed format
        uint32_t b0, b1;
        if (k < 2) {
            half2 res = __halves2half2(__ushort_as_half(static_cast<unsigned short>((B[0] >> (k*16)) & 0xFFFF)),
                                       __ushort_as_half(static_cast<unsigned short>((B2[0] >> (k*16)) & 0xFFFF)));
            b0 = *reinterpret_cast<uint32_t*>(&res); b1 = b0;
        } else {
            half2 res = __halves2half2(__ushort_as_half(static_cast<unsigned short>((B[1] >> ((k-2)*16)) & 0xFFFF)),
                                       __ushort_as_half(static_cast<unsigned short>((B2[1] >> ((k-2)*16)) & 0xFFFF)));
            b0 = *reinterpret_cast<uint32_t*>(&res); b1 = b0;
        }

        float c_step_t[8] = {0,0,0,0,0,0,0,0};
        mma_m8n8k4_sm70(a0_t, a1_t, b0, b1, c_step_t);
        frag_c[0] += c_step_t[col_pair * 2 + 0];
        frag_c[1] += c_step_t[col_pair * 2 + 1];

        float c_step_b[8] = {0,0,0,0,0,0,0,0};
        mma_m8n8k4_sm70(a0_b, a1_b, b0, b1, c_step_b);
        frag_c[2] += c_step_b[col_pair * 2 + 0];
        frag_c[3] += c_step_b[col_pair * 2 + 1];
    }
}

__device__ void mma_m16n8k16_sm70_fp16(
    const uint32_t* A, const uint32_t* B, uint32_t* frag_c) {
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    int tid = threadIdx.x % 32;
    int col_pair = tid % 4;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int k_part = (k / 2);
        uint32_t a0_t = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 0], (tid % 8) + k_part * 8);
        uint32_t a1_t = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 1], (tid % 8) + k_part * 8);
        uint32_t a0_b = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 4], (tid % 8) + 16 + k_part * 8);
        uint32_t a1_b = __shfl_sync(FULL_MASK, A[(k % 2) * 2 + 5], (tid % 8) + 16 + k_part * 8);
        // Use thread-local B (no shuffle)
        uint32_t b0 = B[k * 2];
        uint32_t b1 = B[k * 2 + 1];

        uint32_t c_step_t[4] = {0,0,0,0};
        mma_m8n8k4_sm70_fp16(a0_t, a1_t, b0, b1, c_step_t);
        half2 res_t = *reinterpret_cast<half2*>(&c_step_t[col_pair]);
        half2 cur_t = *reinterpret_cast<half2*>(&frag_c[0]);
        cur_t = __hadd2(cur_t, res_t);
        frag_c[0] = *reinterpret_cast<uint32_t*>(&cur_t);

        uint32_t c_step_b[4] = {0,0,0,0};
        mma_m8n8k4_sm70_fp16(a0_b, a1_b, b0, b1, c_step_b);
        half2 res_b = *reinterpret_cast<half2*>(&c_step_b[col_pair]);
        half2 cur_b = *reinterpret_cast<half2*>(&frag_c[1]);
        cur_b = __hadd2(cur_b, res_b);
        frag_c[1] = *reinterpret_cast<uint32_t*>(&cur_b);
    }
}

// =============================================================================
// Higher-dimensional MMA operations (composed)
// =============================================================================

// Per-thread fragment API: A[16], B[16], frag_c[4]. Two 16x8x16 steps.
__device__ void mma_m16n8k32_sm70(const uint32_t* A, const uint32_t* B,
                                  float* frag_c) {
    mma_m16n8k16_sm70(A, B, frag_c);
    mma_m16n8k16_sm70(A + 8, B + 8, frag_c);
}

// =============================================================================
// Direct WMMA-based MMA (Shared Memory Interface)
// 
// Operates on matrices already in shared memory. 
// Pads B (16x8 -> 16x16) and computes C = A * B.
// Results extracted into Marlin layout.
// =============================================================================
__device__ void mma_m16n8k16_sm70_direct(const half* sh_a, const half* sh_b,
                                          float* frag_c, int lda = 16, int ldb = 8) {
    int tid = threadIdx.x % 32;
    int wid = (threadIdx.x / 32) % 8;

    // Use isolated shared memory pool. Each warp needs 512 + 1024 = 1.5KB.
    __shared__ char smem_pool_direct[8 * 1536];

    half* sh_b_padded = reinterpret_cast<half*>(&smem_pool_direct[wid * 1536]);
    float* sh_c = reinterpret_cast<float*>(&smem_pool_direct[wid * 1536 + 512]);

    // Pad B from 16x8 to 16x16
    int elements_per_thread = (16 * 16) / 32;
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid * elements_per_thread + i;
        int row = idx / 16;
        int col = idx % 16;
        if (col < 8) {
            sh_b_padded[row * 16 + col] = sh_b[row * ldb + col];
        } else {
            sh_b_padded[row * 16 + col] = __float2half(0.0f);
        }
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

// Accumulating version - adds to existing frag_c values
__device__ void mma_m16n8k16_sm70_direct_accum(const half* sh_a, const half* sh_b,
                                               float* frag_c, int lda = 16, int ldb = 8) {
    int tid = threadIdx.x % 32;
    int wid = (threadIdx.x / 32) % 8;

    __shared__ char smem_pool_acc[8 * 1536];

    half* sh_b_padded = reinterpret_cast<half*>(&smem_pool_acc[wid * 1536]);
    float* sh_c = reinterpret_cast<float*>(&smem_pool_acc[wid * 1536 + 512]);

    // Pad B
    int elements_per_thread = (16 * 16) / 32;
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid * elements_per_thread + i;
        int row = idx / 16;
        int col = idx % 16;
        if (col < 8) {
            sh_b_padded[row * 16 + col] = sh_b[row * ldb + col];
        } else {
            sh_b_padded[row * 16 + col] = __float2half(0.0f);
        }
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

} // namespace MARLIN_NAMESPACE_NAME