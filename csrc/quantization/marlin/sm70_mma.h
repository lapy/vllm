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
    
    // To avoid divergence (where threads process different 'word_base' indices
    // and thus execute different shuffle instructions), we shuffle ALL words.
    // This ensures Thread 0 (producing word 0) and Thread 8 (consuming word 2 from Thread 0)
    // coordinate correctly.
    
    uint32_t b0 = __shfl_sync(FULL_MASK, my_words[0], src_row_top);
    uint32_t b1 = __shfl_sync(FULL_MASK, my_words[1], src_row_top);
    uint32_t b2 = __shfl_sync(FULL_MASK, my_words[2], src_row_top);
    uint32_t b3 = __shfl_sync(FULL_MASK, my_words[3], src_row_top);

    // Word selection based on column grouping
    if (col_pair == 0) {
        dst[0] = b0;
        dst[1] = b1;
    } else {
        dst[0] = b2;
        dst[1] = b3;
    }
    
    // Repeat for bottom half
    b0 = __shfl_sync(FULL_MASK, my_words[0], src_row_bot);
    b1 = __shfl_sync(FULL_MASK, my_words[1], src_row_bot);
    b2 = __shfl_sync(FULL_MASK, my_words[2], src_row_bot);
    b3 = __shfl_sync(FULL_MASK, my_words[3], src_row_bot);
    
    if (col_pair == 0) {
        dst[2] = b0;
        dst[3] = b1;
    } else {
        dst[2] = b2;
        dst[3] = b3;
    }
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
// WMMA-based mma_m16n8k16 for Volta (SM70)
// Uses nvcuda::wmma (m16n16k16) with shared memory reconstruction.
// Results extracted into Marlin layout: 4 floats per thread.
// =============================================================================
__device__ void mma_m16n8k16_sm70(const uint32_t* A, const uint32_t* B,
                                  float* frag_c) {
    int tid = threadIdx.x % 32;

    // Use shared memory to convert between register layout and WMMA layout
    __shared__ half sh_a[16 * 16];  // A matrix: 16x16 row-major
    __shared__ half sh_b[16 * 16];  // B matrix: 16x16 col-major (padded, only 16x8 used)
    __shared__ float sh_c[16 * 16]; // C matrix: 16x16 row-major

    // Step 1: Write A fragments from registers to shared memory
    // Each thread has 8 uint32 = 16 halves for A
    // The test kernel layout: threads distribute rows and K slices
    // Thread t: A[0,1] = row (t%8 or t%8+8), K=0-3
    //           A[2,3] = row (t%8 or t%8+8), K=4-7 (for threads 0-7) or K=0-3 for bottom
    // Actually the test loads based on ldmatrix which shuffles.
    // For WMMA, we need A in row-major: A[row][col]

    // Unpack A fragments and write to shared memory in row-major order
    // The test kernel loads A with ldmatrix emulation.
    // For simplicity, let's reconstruct from the fragment values.
    // Thread t contributes to specific rows based on the ldmatrix layout.

    // ldmatrix x4 for m16n8k16 A layout:
    // Threads 0-7: provide rows 0-7 data
    // Threads 8-15: provide rows 0-7 data (different K columns)
    // Threads 16-23: provide rows 8-15 data
    // Threads 24-31: provide rows 8-15 data (different K columns)

    const half* a_halves = reinterpret_cast<const half*>(A);


    // Each thread has 16 halves for A covering K=k_base to k_base+7 (in two sets of 4)
    // A[0,1] = 4 halves for K=k_base to k_base+3
    // A[2,3] = 4 halves for rows 8-15 at same K (bottom tile)
    // A[4,5] = 4 halves for K=k_base+4 to k_base+7
    // etc.

    // Write top tile (rows 0-7) K values
    if (tid < 16) {  // Threads 0-15 have top tile data
        int row = tid % 8;
        int k_off = (tid / 8) * 8;  // 0 or 8

        // A[0,1] contains 4 halves for this row, K=k_off to k_off+3
        sh_a[row * 16 + k_off + 0] = a_halves[0];
        sh_a[row * 16 + k_off + 1] = a_halves[1];
        sh_a[row * 16 + k_off + 2] = a_halves[2];
        sh_a[row * 16 + k_off + 3] = a_halves[3];

        // A[4,5] contains 4 halves for this row, K=k_off+4 to k_off+7
        sh_a[row * 16 + k_off + 4] = a_halves[8];
        sh_a[row * 16 + k_off + 5] = a_halves[9];
        sh_a[row * 16 + k_off + 6] = a_halves[10];
        sh_a[row * 16 + k_off + 7] = a_halves[11];
    }

    // Write bottom tile (rows 8-15) K values
    if (tid >= 16) {  // Threads 16-31 have bottom tile data
        int row = 8 + (tid % 8);
        int k_off = ((tid - 16) / 8) * 8;  // 0 or 8

        // A[2,3] contains 4 halves for this row (actually stored in A[0,1] for these threads)
        sh_a[row * 16 + k_off + 0] = a_halves[0];
        sh_a[row * 16 + k_off + 1] = a_halves[1];
        sh_a[row * 16 + k_off + 2] = a_halves[2];
        sh_a[row * 16 + k_off + 3] = a_halves[3];

        sh_a[row * 16 + k_off + 4] = a_halves[8];
        sh_a[row * 16 + k_off + 5] = a_halves[9];
        sh_a[row * 16 + k_off + 6] = a_halves[10];
        sh_a[row * 16 + k_off + 7] = a_halves[11];
    }

    // Step 2: Write B fragments to shared memory in column-major order
    // B is 16x8, but WMMA needs 16x16, so we'll pad with zeros
    // The test kernel loads B with groupID = tid/4 determining the column
    const half* b_halves = reinterpret_cast<const half*>(B);
    int b_col = tid / 4;  // 0-7

    // Each thread has 16 halves for B covering all 16 K rows for one column
    // B fragments: B[k][col] where k=0..15
    for (int k = 0; k < 16; k++) {
        sh_b[k * 16 + b_col] = b_halves[k];
    }

    // Zero out the padding columns (8-15)
    if (tid < 8) {
        for (int k = 0; k < 16; k++) {
            sh_b[k * 16 + 8 + tid] = __float2half(0.0f);
        }
    }

    __syncwarp();

    // Step 3: Use WMMA to compute C = A * B
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_acc;

    // Initialize accumulator to zero
    wmma::fill_fragment(frag_acc, 0.0f);

    // Load A and B from shared memory
    wmma::load_matrix_sync(frag_a, sh_a, 16);  // 16 = leading dimension
    wmma::load_matrix_sync(frag_b, sh_b, 16);  // 16 = leading dimension

    // Perform the matrix multiplication
    wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);

    // Store result to shared memory
    wmma::store_matrix_sync(sh_c, frag_acc, 16, wmma::mem_row_major);

    __syncwarp();

    // Step 4: Extract results in Marlin layout
    // Marlin: thread t handles row (t/4), cols ((t%4)*2, (t%4)*2+1)
    int marlin_row = tid / 4;
    int marlin_col = (tid % 4) * 2;

    frag_c[0] = sh_c[marlin_row * 16 + marlin_col];
    frag_c[1] = sh_c[marlin_row * 16 + marlin_col + 1];
    frag_c[2] = sh_c[(marlin_row + 8) * 16 + marlin_col];
    frag_c[3] = sh_c[(marlin_row + 8) * 16 + marlin_col + 1];
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

    // Use additional shared memory for padded B and output C
    __shared__ half sh_b_padded[16 * 16];  // B padded to 16x16
    __shared__ float sh_c[16 * 16];        // C output 16x16

    // Pad B from 16x8 to 16x16 (copy existing columns, zero the rest)
    // Each thread handles part of the padding
    int elements_per_thread = (16 * 16) / 32;  // 8 elements per thread
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

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_acc;

    // Initialize accumulator
    wmma::fill_fragment(frag_acc, 0.0f);

    // Load matrices
    wmma::load_matrix_sync(frag_a, sh_a, lda);
    wmma::load_matrix_sync(frag_b, sh_b_padded, 16);

    // Compute
    wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);

    // Store result
    wmma::store_matrix_sync(sh_c, frag_acc, 16, wmma::mem_row_major);

    __syncwarp();

    // Extract in Marlin layout
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

    __shared__ half sh_b_padded[16 * 16];
    __shared__ float sh_c[16 * 16];

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

    // Accumulate to existing values
    frag_c[0] += sh_c[marlin_row * 16 + marlin_col];
    frag_c[1] += sh_c[marlin_row * 16 + marlin_col + 1];
    frag_c[2] += sh_c[(marlin_row + 8) * 16 + marlin_col];
    frag_c[3] += sh_c[(marlin_row + 8) * 16 + marlin_col + 1];
}

} // namespace MARLIN_NAMESPACE_NAME