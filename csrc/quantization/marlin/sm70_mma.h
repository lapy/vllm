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
// Tensor Core MMA for Volta (SM70) using mma.m8n8k4
// 
// Composes multiple m8n8k4 operations to emulate m16n8k16.
// Uses actual Volta tensor cores for performance.
//
// Fragment layouts (for m16n8k16 as documented in PTX ISA):
//   A matrix [16×16]: 4 registers per thread (8 half values)
//   B matrix [16×8]:  2 registers per thread (4 half values)  
//   C matrix [16×8]:  4 floats per thread
//
// m8n8k4 uses:
//   A: 2 registers (4 halves) per thread
//   B: 2 registers (4 halves) per thread  
//   C: 8 floats per thread (but we only use partial results)
// =============================================================================

// Helper to extract half from half2 at specific position
__device__ __forceinline__ half extract_half(uint32_t reg, int idx) {
    half2 h2 = *reinterpret_cast<const half2*>(&reg);
    return (idx == 0) ? __low2half(h2) : __high2half(h2);
}

// Pack two halves into a half2/uint32_t
__device__ __forceinline__ uint32_t pack_halves(half h0, half h1) {
    half2 h2 = __halves2half2(h0, h1);
    return *reinterpret_cast<uint32_t*>(&h2);
}

__device__ void mma_m16n8k16_sm70(const uint32_t* A, const uint32_t* B,
                                  float* frag_c) {
    const int lane = threadIdx.x % 32;
    
    // ==========================================================================
    // SM70 m8n8k4 based implementation following CUTLASS documentation exactly:
    // https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html
    //
    // SM70_QuadPair: Layout<Shape<_4,_2>, Stride<_1,_16>>
    //   Maps logical thread id [0,8) to warp lane [0,4)U[16,20)
    //
    // SM70_8x4_Row: Layout<Shape<_8,_4>, Stride<_1,_8>>
    //   Thread t holds row t of the 8x4 A/B tile
    //
    // SM70_8x8_32b: Layout<Shape<Shape<_2,_2,_2>, Shape<_2,_2,_2>>,
    //                      Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>>
    //   Maps (thread_id, value_id) to (m,n) coordinate in 8x8 C matrix
    // ==========================================================================
    
    // Shared memory for the full matrices
    __shared__ __align__(16) half sh_A[16 * 16];
    __shared__ __align__(16) half sh_B[16 * 8];
    __shared__ __align__(16) float sh_C[16 * 8];
    
    // Initialize C to zero
    for (int i = lane; i < 16 * 8; i += 32) {
        sh_C[i] = 0.0f;
    }
    
    // Unpack A fragment from m16n8k16 layout to shared memory
    // A fragment layout: lane = row_group*4 + col_pair
    //   A[0]: (k=col_pair*2, col_pair*2+1) for row_group (rows 0-7)
    //   A[1]: (k=col_pair*2+8, col_pair*2+9) for row_group
    //   A[2]: (k=col_pair*2, col_pair*2+1) for row_group+8
    //   A[3]: (k=col_pair*2+8, col_pair*2+9) for row_group+8
    const half2* a_h2 = reinterpret_cast<const half2*>(A);
    int row_group = lane / 4;
    int col_pair = lane % 4;
    
    half2 a0 = a_h2[0];
    sh_A[row_group * 16 + col_pair * 2 + 0] = __low2half(a0);
    sh_A[row_group * 16 + col_pair * 2 + 1] = __high2half(a0);
    
    half2 a1 = a_h2[1];
    sh_A[row_group * 16 + col_pair * 2 + 8] = __low2half(a1);
    sh_A[row_group * 16 + col_pair * 2 + 9] = __high2half(a1);
    
    half2 a2 = a_h2[2];
    sh_A[(row_group + 8) * 16 + col_pair * 2 + 0] = __low2half(a2);
    sh_A[(row_group + 8) * 16 + col_pair * 2 + 1] = __high2half(a2);
    
    half2 a3 = a_h2[3];
    sh_A[(row_group + 8) * 16 + col_pair * 2 + 8] = __low2half(a3);
    sh_A[(row_group + 8) * 16 + col_pair * 2 + 9] = __high2half(a3);
    
    // Unpack B fragment to shared memory
    // B fragment layout: lane = b_col*4 + k_pair
    //   B[0]: (k=k_pair*2, k_pair*2+1) for column b_col
    //   B[1]: (k=k_pair*2+8, k_pair*2+9) for column b_col
    const half2* b_h2 = reinterpret_cast<const half2*>(B);
    int b_col = lane / 4;
    int k_pair = lane % 4;
    
    half2 b0 = b_h2[0];
    sh_B[(k_pair * 2 + 0) * 8 + b_col] = __low2half(b0);
    sh_B[(k_pair * 2 + 1) * 8 + b_col] = __high2half(b0);
    
    half2 b1 = b_h2[1];
    sh_B[(k_pair * 2 + 8) * 8 + b_col] = __low2half(b1);
    sh_B[(k_pair * 2 + 9) * 8 + b_col] = __high2half(b1);
    
    __syncwarp();
    
    // QuadPair: only lanes {0,1,2,3,16,17,18,19} participate in m8n8k4
    bool participates = (lane < 4) || (lane >= 16 && lane < 20);
    int logical_tid = participates ? ((lane < 4) ? lane : (lane - 16 + 4)) : 0;
    
    // Compute C[16x8] = A[16x16] * B[16x8]
    // Using m8n8k4: 2 row blocks × 4 k blocks
    for (int row_block = 0; row_block < 2; row_block++) {
        for (int k_block = 0; k_block < 4; k_block++) {
            int row_offset = row_block * 8;
            int k_offset = k_block * 4;
            
            // Load A fragment: thread t loads A[row_offset + t, k_offset : k_offset+4]
            uint32_t a0_reg = 0, a1_reg = 0;
            if (participates) {
                int a_row = row_offset + logical_tid;
                half a_k0 = sh_A[a_row * 16 + k_offset + 0];
                half a_k1 = sh_A[a_row * 16 + k_offset + 1];
                half a_k2 = sh_A[a_row * 16 + k_offset + 2];
                half a_k3 = sh_A[a_row * 16 + k_offset + 3];
                
                half2 a01 = __halves2half2(a_k0, a_k1);
                half2 a23 = __halves2half2(a_k2, a_k3);
                a0_reg = *reinterpret_cast<uint32_t*>(&a01);
                a1_reg = *reinterpret_cast<uint32_t*>(&a23);
            }
            
            // Load B fragment: thread t loads B[k_offset : k_offset+4, t]
            uint32_t b0_reg = 0, b1_reg = 0;
            if (participates) {
                int b_col_idx = logical_tid;
                half b_k0 = sh_B[(k_offset + 0) * 8 + b_col_idx];
                half b_k1 = sh_B[(k_offset + 1) * 8 + b_col_idx];
                half b_k2 = sh_B[(k_offset + 2) * 8 + b_col_idx];
                half b_k3 = sh_B[(k_offset + 3) * 8 + b_col_idx];
                
                half2 b01 = __halves2half2(b_k0, b_k1);
                half2 b23 = __halves2half2(b_k2, b_k3);
                b0_reg = *reinterpret_cast<uint32_t*>(&b01);
                b1_reg = *reinterpret_cast<uint32_t*>(&b23);
            }
            
            // Execute m8n8k4
            float c[8] = {0,0,0,0,0,0,0,0};
            
            asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                "{%8, %9}, {%10, %11}, "
                "{%0, %1, %2, %3, %4, %5, %6, %7};"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
                  "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                : "r"(a0_reg), "r"(a1_reg), "r"(b0_reg), "r"(b1_reg));
            
            // Store results using SM70_8x8_32b layout exactly as documented
            // index = t0*1 + t1*16 + t2*4 + v0*8 + v1*2 + v2*32
            // m = index % 8, n = index / 8
            if (participates) {
                int t = logical_tid;
                int t0 = t & 1;
                int t1 = (t >> 1) & 1;
                int t2 = (t >> 2) & 1;
                
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    int v0 = v & 1;
                    int v1 = (v >> 1) & 1;
                    int v2 = (v >> 2) & 1;
                    
                    int index = t0*1 + t1*16 + t2*4 + v0*8 + v1*2 + v2*32;
                    int m = index % 8;
                    int n = index / 8;
                    
                    atomicAdd(&sh_C[(row_offset + m) * 8 + n], c[v]);
                }
            }
            
            __syncwarp();
        }
    }
    
    __syncwarp();
    
    // Extract results in Marlin m16n8k16 layout
    int c_row0 = lane / 4;
    int c_col0 = (lane % 4) * 2;
    frag_c[0] = sh_C[c_row0 * 8 + c_col0];
    frag_c[1] = sh_C[c_row0 * 8 + c_col0 + 1];
    frag_c[2] = sh_C[(c_row0 + 8) * 8 + c_col0];
    frag_c[3] = sh_C[(c_row0 + 8) * 8 + c_col0 + 1];
}

__device__ void mma_m16n8k16_sm70_trans(const uint32_t* A, const uint32_t* B,
                                        const uint32_t* B2, float* frag_c) {
    // For transposed version, just call the regular version for now
    // The B2 parameter is for k>16 which we handle via iteration
    mma_m16n8k16_sm70(A, B, frag_c);
}

__device__ void mma_m16n8k16_sm70_fp16(
    const uint32_t* A, const uint32_t* B, uint32_t* frag_c) {
    // Use FP32 version and convert at the end
    float frag_f32[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Load existing FP16 accumulators and convert to FP32
    half2 cur0 = *reinterpret_cast<const half2*>(&frag_c[0]);
    half2 cur1 = *reinterpret_cast<const half2*>(&frag_c[1]);
    frag_f32[0] = __half2float(__low2half(cur0));
    frag_f32[1] = __half2float(__high2half(cur0));
    frag_f32[2] = __half2float(__low2half(cur1));
    frag_f32[3] = __half2float(__high2half(cur1));
    
    // Call the FP32 tensor core version
    mma_m16n8k16_sm70(A, B, frag_f32);
    
    // Convert back to FP16
    half2 result0 = __halves2half2(__float2half(frag_f32[0]), __float2half(frag_f32[1]));
    half2 result1 = __halves2half2(__float2half(frag_f32[2]), __float2half(frag_f32[3]));
    frag_c[0] = *reinterpret_cast<uint32_t*>(&result0);
    frag_c[1] = *reinterpret_cast<uint32_t*>(&result1);
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