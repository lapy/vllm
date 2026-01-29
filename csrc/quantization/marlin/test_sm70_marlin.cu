// SM70 m8n8k4 Tensor Core Test
// Following CUTLASS documentation exactly:
// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

// =============================================================================
// CUTLASS Layout Definitions for SM70 m8n8k4
// =============================================================================

// SM70_QuadPair: Thread ID mapping
// Layout<Shape<_4,_2>, Stride<_1,_16>>
// Maps logical thread id [0,8) to warp lane [0,4)U[16,20)
__device__ __forceinline__ int logical_tid_to_lane(int logical_tid) {
    // logical_tid = t0 + t1*4 where t0 in [0,4), t1 in [0,2)
    int t0 = logical_tid % 4;
    int t1 = logical_tid / 4;
    return t0 * 1 + t1 * 16;  // stride (1, 16)
}

__device__ __forceinline__ int lane_to_logical_tid(int lane) {
    // Inverse of above
    if (lane < 4) return lane;
    if (lane >= 16 && lane < 20) return (lane - 16) + 4;
    return -1;  // Not participating
}

// =============================================================================
// SM70_8x8_32b: C matrix layout for F32 accumulators
// Layout<Shape<Shape<_2,_2,_2>, Shape<_2,_2,_2>>,
//        Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>>
//
// From documentation:
// (T0,V0) -> (m,n) = (0,0), encoded as 0
// (T0,V1) -> (m,n) = (0,1), encoded as 8
// (T0,V2) -> (m,n) = (2,0), encoded as 2
// (T0,V3) -> (m,n) = (2,1), encoded as 10
// (T0,V4) -> (m,n) = (0,4), encoded as 32
// (T0,V5) -> (m,n) = (0,5), encoded as 40
// (T0,V6) -> (m,n) = (2,4), encoded as 34
// (T0,V7) -> (m,n) = (2,5), encoded as 42
//
// (T1,V0) -> (m,n) = (1,0), encoded as 1
// (T2,V0) -> (m,n) = (0,2), encoded as 16
// (T4,V0) -> (m,n) = (4,0), encoded as 4
// etc.
//
// The encoding is: index = m + n*8 for 8x8 matrix
// =============================================================================

// Compute (m,n) from (thread_id, value_id) using SM70_8x8_32b layout
__device__ __forceinline__ void sm70_8x8_32b_decode(int t, int v, int& m, int& n) {
    // Thread t decomposition: t = t0 + 2*t1 + 4*t2
    int t0 = t & 1;
    int t1 = (t >> 1) & 1;
    int t2 = (t >> 2) & 1;
    
    // Value v decomposition: v = v0 + 2*v1 + 4*v2
    int v0 = v & 1;
    int v1 = (v >> 1) & 1;
    int v2 = (v >> 2) & 1;
    
    // From CUTLASS documentation:
    // Stride for thread: (1, 16, 4) applied to (t0, t1, t2)
    // Stride for value: (8, 2, 32) applied to (v0, v1, v2)
    // 
    // index = t0*1 + t1*16 + t2*4 + v0*8 + v1*2 + v2*32
    //
    // For 8x8 matrix with encoding index = m + n*8:
    // m = index % 8
    // n = index / 8
    
    int index = t0*1 + t1*16 + t2*4 + v0*8 + v1*2 + v2*32;
    m = index % 8;
    n = index / 8;
}

// =============================================================================
// Scalar reference implementation
// =============================================================================
__device__ void mma_m16n8k16_scalar_ref(const uint32_t* A, const uint32_t* B,
                                        float* frag_c) {
    const int lane = threadIdx.x % 32;
    
    __shared__ __align__(16) half sh_A[16 * 16];
    __shared__ __align__(16) half sh_B[16 * 8];
    __shared__ __align__(16) float sh_C[16 * 8];
    
    // Initialize C
    for (int i = lane; i < 16 * 8; i += 32) {
        sh_C[i] = 0.0f;
    }
    
    // Unpack A fragment to shared memory
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
    
    // Unpack B fragment
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
    
    // Scalar GEMM
    for (int r = lane; r < 16; r += 32) {
        for (int c = 0; c < 8; c++) {
            float sum = 0.0f;
            for (int k = 0; k < 16; k++) {
                sum += __half2float(sh_A[r * 16 + k]) * __half2float(sh_B[k * 8 + c]);
            }
            sh_C[r * 8 + c] = sum;
        }
    }
    
    __syncwarp();
    
    // Extract in Marlin layout
    int c_row0 = lane / 4;
    int c_col0 = (lane % 4) * 2;
    frag_c[0] = sh_C[c_row0 * 8 + c_col0];
    frag_c[1] = sh_C[c_row0 * 8 + c_col0 + 1];
    frag_c[2] = sh_C[(c_row0 + 8) * 8 + c_col0];
    frag_c[3] = sh_C[(c_row0 + 8) * 8 + c_col0 + 1];
}

// =============================================================================
// m8n8k4 implementation using CUTLASS layouts exactly
// 
// Using mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32
// This is the TN variant (A row-major, B col-major)
//
// For TN: ALayout = SM70_8x4_Row, BLayout = SM70_8x4_Row
// SM70_8x4_Row = Layout<Shape<_8,_4>, Stride<_1,_8>>
// Thread t holds row t of the 8x4 matrix
// =============================================================================
__device__ void mma_m16n8k16_m8n8k4(const uint32_t* A, const uint32_t* B,
                                    float* frag_c) {
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    // Shared memory for the full matrices
    __shared__ __align__(16) half sh_A[16 * 16];
    __shared__ __align__(16) half sh_B[16 * 8];
    __shared__ __align__(16) float sh_C[16 * 8];
    
    // Initialize C to zero
    for (int i = lane; i < 16 * 8; i += 32) {
        sh_C[i] = 0.0f;
    }
    
    // Unpack A fragment from m16n8k16 layout to shared memory
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
    
    // QuadPair: only lanes {0,1,2,3,16,17,18,19} participate
    bool participates = (lane < 4) || (lane >= 16 && lane < 20);
    int logical_tid = lane_to_logical_tid(lane);
    
    // We need to compute C[16x8] = A[16x16] * B[16x8]
    // Using m8n8k4: we need 2 row blocks × 4 k blocks
    
    // For each m8n8k4:
    // - A tile is 8x4, B tile is 4x8, C tile is 8x8
    // - SM70_8x4_Row: thread t (0-7) holds row t of the 8x4 tile
    // - SM70_8x8_32b: thread t, value v maps to C[m,n] as decoded above
    
    for (int row_block = 0; row_block < 2; row_block++) {
        for (int k_block = 0; k_block < 4; k_block++) {
            int row_offset = row_block * 8;
            int k_offset = k_block * 4;
            
            // Load A fragment: each participating thread loads one row
            // Thread t (0-7) loads A[row_offset + t, k_offset : k_offset+4]
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
            
            // Load B fragment: for row.col, B is column-major
            // Thread t (0-7) loads B[k_offset : k_offset+4, t]
            // This is one column of B, 4 elements
            uint32_t b0_reg = 0, b1_reg = 0;
            if (participates) {
                int b_col = logical_tid;
                half b_k0 = sh_B[(k_offset + 0) * 8 + b_col];
                half b_k1 = sh_B[(k_offset + 1) * 8 + b_col];
                half b_k2 = sh_B[(k_offset + 2) * 8 + b_col];
                half b_k3 = sh_B[(k_offset + 3) * 8 + b_col];
                
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
            if (participates) {
                #pragma unroll
                for (int v = 0; v < 8; v++) {
                    int m, n;
                    sm70_8x8_32b_decode(logical_tid, v, m, n);
                    
                    // Add to output at (row_offset + m, n)
                    atomicAdd(&sh_C[(row_offset + m) * 8 + n], c[v]);
                }
            }
            
            __syncwarp();
        }
    }
    
    __syncwarp();
    
    // Extract results in Marlin layout
    int c_row0 = lane / 4;
    int c_col0 = (lane % 4) * 2;
    frag_c[0] = sh_C[c_row0 * 8 + c_col0];
    frag_c[1] = sh_C[c_row0 * 8 + c_col0 + 1];
    frag_c[2] = sh_C[(c_row0 + 8) * 8 + c_col0];
    frag_c[3] = sh_C[(c_row0 + 8) * 8 + c_col0 + 1];
}

// =============================================================================
// Test kernel
// =============================================================================
__global__ void test_mma_kernel(float* results, int test_type) {
    const int tid = threadIdx.x % 32;
    
    // Create test data in m16n8k16 fragment format
    // A[16x16] where A[i][j] = (i+1) * 0.1
    // B[16x8] where B[i][j] = (j+1) * 0.1
    
    uint32_t frag_a[4];
    uint32_t frag_b[2];
    
    // Populate A fragment
    int row_in_group = tid / 4;
    int col_pair = tid % 4;
    
    // A[0]: k=0,1 for rows 0-7
    half a00 = __float2half((row_in_group + 1) * 0.1f);
    half a01 = __float2half((row_in_group + 1) * 0.1f);
    half2 a0_h2 = __halves2half2(a00, a01);
    frag_a[0] = *reinterpret_cast<uint32_t*>(&a0_h2);
    
    // A[1]: k=8,9 for rows 0-7
    half a10 = __float2half((row_in_group + 1) * 0.1f);
    half a11 = __float2half((row_in_group + 1) * 0.1f);
    half2 a1_h2 = __halves2half2(a10, a11);
    frag_a[1] = *reinterpret_cast<uint32_t*>(&a1_h2);
    
    // A[2]: k=0,1 for rows 8-15
    half a20 = __float2half((row_in_group + 8 + 1) * 0.1f);
    half a21 = __float2half((row_in_group + 8 + 1) * 0.1f);
    half2 a2_h2 = __halves2half2(a20, a21);
    frag_a[2] = *reinterpret_cast<uint32_t*>(&a2_h2);
    
    // A[3]: k=8,9 for rows 8-15
    half a30 = __float2half((row_in_group + 8 + 1) * 0.1f);
    half a31 = __float2half((row_in_group + 8 + 1) * 0.1f);
    half2 a3_h2 = __halves2half2(a30, a31);
    frag_a[3] = *reinterpret_cast<uint32_t*>(&a3_h2);
    
    // Populate B fragment
    int b_col = tid / 4;
    
    // B[0]: k=0,1 for col b_col
    half b00 = __float2half((b_col + 1) * 0.1f);
    half b01 = __float2half((b_col + 1) * 0.1f);
    half2 b0_h2 = __halves2half2(b00, b01);
    frag_b[0] = *reinterpret_cast<uint32_t*>(&b0_h2);
    
    // B[1]: k=8,9 for col b_col
    half b10 = __float2half((b_col + 1) * 0.1f);
    half b11 = __float2half((b_col + 1) * 0.1f);
    half2 b1_h2 = __halves2half2(b10, b11);
    frag_b[1] = *reinterpret_cast<uint32_t*>(&b1_h2);
    
    float frag_c[4] = {0, 0, 0, 0};
    
    if (test_type == 0) {
        mma_m16n8k16_scalar_ref(frag_a, frag_b, frag_c);
    } else {
        mma_m16n8k16_m8n8k4(frag_a, frag_b, frag_c);
    }
    
    // Store results
    results[tid * 4 + 0] = frag_c[0];
    results[tid * 4 + 1] = frag_c[1];
    results[tid * 4 + 2] = frag_c[2];
    results[tid * 4 + 3] = frag_c[3];
}

int main() {
    printf("=== SM70 m8n8k4 Tensor Core Test ===\n");
    printf("Using CUTLASS SM70_8x8_32b layout exactly\n\n");
    
    float *d_results_ref, *d_results_tc;
    cudaMalloc(&d_results_ref, 128 * sizeof(float));
    cudaMalloc(&d_results_tc, 128 * sizeof(float));
    
    test_mma_kernel<<<1, 32>>>(d_results_ref, 0);
    test_mma_kernel<<<1, 32>>>(d_results_tc, 1);
    cudaDeviceSynchronize();
    
    float h_results_ref[128], h_results_tc[128];
    cudaMemcpy(h_results_ref, d_results_ref, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_results_tc, d_results_tc, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print thread 0 comparison
    printf("Thread 0 comparison (frag_c[0..3]):\n");
    printf("  Scalar Ref: %.4f, %.4f, %.4f, %.4f\n",
           h_results_ref[0], h_results_ref[1], h_results_ref[2], h_results_ref[3]);
    printf("  m8n8k4 TC:  %.4f, %.4f, %.4f, %.4f\n",
           h_results_tc[0], h_results_tc[1], h_results_tc[2], h_results_tc[3]);
    
    // Reconstruct and print full C matrices
    printf("\nScalar Reference C[16x8]:\n");
    for (int row = 0; row < 16; row++) {
        printf("  ");
        for (int col = 0; col < 8; col++) {
            int tid = (row % 8) * 4 + (col / 2);
            int frag_idx = (row >= 8 ? 2 : 0) + (col % 2);
            printf("%7.4f ", h_results_ref[tid * 4 + frag_idx]);
        }
        printf("\n");
    }
    
    printf("\nm8n8k4 TC C[16x8]:\n");
    for (int row = 0; row < 16; row++) {
        printf("  ");
        for (int col = 0; col < 8; col++) {
            int tid = (row % 8) * 4 + (col / 2);
            int frag_idx = (row >= 8 ? 2 : 0) + (col % 2);
            printf("%7.4f ", h_results_tc[tid * 4 + frag_idx]);
        }
        printf("\n");
    }
    
    // Check accuracy
    float max_diff = 0;
    int error_count = 0;
    for (int i = 0; i < 128; i++) {
        float diff = fabsf(h_results_ref[i] - h_results_tc[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) error_count++;
    }
    
    printf("\nMax diff: %f\n", max_diff);
    printf("Errors (diff > 0.01): %d\n", error_count);
    printf("\nTest: %s\n", (max_diff < 0.01f) ? "PASSED" : "FAILED");
    
    cudaFree(d_results_ref);
    cudaFree(d_results_tc);
    
    return (max_diff < 0.01f) ? 0 : 1;
}
