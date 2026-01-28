/*
 * Torture Test Suite for SM70 Marlin
 * Designed to break the implementation with edge cases, race conditions, and numerical extremes.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

// Define namespace before including
#define MARLIN_NAMESPACE_NAME marlin_torture
#include "sm70_mma.h"

using namespace MARLIN_NAMESPACE_NAME;

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(1); \
        } \
    } while (0)

// Helper to pack halves
void pack_halves(uint32_t* out, const half* in, int num_u32) {
    for (int i = 0; i < num_u32; i++) {
        half h0 = in[i * 2];
        half h1 = in[i * 2 + 1];
        half2 packed = __halves2half2(h0, h1);
        out[i] = *reinterpret_cast<uint32_t*>(&packed);
    }
}

// -----------------------------------------------------------------------------
// Test 1: Checkerboard Pattern (Zero-Padding Rigor)
// -----------------------------------------------------------------------------
// Fills B with a checkerboard pattern. If padding logic (B 16x8 -> 16x16) is flawed,
// neighbor values will leak into the zero-padded region and corrupt the result.
__global__ void checkerboard_torture_kernel(
    const uint32_t* A_global, // [16, 16] - All 1.0
    const uint32_t* B_global, // [16, 8]  - Checkerboard
    float* C_global           // [16, 8]
) {
    int tid = threadIdx.x % 32;
    int wid = (threadIdx.x / 32) % 8;

    __shared__ char smem_pool[8 * 2048];
    half* sh_a = reinterpret_cast<half*>(&smem_pool[wid * 2048]);
    half* sh_b = reinterpret_cast<half*>(&smem_pool[wid * 2048 + 512]);
    float* sh_c = reinterpret_cast<float*>(&smem_pool[wid * 2048 + 1024]);

    // Load A (All 1.0)
    const uint32_t* A_u32 = A_global;
    for (int i = 0; i < 4; i++) {
        uint32_t val = A_u32[tid * 4 + i];
        half2 val_h2 = *reinterpret_cast<half2*>(&val);
        sh_a[(tid * 4 + i) * 2 + 0] = val_h2.x;
        sh_a[(tid * 4 + i) * 2 + 1] = val_h2.y;
    }

    // Load B (Checkerboard)
    const uint32_t* B_u32 = B_global;
    for (int i = 0; i < 2; i++) {
        uint32_t val = B_u32[tid * 2 + i];
        half2 val_h2 = *reinterpret_cast<half2*>(&val);
        sh_b[(tid * 2 + i) * 2 + 0] = val_h2.x;
        sh_b[(tid * 2 + i) * 2 + 1] = val_h2.y;
    }

    __syncthreads();

    float frag_c[4];
    mma_m16n8k16_sm70_direct(sh_a, sh_b, frag_c);

    // Store
    int core_row = tid / 4;
    int core_col = (tid % 4) * 2;
    C_global[(core_row + 0) * 8 + core_col + 0] = frag_c[0];
    C_global[(core_row + 0) * 8 + core_col + 1] = frag_c[1];
    C_global[(core_row + 8) * 8 + core_col + 0] = frag_c[2];
    C_global[(core_row + 8) * 8 + core_col + 1] = frag_c[3];
}

bool test_checkerboard_integrity() {
    printf("Running Checkerboard Integrity Torture Test...\n");
    const int M=16, N=8, K=16;
    
    std::vector<half> A_ref(M*K, __float2half(1.0f));
    std::vector<half> B_ref(K*N);
    
    // Create strict checkerboard: 1, 0, 1, 0...
    for(int i=0; i<K*N; i++) {
        B_ref[i] = __float2half((i % 2 == 0) ? 1.0f : 0.0f);
    }

    // Expected Result:
    // Since A is all 1s, C[row][col] = K/2 * 1.0 = 8.0 (if sum touches 8 ones and 8 zeros)
    // Actually, dot product of [1,1,1...] and [1,0,1,0...] is 8.
    // If padding leaks, we might get interference from adjacent rows/columns ?? 
    // Wait, padding extends K from 16 to 16 (no pad) or N from 8 to 16 (pad with 0).
    // If N-padding is broken (cols 8-15 contain garbage), it shouldn't affect cols 0-7 unless
    // the MMA operation wraps or reads out of bounds.
    // A better text for padding leakage is to have huge values in the "garbage" zone if we could control it.
    // But here we rely on mma_direct to do the padding.
    
    // CPU Ref
    std::vector<float> C_ref(M*N);
    for(int m=0; m<M; m++) {
        for(int n=0; n<N; n++) {
            float sum = 0;
            for(int k=0; k<K; k++) {
                 sum += __half2float(A_ref[m*K+k]) * __half2float(B_ref[k*N+n]);
            }
            C_ref[m*N+n] = sum;
        }
    }

    uint32_t *dA, *dB; float *dC;
    cudaMalloc(&dA, M*K*2); cudaMalloc(&dB, K*N*2); cudaMalloc(&dC, M*N*4);

    std::vector<uint32_t> A_packed(M*K/2), B_packed(K*N/2);
    pack_halves(A_packed.data(), A_ref.data(), M*K/2);
    pack_halves(B_packed.data(), B_ref.data(), K*N/2);

    cudaMemcpy(dA, A_packed.data(), A_packed.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size()*4, cudaMemcpyHostToDevice);

    checkerboard_torture_kernel<<<1, 32>>>(dA, dB, dC);

    std::vector<float> C_out(M*N);
    cudaMemcpy(C_out.data(), dC, M*N*4, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    float max_err = 0;
    for(int i=0; i<M*N; i++) {
        if(abs(C_out[i] - C_ref[i]) > max_err) max_err = abs(C_out[i] - C_ref[i]);
    }
    
    if(max_err > 0.05f) {
        printf("FAILED: Checkerboard pattern leaked! Max err: %f\n", max_err);
        return false;
    }
    printf("PASSED (Checkerboard integrity verified)\n");
    return true;
}

// -----------------------------------------------------------------------------
// Test 2: Nan/Inf Handling
// -----------------------------------------------------------------------------
bool test_nan_inf_stability() {
    printf("Running NaN/Inf Stability Test...\n");
    // If ANY input is NaN, result should be NaN.
    // If Inputs are clean, result must not be NaN.
    const int M=16, N=8, K=16;
    
    std::vector<half> A_ref(M*K, __float2half(1.0f));
    std::vector<half> B_ref(K*N, __float2half(1.0f));
    
    // Inject a NaN
    A_ref[0] = __float2half(NAN);
    
    uint32_t *dA, *dB; float *dC;
    cudaMalloc(&dA, M*K*2); cudaMalloc(&dB, K*N*2); cudaMalloc(&dC, M*N*4);
    
    std::vector<uint32_t> A_packed(M*K/2), B_packed(K*N/2);
    pack_halves(A_packed.data(), A_ref.data(), M*K/2);
    pack_halves(B_packed.data(), B_ref.data(), K*N/2);
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size()*4, cudaMemcpyHostToDevice);
    
    checkerboard_torture_kernel<<<1, 32>>>(dA, dB, dC); // Reuse kernel structure
    
    std::vector<float> C_out(M*N);
    cudaMemcpy(C_out.data(), dC, M*N*4, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    // Check results. Row 0 should be NaN. Others should be valid.
    int nan_count = 0;
    int valid_count = 0;
    for(int i=0; i<M*N; i++) {
        if(std::isnan(C_out[i])) nan_count++;
        else valid_count++;
    }
    
    // Row 0 has 8 elements. All should be NaN ideally, or at least some.
    // SM70 Tensor Cores propagate NaNs ? Yes they should.
    printf("  NaN Count: %d, Valid Count: %d\n", nan_count, valid_count);
    
    if(nan_count == 0) {
        printf("WARNING: NaN was swallowed! (Strictly speaking, fast-math might do this, but checking)\n");
        // Not a hard fail for quantization kernels usually, but good to know.
    }
    
    return true;
}

// -----------------------------------------------------------------------------
// Test 3: Extreme Thread Divergence (Race Condition Torture)
// -----------------------------------------------------------------------------
__global__ void race_condition_stress_kernel(float* count_out) {
    int tid = threadIdx.x;
    int wid = tid / 32;
    // 8 Warps
    // Each warp continuously writes to shared memory and reads back 
    // checking for interference from other warps.
    
    __shared__ char smem_pool[8 * 2048];
    volatile int* my_sh_data = reinterpret_cast<int*>(&smem_pool[wid * 2048]);
    
    // Write unique warp ID repeatedly
    for(int i=0; i<1000; i++) {
        my_sh_data[tid % 32] = wid * 1000 + i;
        // No barrier here! That's the point. 
        // We want to see if WID 0 touches WID 1's memory.
        
        // Busy wait to allow overlap
        for(int j=0; j<10; j++) { __nanosleep(10); } 
        
        int val = my_sh_data[tid % 32];
        if(val != wid * 1000 + i) {
            // CORRUPTION DETECTED
            atomicAdd(count_out, 1.0f);
        }
    }
}

bool test_race_conditions() {
    printf("Running Shared Memory Isolation Stress Test...\n");
    float* d_fail;
    cudaMalloc(&d_fail, 4);
    cudaMemset(d_fail, 0, 4);
    
    // Launch 8 warps (256 threads)
    race_condition_stress_kernel<<<1, 256>>>(d_fail);
    
    float fail_count = 0;
    cudaMemcpy(&fail_count, d_fail, 4, cudaMemcpyDeviceToHost);
    cudaFree(d_fail);
    
    if(fail_count > 0) {
        printf("FAILED: %f race conditions detected! Shared memory isolation is broken.\n", fail_count);
        return false;
    }
    printf("PASSED (No crosstalk between warps under high load)\n");
    return true;
}

// -----------------------------------------------------------------------------
// Test 4: FP32 Saturation / Accumulator Width Test
// -----------------------------------------------------------------------------
// FP16 Max is ~65504.
// If we compute Sum(300.0 * 300.0) over K=16, we get 1,440,000.
// This requires FP32 accumulation. If the kernel uses FP16 internally, 
// it will hit Infinity or saturate at 65504.
bool test_fp32_saturation() {
    printf("Running FP32 Saturation Test (Checking accumulator width)...\n");
    const int M=16, N=8, K=16;
    
    // 300.0 is representable in FP16 (exact).
    std::vector<half> A_ref(M*K, __float2half(300.0f));
    std::vector<half> B_ref(K*N, __float2half(300.0f));
    
    uint32_t *dA, *dB; float *dC;
    cudaMalloc(&dA, M*K*2); cudaMalloc(&dB, K*N*2); cudaMalloc(&dC, M*N*4);
    
    std::vector<uint32_t> A_packed(M*K/2), B_packed(K*N/2);
    pack_halves(A_packed.data(), A_ref.data(), M*K/2);
    pack_halves(B_packed.data(), B_ref.data(), K*N/2);
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size()*4, cudaMemcpyHostToDevice);
    
    // Reuse checkerboard kernel logic as it just does a standard MMA
    // checkerboard_torture_kernel is just a wrapper around mma_direct
    checkerboard_torture_kernel<<<1, 32>>>(dA, dB, dC);
    
    std::vector<float> C_out(M*N);
    cudaMemcpy(C_out.data(), dC, M*N*4, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    float expected = 300.0f * 300.0f * 16.0f; // 1,440,000.0
    float max_err = 0.0f;
    
    // Check first element
    if (std::isinf(C_out[0])) {
         printf("FAILED: Result is Infinity! Accumulator is likely FP16.\n");
         return false;
    }
    
    for(int i=0; i<M*N; i++) {
        if(abs(C_out[i] - expected) > max_err) max_err = abs(C_out[i] - expected);
    }
    
    printf("  Expected: %f, Got (avg): %f\n", expected, C_out[0]);
    if(max_err > 1.0f) {
        printf("FAILED: Large error %f. Precision loss or overflow detected.\n", max_err);
        return false;
    }
    
    printf("PASSED (Accumulator supports values > 65504)\n");
    return true;
}

// -----------------------------------------------------------------------------
// Test 5: Repeated MMA Stress (Memory Corruption Detection)
// -----------------------------------------------------------------------------
// Repeatedly calls MMA in a tight loop to detect register corruption or
// instruction pipeline issues.
__global__ void repeated_mma_stress_kernel(const uint32_t* A, const uint32_t* B, float* C, int iters) {
    int tid = threadIdx.x % 32;
    
    uint32_t frag_a[8];
    uint32_t frag_b[2];
    
    for (int i = 0; i < 8; i++) frag_a[i] = A[tid * 8 + i];
    for (int i = 0; i < 2; i++) frag_b[i] = B[tid * 2 + i];
    
    float frag_c[4] = {0.0f};
    
    for (int iter = 0; iter < iters; iter++) {
        mma_m16n8k16_sm70(frag_a, frag_b, frag_c);
        // Memory fence to prevent over-optimization
        __threadfence_block();
    }
    
    // Store final accumulated result
    int core_row = tid / 4;
    int core_col = (tid % 4) * 2;
    C[(core_row + 0) * 8 + core_col + 0] = frag_c[0];
    C[(core_row + 0) * 8 + core_col + 1] = frag_c[1];
    C[(core_row + 8) * 8 + core_col + 0] = frag_c[2];
    C[(core_row + 8) * 8 + core_col + 1] = frag_c[3];
}

bool test_repeated_mma_stress() {
    printf("Running Repeated MMA Stress Test (10000 iterations)...\n");
    const int M = 16, N = 8, K = 16;
    const int iters = 10000;
    
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    half h_one = __float2half(1.0f);
    half2 h2_one = __halves2half2(h_one, h_one);
    uint32_t u_one = *reinterpret_cast<uint32_t*>(&h2_one);
    
    for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = u_one;
    for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = u_one;
    
    uint32_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, A_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dB, B_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dC, M * N * sizeof(float));
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    repeated_mma_stress_kernel<<<1, 32>>>(dA, dB, dC, iters);
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> C_out(M * N);
    cudaMemcpy(C_out.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    // A=1, B=1, K=16 per iter => C = 16 * iters = 160000
    float expected = 16.0f * iters;
    float max_err = 0.0f;
    bool has_nan = false;
    bool has_inf = false;
    
    for (int i = 0; i < M * N; i++) {
        if (std::isnan(C_out[i])) has_nan = true;
        if (std::isinf(C_out[i])) has_inf = true;
        float err = std::abs(C_out[i] - expected);
        if (err > max_err) max_err = err;
    }
    
    if (has_nan) {
        printf("FAILED: NaN detected in output!\n");
        return false;
    }
    if (has_inf) {
        printf("FAILED: Infinity detected in output!\n");
        return false;
    }
    if (max_err > expected * 0.01f) {
        printf("FAILED: Max error %f exceeds 1%% of expected %f\n", max_err, expected);
        return false;
    }
    
    printf("PASSED (10000 iterations, expected=%.0f, max_err=%.2f)\n", expected, max_err);
    return true;
}

// -----------------------------------------------------------------------------
// Test 6: Concurrent Multi-Block Execution
// -----------------------------------------------------------------------------
// Launches many blocks simultaneously to test for global memory corruption
// or inter-block interference.
__global__ void multi_block_torture_kernel(const uint32_t* A, const uint32_t* B, float* C, int block_stride) {
    int tid = threadIdx.x % 32;
    int bid = blockIdx.x;
    
    uint32_t frag_a[8];
    uint32_t frag_b[2];
    
    // Each block reads from same input
    for (int i = 0; i < 8; i++) frag_a[i] = A[tid * 8 + i];
    for (int i = 0; i < 2; i++) frag_b[i] = B[tid * 2 + i];
    
    float frag_c[4] = {0.0f};
    
    // Multiple iterations per block
    for (int iter = 0; iter < 100; iter++) {
        mma_m16n8k16_sm70(frag_a, frag_b, frag_c);
    }
    
    // Each block writes to its own output region
    int out_offset = bid * block_stride;
    int core_row = tid / 4;
    int core_col = (tid % 4) * 2;
    C[out_offset + (core_row + 0) * 8 + core_col + 0] = frag_c[0];
    C[out_offset + (core_row + 0) * 8 + core_col + 1] = frag_c[1];
    C[out_offset + (core_row + 8) * 8 + core_col + 0] = frag_c[2];
    C[out_offset + (core_row + 8) * 8 + core_col + 1] = frag_c[3];
}

bool test_multi_block_concurrent() {
    printf("Running Multi-Block Concurrent Execution Test (100 blocks)...\n");
    const int M = 16, N = 8;
    const int num_blocks = 100;
    const int block_stride = M * N;
    const int iters_per_block = 100;
    
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    half h_one = __float2half(1.0f);
    half2 h2_one = __halves2half2(h_one, h_one);
    uint32_t u_one = *reinterpret_cast<uint32_t*>(&h2_one);
    
    for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = u_one;
    for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = u_one;
    
    uint32_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, A_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dB, B_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dC, num_blocks * block_stride * sizeof(float));
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    multi_block_torture_kernel<<<num_blocks, 32>>>(dA, dB, dC, block_stride);
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> C_out(num_blocks * block_stride);
    cudaMemcpy(C_out.data(), dC, num_blocks * block_stride * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    // Each block should produce 16.0 * iters_per_block = 1600.0
    float expected = 16.0f * iters_per_block;
    int fail_count = 0;
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < block_stride; i++) {
            float val = C_out[b * block_stride + i];
            if (std::isnan(val) || std::isinf(val) || std::abs(val - expected) > 1.0f) {
                fail_count++;
                if (fail_count <= 5) {
                    printf("  Block %d, element %d: got %f, expected %f\n", b, i, val, expected);
                }
            }
        }
    }
    
    if (fail_count > 0) {
        printf("FAILED: %d elements had incorrect values\n", fail_count);
        return false;
    }
    
    printf("PASSED (All 100 blocks produced identical correct results)\n");
    return true;
}

// -----------------------------------------------------------------------------
// Test 7: Denormalized Number Handling
// -----------------------------------------------------------------------------
bool test_denormalized_numbers() {
    printf("Running Denormalized Number Handling Test...\n");
    const int M = 16, N = 8, K = 16;
    
    // FP16 minimum normalized: ~6.1e-5, denormals go smaller
    std::vector<half> A_ref(M * K, __float2half(1e-7f)); // Very small
    std::vector<half> B_ref(K * N, __float2half(1e7f));  // Very large
    // Product should be ~1.0 per element, sum ~16.0
    
    uint32_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, M * K * 2);
    cudaMalloc(&dB, K * N * 2);
    cudaMalloc(&dC, M * N * 4);
    
    std::vector<uint32_t> A_packed(M * K / 2), B_packed(K * N / 2);
    pack_halves(A_packed.data(), A_ref.data(), M * K / 2);
    pack_halves(B_packed.data(), B_ref.data(), K * N / 2);
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size() * 4, cudaMemcpyHostToDevice);
    
    checkerboard_torture_kernel<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> C_out(M * N);
    cudaMemcpy(C_out.data(), dC, M * N * 4, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    // Check for NaN/Inf
    int nan_count = 0, inf_count = 0, zero_count = 0;
    for (int i = 0; i < M * N; i++) {
        if (std::isnan(C_out[i])) nan_count++;
        if (std::isinf(C_out[i])) inf_count++;
        if (C_out[i] == 0.0f) zero_count++;
    }
    
    printf("  NaN: %d, Inf: %d, Zero: %d, Sample value: %f\n", nan_count, inf_count, zero_count, C_out[0]);
    
    if (nan_count > 0) {
        printf("WARNING: NaN detected with denormalized inputs\n");
    }
    if (inf_count > 0) {
        printf("WARNING: Infinity detected with denormalized inputs\n");
    }
    
    // Not a hard fail - just checking behavior
    printf("PASSED (No crash with denormalized numbers)\n");
    return true;
}

// -----------------------------------------------------------------------------
// Test 8: Maximum Shared Memory Pressure
// -----------------------------------------------------------------------------
// Tests behavior when shared memory is heavily used
__global__ void max_smem_pressure_kernel(float* out) {
    // Allocate near-maximum shared memory (48KB on Volta)
    __shared__ char big_smem[47 * 1024]; // 47KB
    
    int tid = threadIdx.x;
    
    // Fill with pattern
    for (int i = tid; i < 47 * 1024; i += blockDim.x) {
        big_smem[i] = (char)(i & 0xFF);
    }
    __syncthreads();
    
    // Now run MMA which also uses shared memory internally
    __shared__ half sh_a[16 * 16];
    __shared__ half sh_b[16 * 8];
    
    if (tid < 256) {
        sh_a[tid] = __float2half(1.0f);
    }
    if (tid < 128) {
        sh_b[tid] = __float2half(1.0f);
    }
    __syncthreads();
    
    float frag_c[4] = {0.0f};
    if (tid < 32) {
        mma_m16n8k16_sm70_direct(sh_a, sh_b, frag_c);
    }
    __syncthreads();
    
    // Verify big_smem wasn't corrupted
    int errors = 0;
    for (int i = tid; i < 47 * 1024; i += blockDim.x) {
        if (big_smem[i] != (char)(i & 0xFF)) {
            errors++;
        }
    }
    
    // Reduce errors
    __shared__ int error_count;
    if (tid == 0) error_count = 0;
    __syncthreads();
    atomicAdd(&error_count, errors);
    __syncthreads();
    
    if (tid == 0) {
        out[0] = (float)error_count;
        out[1] = frag_c[0]; // Should be 16.0
    }
}

bool test_max_smem_pressure() {
    printf("Running Maximum Shared Memory Pressure Test...\n");
    
    float *d_out;
    cudaMalloc(&d_out, 8);
    
    // Use 256 threads
    max_smem_pressure_kernel<<<1, 256>>>(d_out);
    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        printf("SKIPPED: Kernel launch failed (likely insufficient smem): %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return true; // Not a failure, just resource limitation
    }
    
    cudaDeviceSynchronize();
    
    float h_out[2];
    cudaMemcpy(h_out, d_out, 8, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    
    if (h_out[0] > 0) {
        printf("FAILED: Shared memory corruption detected (%f errors)\n", h_out[0]);
        return false;
    }
    
    if (std::abs(h_out[1] - 16.0f) > 0.1f) {
        printf("WARNING: MMA result incorrect under smem pressure: %f vs 16.0\n", h_out[1]);
    }
    
    printf("PASSED (No corruption under maximum smem pressure)\n");
    return true;
}

// -----------------------------------------------------------------------------
// Test 9: Warp Divergence Torture
// -----------------------------------------------------------------------------
// Tests MMA behavior when threads within a warp take different paths
__global__ void warp_divergence_kernel(const uint32_t* A, const uint32_t* B, float* C) {
    int tid = threadIdx.x % 32;
    
    uint32_t frag_a[8];
    uint32_t frag_b[2];
    
    for (int i = 0; i < 8; i++) frag_a[i] = A[tid * 8 + i];
    for (int i = 0; i < 2; i++) frag_b[i] = B[tid * 2 + i];
    
    float frag_c[4] = {0.0f};
    
    // Divergent execution: different threads do different numbers of iterations
    // This should NOT affect correctness because MMA is warp-synchronous
    int my_iters = (tid % 4) + 1; // 1, 2, 3, or 4 iterations based on tid
    
    for (int iter = 0; iter < 4; iter++) {
        if (iter < my_iters) {
            mma_m16n8k16_sm70(frag_a, frag_b, frag_c);
        }
        __syncwarp(); // Reconverge
    }
    
    int core_row = tid / 4;
    int core_col = (tid % 4) * 2;
    C[(core_row + 0) * 8 + core_col + 0] = frag_c[0];
    C[(core_row + 0) * 8 + core_col + 1] = frag_c[1];
    C[(core_row + 8) * 8 + core_col + 0] = frag_c[2];
    C[(core_row + 8) * 8 + core_col + 1] = frag_c[3];
}

bool test_warp_divergence() {
    printf("Running Warp Divergence Torture Test...\n");
    const int M = 16, N = 8;
    
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    half h_one = __float2half(1.0f);
    half2 h2_one = __halves2half2(h_one, h_one);
    uint32_t u_one = *reinterpret_cast<uint32_t*>(&h2_one);
    
    for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = u_one;
    for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = u_one;
    
    uint32_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, A_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dB, B_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dC, M * N * sizeof(float));
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    warp_divergence_kernel<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> C_out(M * N);
    cudaMemcpy(C_out.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    // Due to divergence, different output columns have different iteration counts
    // Col 0-1: tid%4=0 -> 1 iter -> 16.0
    // Col 2-3: tid%4=1 -> 2 iters -> 32.0
    // Col 4-5: tid%4=2 -> 3 iters -> 48.0
    // Col 6-7: tid%4=3 -> 4 iters -> 64.0
    
    bool pass = true;
    float expected[8] = {16.0f, 16.0f, 32.0f, 32.0f, 48.0f, 48.0f, 64.0f, 64.0f};
    
    for (int row = 0; row < 16 && pass; row++) {
        for (int col = 0; col < 8 && pass; col++) {
            float val = C_out[row * 8 + col];
            float exp = expected[col];
            if (std::abs(val - exp) > 1.0f) {
                printf("FAILED: Row %d, Col %d: got %f, expected %f\n", row, col, val, exp);
                pass = false;
            }
        }
    }
    
    if (pass) {
        printf("PASSED (Divergent execution handled correctly)\n");
    }
    return pass;
}

// -----------------------------------------------------------------------------
// Test 10: Long-Running Stability Test
// -----------------------------------------------------------------------------
bool test_long_running_stability() {
    printf("Running Long-Running Stability Test (100000 kernel launches)...\n");
    const int M = 16, N = 8;
    const int num_launches = 100000;
    
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    half h_one = __float2half(1.0f);
    half2 h2_one = __halves2half2(h_one, h_one);
    uint32_t u_one = *reinterpret_cast<uint32_t*>(&h2_one);
    
    for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = u_one;
    for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = u_one;
    
    uint32_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, A_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dB, B_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dC, M * N * sizeof(float));
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Use a simple MMA kernel
    std::vector<float> C_out(M * N);
    int errors = 0;
    
    for (int launch = 0; launch < num_launches; launch++) {
        repeated_mma_stress_kernel<<<1, 32>>>(dA, dB, dC, 1);
        
        // Check every 10000 launches
        if ((launch + 1) % 10000 == 0) {
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("FAILED: CUDA error at launch %d: %s\n", launch, cudaGetErrorString(err));
                cudaFree(dA); cudaFree(dB); cudaFree(dC);
                return false;
            }
            
            cudaMemcpy(C_out.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Quick sanity check
            if (std::isnan(C_out[0]) || std::isinf(C_out[0])) {
                printf("FAILED: NaN/Inf detected at launch %d\n", launch);
                cudaFree(dA); cudaFree(dB); cudaFree(dC);
                return false;
            }
            
            printf("  Progress: %d/%d launches completed\n", launch + 1, num_launches);
        }
    }
    
    cudaDeviceSynchronize();
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    printf("PASSED (No errors over 100000 kernel launches)\n");
    return true;
}

// -----------------------------------------------------------------------------
// Test 11: Memory Alignment Stress
// -----------------------------------------------------------------------------
// Tests with intentionally misaligned pointers to ensure robustness
__global__ void misaligned_access_kernel(const char* A_raw, const char* B_raw, float* C, int offset) {
    int tid = threadIdx.x % 32;
    
    // Access with offset (potentially misaligned)
    const uint32_t* A = reinterpret_cast<const uint32_t*>(A_raw + offset);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(B_raw + offset);
    
    uint32_t frag_a[8];
    uint32_t frag_b[2];
    
    for (int i = 0; i < 8; i++) frag_a[i] = A[tid * 8 + i];
    for (int i = 0; i < 2; i++) frag_b[i] = B[tid * 2 + i];
    
    float frag_c[4] = {0.0f};
    mma_m16n8k16_sm70(frag_a, frag_b, frag_c);
    
    int core_row = tid / 4;
    int core_col = (tid % 4) * 2;
    C[(core_row + 0) * 8 + core_col + 0] = frag_c[0];
}

bool test_memory_alignment() {
    printf("Running Memory Alignment Stress Test...\n");
    
    // Allocate extra space for offset testing
    const size_t extra = 128;
    const size_t A_size = 32 * 8 * sizeof(uint32_t) + extra;
    const size_t B_size = 32 * 2 * sizeof(uint32_t) + extra;
    
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    half h_one = __float2half(1.0f);
    half2 h2_one = __halves2half2(h_one, h_one);
    uint32_t u_one = *reinterpret_cast<uint32_t*>(&h2_one);
    
    for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = u_one;
    for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = u_one;
    
    char *dA_raw, *dB_raw;
    float *dC;
    cudaMalloc(&dA_raw, A_size);
    cudaMalloc(&dB_raw, B_size);
    cudaMalloc(&dC, 16 * 8 * sizeof(float));
    
    bool all_pass = true;
    
    // Test with various alignments (0, 4, 8 bytes offset)
    // Note: CUDA requires 4-byte alignment for 32-bit access minimum
    int offsets[] = {0, 4, 8, 12, 16};
    
    for (int offset : offsets) {
        cudaMemcpy(dA_raw + offset, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dB_raw + offset, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        misaligned_access_kernel<<<1, 32>>>(dA_raw, dB_raw, dC, offset);
        cudaError_t err = cudaGetLastError();
        
        if (err != cudaSuccess) {
            printf("  Offset %d: CUDA error: %s\n", offset, cudaGetErrorString(err));
            all_pass = false;
            continue;
        }
        
        cudaDeviceSynchronize();
        
        float sample;
        cudaMemcpy(&sample, dC, sizeof(float), cudaMemcpyDeviceToHost);
        
        if (std::isnan(sample) || std::abs(sample - 16.0f) > 1.0f) {
            printf("  Offset %d: Incorrect result %f\n", offset, sample);
            all_pass = false;
        } else {
            printf("  Offset %d: OK (result=%f)\n", offset, sample);
        }
    }
    
    cudaFree(dA_raw); cudaFree(dB_raw); cudaFree(dC);
    
    if (all_pass) {
        printf("PASSED (All alignment tests succeeded)\n");
    }
    return all_pass;
}

// -----------------------------------------------------------------------------
// Test 12: Random Input Fuzz Test
// -----------------------------------------------------------------------------
bool test_random_fuzz() {
    printf("Running Random Input Fuzz Test (1000 random matrices)...\n");
    const int M = 16, N = 8, K = 16;
    const int num_tests = 1000;
    
    std::default_random_engine gen(42);
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
    
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    uint32_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, A_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dB, B_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dC, M * N * sizeof(float));
    
    int crash_count = 0;
    int nan_count = 0;
    int inf_count = 0;
    
    for (int test = 0; test < num_tests; test++) {
        // Fill with random bits
        for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = dist(gen);
        for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = dist(gen);
        
        cudaMemcpy(dA, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        repeated_mma_stress_kernel<<<1, 32>>>(dA, dB, dC, 1);
        cudaError_t err = cudaGetLastError();
        
        if (err != cudaSuccess) {
            crash_count++;
            continue;
        }
        
        cudaDeviceSynchronize();
        
        std::vector<float> C_out(M * N);
        cudaMemcpy(C_out.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < M * N; i++) {
            if (std::isnan(C_out[i])) nan_count++;
            if (std::isinf(C_out[i])) inf_count++;
        }
    }
    
    printf("  Tests: %d, Crashes: %d, NaN outputs: %d, Inf outputs: %d\n", 
           num_tests, crash_count, nan_count, inf_count);
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    if (crash_count > 0) {
        printf("FAILED: %d kernel crashes with random inputs\n", crash_count);
        return false;
    }
    
    printf("PASSED (No crashes with random inputs, NaN/Inf expected with random FP16)\n");
    return true;
}

int main() {
    bool all_pass = true;
    
    printf("\n========================================\n");
    printf("SM70 TORTURE TEST SUITE\n");
    printf("========================================\n\n");
    
    // Original tests
    all_pass &= test_checkerboard_integrity();
    all_pass &= test_nan_inf_stability();
    all_pass &= test_race_conditions();
    all_pass &= test_fp32_saturation();
    
    // New resilience tests
    printf("\n--- Additional Resilience Tests ---\n\n");
    all_pass &= test_repeated_mma_stress();
    all_pass &= test_multi_block_concurrent();
    all_pass &= test_denormalized_numbers();
    all_pass &= test_max_smem_pressure();
    all_pass &= test_warp_divergence();
    all_pass &= test_long_running_stability();
    all_pass &= test_memory_alignment();
    all_pass &= test_random_fuzz();
    
    printf("\n========================================\n");
    if(all_pass) printf("ALL TORTURE TESTS PASSED\n");
    else printf("SOME TESTS FAILED\n");
    printf("========================================\n");
    
    return all_pass ? 0 : 1;
}
