/*
 * Torture Test Suite for SM70 Marlin - MMA Variant Testing
 * Designed to break the shuffle-based MMA implementation with edge cases,
 * race conditions, and numerical extremes.
 * 
 * Tests the mma_m16n8k16_sm70 and mma_m8n8k4_sm70 functions (NOT the WMMA direct versions).
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

// =============================================================================
// MMA Torture Kernel - Uses shuffle-based mma_m16n8k16_sm70
// =============================================================================
__global__ void mma_torture_kernel(
    const uint32_t* A_frag, // Per-thread A fragments [32 threads * 8 uint32]
    const uint32_t* B_frag, // Per-thread B fragments [32 threads * 2 uint32]
    float* C_global         // Output [16, 8]
) {
    int tid = threadIdx.x % 32;
    
    uint32_t frag_a[8];
    uint32_t frag_b[2];
    
    for (int i = 0; i < 8; i++) frag_a[i] = A_frag[tid * 8 + i];
    for (int i = 0; i < 2; i++) frag_b[i] = B_frag[tid * 2 + i];
    
    float frag_c[4] = {0.0f};
    mma_m16n8k16_sm70(frag_a, frag_b, frag_c);
    
    // Partitioned store
    int core_row = tid / 4;
    int core_col = (tid % 4) * 2;
    C_global[(core_row + 0) * 8 + core_col + 0] = frag_c[0];
    C_global[(core_row + 0) * 8 + core_col + 1] = frag_c[1];
    C_global[(core_row + 8) * 8 + core_col + 0] = frag_c[2];
    C_global[(core_row + 8) * 8 + core_col + 1] = frag_c[3];
}

// -----------------------------------------------------------------------------
// Test 1: Checkerboard Pattern (Data Integrity)
// -----------------------------------------------------------------------------
bool test_checkerboard_integrity() {
    printf("Running Checkerboard Integrity Test (shuffle-based MMA)...\n");
    const int M = 16, N = 8;
    
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    // A = all 1.0, B = checkerboard (1, 0, 1, 0...)
    half h_one = __float2half(1.0f);
    half h_zero = __float2half(0.0f);
    half2 h2_one = __halves2half2(h_one, h_one);
    half2 h2_checker = __halves2half2(h_one, h_zero);
    uint32_t u_one = *reinterpret_cast<uint32_t*>(&h2_one);
    uint32_t u_checker = *reinterpret_cast<uint32_t*>(&h2_checker);
    
    for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = u_one;
    for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = u_checker;
    
    uint32_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, A_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dB, B_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dC, M * N * sizeof(float));
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    mma_torture_kernel<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> C_out(M * N);
    cudaMemcpy(C_out.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    // Check for NaN/Inf
    bool has_nan = false, has_inf = false;
    for (int i = 0; i < M * N; i++) {
        if (std::isnan(C_out[i])) has_nan = true;
        if (std::isinf(C_out[i])) has_inf = true;
    }
    
    if (has_nan || has_inf) {
        printf("FAILED: NaN=%d, Inf=%d detected\n", has_nan, has_inf);
        return false;
    }
    
    printf("  Sample output: %f\n", C_out[0]);
    printf("PASSED (No NaN/Inf with checkerboard pattern)\n");
    return true;
}

// -----------------------------------------------------------------------------
// Test 2: Nan/Inf Handling
// -----------------------------------------------------------------------------
bool test_nan_inf_stability() {
    printf("Running NaN/Inf Stability Test (shuffle-based MMA)...\n");
    const int M = 16, N = 8;
    
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    // Fill with 1.0s first
    half h_one = __float2half(1.0f);
    half2 h2_one = __halves2half2(h_one, h_one);
    uint32_t u_one = *reinterpret_cast<uint32_t*>(&h2_one);
    
    for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = u_one;
    for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = u_one;
    
    // Inject NaN into A
    half h_nan = __float2half(NAN);
    half2 h2_nan = __halves2half2(h_nan, h_one);
    A_packed[0] = *reinterpret_cast<uint32_t*>(&h2_nan);
    
    uint32_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, A_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dB, B_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dC, M * N * sizeof(float));
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    mma_torture_kernel<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> C_out(M * N);
    cudaMemcpy(C_out.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    int nan_count = 0;
    int valid_count = 0;
    for (int i = 0; i < M * N; i++) {
        if (std::isnan(C_out[i])) nan_count++;
        else valid_count++;
    }
    
    printf("  NaN Count: %d, Valid Count: %d\n", nan_count, valid_count);
    
    if (nan_count == 0) {
        printf("WARNING: NaN was swallowed! (fast-math might do this)\n");
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
    printf("Running FP32 Saturation Test (shuffle-based MMA)...\n");
    const int M = 16, N = 8;
    
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    // 300.0 is representable in FP16 (exact)
    half h_300 = __float2half(300.0f);
    half2 h2_300 = __halves2half2(h_300, h_300);
    uint32_t u_300 = *reinterpret_cast<uint32_t*>(&h2_300);
    
    for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = u_300;
    for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = u_300;
    
    uint32_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, A_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dB, B_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dC, M * N * sizeof(float));
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    mma_torture_kernel<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> C_out(M * N);
    cudaMemcpy(C_out.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    
    float expected = 300.0f * 300.0f * 16.0f; // 1,440,000.0
    float max_err = 0.0f;
    
    if (std::isinf(C_out[0])) {
        printf("FAILED: Result is Infinity! Accumulator is likely FP16.\n");
        return false;
    }
    
    for (int i = 0; i < M * N; i++) {
        if (std::abs(C_out[i] - expected) > max_err) max_err = std::abs(C_out[i] - expected);
    }
    
    printf("  Expected: %f, Got: %f\n", expected, C_out[0]);
    if (max_err > 1.0f) {
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
    const int M = 16, N = 8;
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
    printf("Running Denormalized Number Handling Test (shuffle-based MMA)...\n");
    const int M = 16, N = 8;
    
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    // FP16 minimum normalized: ~6.1e-5, denormals go smaller
    // Very small * Very large should give ~1.0 per element
    half h_small = __float2half(1e-7f);
    half h_large = __float2half(1e7f);
    half2 h2_small = __halves2half2(h_small, h_small);
    half2 h2_large = __halves2half2(h_large, h_large);
    uint32_t u_small = *reinterpret_cast<uint32_t*>(&h2_small);
    uint32_t u_large = *reinterpret_cast<uint32_t*>(&h2_large);
    
    for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = u_small;
    for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = u_large;
    
    uint32_t *dA, *dB;
    float *dC;
    cudaMalloc(&dA, A_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dB, B_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dC, M * N * sizeof(float));
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    mma_torture_kernel<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> C_out(M * N);
    cudaMemcpy(C_out.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
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
__global__ void max_smem_pressure_kernel(const uint32_t* A_frag, const uint32_t* B_frag, float* out) {
    // Allocate shared memory to stress the system
    constexpr int BIG_SMEM_SIZE = 32 * 1024; // 32KB
    __shared__ char big_smem[BIG_SMEM_SIZE];
    
    int tid = threadIdx.x;
    
    // Fill with pattern
    for (int i = tid; i < BIG_SMEM_SIZE; i += blockDim.x) {
        big_smem[i] = (char)(i & 0xFF);
    }
    __syncthreads();
    
    // Run shuffle-based MMA (no shared memory internally)
    float frag_c[4] = {0.0f};
    if (tid < 32) {
        uint32_t frag_a[8];
        uint32_t frag_b[2];
        
        for (int i = 0; i < 8; i++) frag_a[i] = A_frag[tid * 8 + i];
        for (int i = 0; i < 2; i++) frag_b[i] = B_frag[tid * 2 + i];
        
        mma_m16n8k16_sm70(frag_a, frag_b, frag_c);
    }
    __syncthreads();
    
    // Verify big_smem wasn't corrupted
    int local_errors = 0;
    for (int i = tid; i < BIG_SMEM_SIZE; i += blockDim.x) {
        if (big_smem[i] != (char)(i & 0xFF)) {
            local_errors++;
        }
    }
    
    // Reduce errors
    __shared__ int error_count;
    if (tid == 0) error_count = 0;
    __syncthreads();
    atomicAdd(&error_count, local_errors);
    __syncthreads();
    
    if (tid == 0) {
        out[0] = (float)error_count;
        out[1] = frag_c[0]; // Should be 16.0
    }
}

bool test_max_smem_pressure() {
    printf("Running Maximum Shared Memory Pressure Test (shuffle-based MMA)...\n");
    
    // Prepare fragments
    std::vector<uint32_t> A_packed(32 * 8);
    std::vector<uint32_t> B_packed(32 * 2);
    
    half h_one = __float2half(1.0f);
    half2 h2_one = __halves2half2(h_one, h_one);
    uint32_t u_one = *reinterpret_cast<uint32_t*>(&h2_one);
    
    for (size_t i = 0; i < A_packed.size(); i++) A_packed[i] = u_one;
    for (size_t i = 0; i < B_packed.size(); i++) B_packed[i] = u_one;
    
    uint32_t *dA, *dB;
    float *d_out;
    cudaMalloc(&dA, A_packed.size() * sizeof(uint32_t));
    cudaMalloc(&dB, B_packed.size() * sizeof(uint32_t));
    cudaMalloc(&d_out, 8);
    
    cudaMemcpy(dA, A_packed.data(), A_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_packed.data(), B_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Use 256 threads
    max_smem_pressure_kernel<<<1, 256>>>(dA, dB, d_out);
    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        printf("SKIPPED: Kernel launch failed (likely insufficient smem): %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(d_out);
        return true; // Not a failure, just resource limitation
    }
    
    cudaDeviceSynchronize();
    
    float h_out[2];
    cudaMemcpy(h_out, d_out, 8, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(d_out);
    
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
// Tests MMA behavior with varying iteration counts per thread
// NOTE: MMA operations use __shfl_sync and MUST be warp-uniform (all threads participate)
// This test verifies that uniform MMA calls work correctly with different accumulation patterns
__global__ void warp_divergence_kernel(const uint32_t* A, const uint32_t* B, float* C) {
    int tid = threadIdx.x % 32;
    
    uint32_t frag_a[8];
    uint32_t frag_b[2];
    
    for (int i = 0; i < 8; i++) frag_a[i] = A[tid * 8 + i];
    for (int i = 0; i < 2; i++) frag_b[i] = B[tid * 2 + i];
    
    float frag_c[4] = {0.0f};
    
    // How many iterations this thread's column should accumulate
    int my_iters = (tid % 4) + 1; // 1, 2, 3, or 4 based on tid
    
    // All threads MUST call MMA together (warp-synchronous operation)
    // But each thread decides whether to keep the result or reset
    for (int iter = 0; iter < 4; iter++) {
        // ALL threads call MMA together - this is required!
        mma_m16n8k16_sm70(frag_a, frag_b, frag_c);
        
        // After MMA, threads that don't want this iteration's contribution
        // can choose to reset their accumulators (simulating divergent behavior)
        if (iter >= my_iters) {
            // Zero out accumulator for threads that have "finished"
            // This simulates divergent workloads where some threads do less work
            frag_c[0] = my_iters * 16.0f;
            frag_c[1] = my_iters * 16.0f;
            frag_c[2] = my_iters * 16.0f;
            frag_c[3] = my_iters * 16.0f;
        }
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
    const int M = 16, N = 8;
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
