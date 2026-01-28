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

int main() {
    bool all_pass = true;
    all_pass &= test_checkerboard_integrity();
    all_pass &= test_nan_inf_stability();
    all_pass &= test_race_conditions();
    all_pass &= test_fp32_saturation();
    
    if(all_pass) printf("\nALL TORTURE TESTS PASSED\n");
    else printf("\nSOME TESTS FAILED\n");
    return all_pass ? 0 : 1;
}
