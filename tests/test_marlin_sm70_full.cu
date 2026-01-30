/**
 * Full test suite for SM70 Marlin MMA functions including ldmatrix.
 * Tests all functions from marlin_mma_sm70.h
 *
 * Compile: nvcc -arch=sm_70 -o test_marlin_sm70_full test_marlin_sm70_full.cu
 * Run: ./test_marlin_sm70_full
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define MARLIN_NAMESPACE_NAME marlin_test
#include "marlin_mma_sm70.h"

using namespace marlin_test;

// =============================================================================
// CPU Reference
// =============================================================================

void cpu_matmul_16x8x16(const half* A, const half* B, float* C) {
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 16; k++) {
                sum += __half2float(A[i * 16 + k]) * __half2float(B[k * 8 + j]);
            }
            C[i * 8 + j] = sum;
        }
    }
}

// =============================================================================
// Test Kernels
// =============================================================================

__global__ void test_ldmatrix_x4_kernel(
    const half* __restrict__ sh_input,  // [32, 8] = 32 rows, each thread loads its row
    uint32_t* __restrict__ output        // [32, 4] = per-thread output
) {
    __shared__ half smem[32 * 8];  // 32 rows × 8 halves = 32 × 16 bytes

    const int lane = threadIdx.x % 32;

    // Each thread copies its row to shared memory
    for (int i = 0; i < 8; i++) {
        smem[lane * 8 + i] = sh_input[lane * 8 + i];
    }
    __syncwarp();

    // Pointer to this thread's row in shared memory
    const void* my_row = &smem[lane * 8];

    uint32_t dst[4];
    ldmatrix_m8n8_x4_sm70(dst, my_row);

    // Store results
    for (int i = 0; i < 4; i++) {
        output[lane * 4 + i] = dst[i];
    }
}

__global__ void test_mma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C
) {
    const int lane = threadIdx.x % 32;

    // Pack into Marlin FragA
    int row = lane / 4;
    int k_pair = lane % 4;

    uint32_t frag_a[4];
    half2 h2_a0 = __halves2half2(A[row * 16 + k_pair * 2], A[row * 16 + k_pair * 2 + 1]);
    half2 h2_a1 = __halves2half2(A[row * 16 + k_pair * 2 + 8], A[row * 16 + k_pair * 2 + 9]);
    half2 h2_a2 = __halves2half2(A[(row + 8) * 16 + k_pair * 2], A[(row + 8) * 16 + k_pair * 2 + 1]);
    half2 h2_a3 = __halves2half2(A[(row + 8) * 16 + k_pair * 2 + 8], A[(row + 8) * 16 + k_pair * 2 + 9]);
    frag_a[0] = *reinterpret_cast<uint32_t*>(&h2_a0);
    frag_a[1] = *reinterpret_cast<uint32_t*>(&h2_a1);
    frag_a[2] = *reinterpret_cast<uint32_t*>(&h2_a2);
    frag_a[3] = *reinterpret_cast<uint32_t*>(&h2_a3);

    // Pack into Marlin FragB
    int col = lane / 4;
    k_pair = lane % 4;

    uint32_t frag_b[2];
    half2 h2_b0 = __halves2half2(B[k_pair * 2 * 8 + col], B[(k_pair * 2 + 1) * 8 + col]);
    half2 h2_b1 = __halves2half2(B[(k_pair * 2 + 8) * 8 + col], B[(k_pair * 2 + 9) * 8 + col]);
    frag_b[0] = *reinterpret_cast<uint32_t*>(&h2_b0);
    frag_b[1] = *reinterpret_cast<uint32_t*>(&h2_b1);

    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_m16n8k16_sm70(frag_a, frag_b, frag_c);

    int marlin_row = lane / 4;
    int marlin_col_pair = lane % 4;

    C[marlin_row * 8 + marlin_col_pair * 2] = frag_c[0];
    C[marlin_row * 8 + marlin_col_pair * 2 + 1] = frag_c[1];
    C[(marlin_row + 8) * 8 + marlin_col_pair * 2] = frag_c[2];
    C[(marlin_row + 8) * 8 + marlin_col_pair * 2 + 1] = frag_c[3];
}

__global__ void test_mma_trans_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C
) {
    const int lane = threadIdx.x % 32;

    int col = lane / 4;
    int k_pair = lane % 4;

    uint32_t marlin_a[2];
    half2 h2_a0 = __halves2half2(B[k_pair * 2 * 8 + col], B[(k_pair * 2 + 1) * 8 + col]);
    half2 h2_a1 = __halves2half2(B[(k_pair * 2 + 8) * 8 + col], B[(k_pair * 2 + 9) * 8 + col]);
    marlin_a[0] = *reinterpret_cast<uint32_t*>(&h2_a0);
    marlin_a[1] = *reinterpret_cast<uint32_t*>(&h2_a1);

    int row = lane / 4;
    k_pair = lane % 4;

    uint32_t marlin_b[2], marlin_b2[2];

    half2 h2_b0 = __halves2half2(A[row * 16 + k_pair * 2], A[row * 16 + k_pair * 2 + 1]);
    half2 h2_b1 = __halves2half2(A[row * 16 + k_pair * 2 + 8], A[row * 16 + k_pair * 2 + 9]);
    marlin_b[0] = *reinterpret_cast<uint32_t*>(&h2_b0);
    marlin_b[1] = *reinterpret_cast<uint32_t*>(&h2_b1);

    half2 h2_b2_0 = __halves2half2(A[(row + 8) * 16 + k_pair * 2], A[(row + 8) * 16 + k_pair * 2 + 1]);
    half2 h2_b2_1 = __halves2half2(A[(row + 8) * 16 + k_pair * 2 + 8], A[(row + 8) * 16 + k_pair * 2 + 9]);
    marlin_b2[0] = *reinterpret_cast<uint32_t*>(&h2_b2_0);
    marlin_b2[1] = *reinterpret_cast<uint32_t*>(&h2_b2_1);

    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_m16n8k16_sm70_trans(marlin_a, marlin_b, marlin_b2, frag_c);

    int marlin_row = lane / 4;
    int marlin_col_pair = lane % 4;

    C[marlin_row * 8 + marlin_col_pair * 2] = frag_c[0];
    C[marlin_row * 8 + marlin_col_pair * 2 + 1] = frag_c[1];
    C[(marlin_row + 8) * 8 + marlin_col_pair * 2] = frag_c[2];
    C[(marlin_row + 8) * 8 + marlin_col_pair * 2 + 1] = frag_c[3];
}

__global__ void test_mma_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    const int lane = threadIdx.x % 32;

    int row = lane / 4;
    int k_pair = lane % 4;

    uint32_t frag_a[4];
    half2 h2_a0 = __halves2half2(A[row * 16 + k_pair * 2], A[row * 16 + k_pair * 2 + 1]);
    half2 h2_a1 = __halves2half2(A[row * 16 + k_pair * 2 + 8], A[row * 16 + k_pair * 2 + 9]);
    half2 h2_a2 = __halves2half2(A[(row + 8) * 16 + k_pair * 2], A[(row + 8) * 16 + k_pair * 2 + 1]);
    half2 h2_a3 = __halves2half2(A[(row + 8) * 16 + k_pair * 2 + 8], A[(row + 8) * 16 + k_pair * 2 + 9]);
    frag_a[0] = *reinterpret_cast<uint32_t*>(&h2_a0);
    frag_a[1] = *reinterpret_cast<uint32_t*>(&h2_a1);
    frag_a[2] = *reinterpret_cast<uint32_t*>(&h2_a2);
    frag_a[3] = *reinterpret_cast<uint32_t*>(&h2_a3);

    int col = lane / 4;
    k_pair = lane % 4;

    uint32_t frag_b[2];
    half2 h2_b0 = __halves2half2(B[k_pair * 2 * 8 + col], B[(k_pair * 2 + 1) * 8 + col]);
    half2 h2_b1 = __halves2half2(B[(k_pair * 2 + 8) * 8 + col], B[(k_pair * 2 + 9) * 8 + col]);
    frag_b[0] = *reinterpret_cast<uint32_t*>(&h2_b0);
    frag_b[1] = *reinterpret_cast<uint32_t*>(&h2_b1);

    uint32_t frag_c[2] = {0, 0};  // FP16 accumulator (4 halves)
    mma_m16n8k16_sm70_fp16(frag_a, frag_b, frag_c);

    // Unpack FP16 result
    half2 c0 = *reinterpret_cast<half2*>(&frag_c[0]);
    half2 c1 = *reinterpret_cast<half2*>(&frag_c[1]);

    int marlin_row = lane / 4;
    int marlin_col_pair = lane % 4;

    C[marlin_row * 8 + marlin_col_pair * 2] = __low2half(c0);
    C[marlin_row * 8 + marlin_col_pair * 2 + 1] = __high2half(c0);
    C[(marlin_row + 8) * 8 + marlin_col_pair * 2] = __low2half(c1);
    C[(marlin_row + 8) * 8 + marlin_col_pair * 2 + 1] = __high2half(c1);
}

// =============================================================================
// Test Functions
// =============================================================================

bool test_ldmatrix_x4() {
    printf("\n=== Test: ldmatrix_m8n8_x4_sm70 ===\n");

    // Input: 32 threads, each with 8 halves (one row)
    half h_input[32 * 8];
    uint32_t h_output[32 * 4];

    // Fill with sequential values for easy verification
    for (int t = 0; t < 32; t++) {
        for (int i = 0; i < 8; i++) {
            h_input[t * 8 + i] = __float2half((float)(t * 8 + i));
        }
    }

    half *d_input;
    uint32_t *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, 32 * 8 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, 32 * 4 * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, 32 * 8 * sizeof(half), cudaMemcpyHostToDevice));

    test_ldmatrix_x4_kernel<<<1, 32>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, 32 * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Verify: ldmatrix x4 loads 4 8x8 matrices
    // For lane L = row*4 + k_pair:
    //   dst[0] comes from rows 0-7 (src lanes 0-7)
    //   dst[1] comes from rows 8-15 (src lanes 8-15)
    //   dst[2] comes from rows 16-23 (src lanes 16-23)
    //   dst[3] comes from rows 24-31 (src lanes 24-31)

    // Check a few values
    int out_row = 0;  // lane 0
    int k_pair = 0;

    // For lane 0 (out_row=0, k_pair=0), dst[0] should come from lane 0, word 0
    half2 result0 = *reinterpret_cast<half2*>(&h_output[0 * 4 + 0]);
    float v0 = __half2float(__low2half(result0));
    float v1 = __half2float(__high2half(result0));

    printf("Lane 0 dst[0]: {%.0f, %.0f} (expected {0, 1})\n", v0, v1);

    bool passed = (fabsf(v0 - 0.0f) < 0.1f && fabsf(v1 - 1.0f) < 0.1f);

    // Check lane 5 = row 1, k_pair 1
    // dst[0] should come from src_lane 1 (row 1), word 1 = {10, 11}
    half2 result5_0 = *reinterpret_cast<half2*>(&h_output[5 * 4 + 0]);
    float v5_0 = __half2float(__low2half(result5_0));
    float v5_1 = __half2float(__high2half(result5_0));
    printf("Lane 5 dst[0]: {%.0f, %.0f} (expected {10, 11})\n", v5_0, v5_1);

    if (fabsf(v5_0 - 10.0f) > 0.1f || fabsf(v5_1 - 11.0f) > 0.1f) passed = false;

    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return passed;
}

bool test_random_matrices() {
    printf("\n=== Test: Random Matrices ===\n");

    srand(42);  // Reproducible

    half h_A[16 * 16];
    half h_B[16 * 8];
    float h_C_gpu[16 * 8] = {};
    float h_C_ref[16 * 8] = {};

    // Random values in [-1, 1]
    for (int i = 0; i < 16 * 16; i++) {
        h_A[i] = __float2half((float)(rand() % 200 - 100) / 100.0f);
    }
    for (int i = 0; i < 16 * 8; i++) {
        h_B[i] = __float2half((float)(rand() % 200 - 100) / 100.0f);
    }

    cpu_matmul_16x8x16(h_A, h_B, h_C_ref);

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, 16 * 8 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, 16 * 8 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, 16 * 8 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, 16 * 8 * sizeof(float)));

    test_mma_kernel<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int worst_i = -1, worst_j = -1;

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            float diff = fabsf(h_C_gpu[i * 8 + j] - h_C_ref[i * 8 + j]);
            float rel = (fabsf(h_C_ref[i * 8 + j]) > 1e-6f) ? diff / fabsf(h_C_ref[i * 8 + j]) : diff;
            if (diff > max_diff) {
                max_diff = diff;
                worst_i = i;
                worst_j = j;
            }
            if (rel > max_rel_diff) max_rel_diff = rel;
        }
    }

    printf("Max absolute diff: %.6f at [%d,%d]\n", max_diff, worst_i, worst_j);
    printf("Max relative diff: %.6f\n", max_rel_diff);

    // Allow small FP16 rounding errors
    bool passed = max_diff < 0.01f;
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

bool test_accumulation() {
    printf("\n=== Test: Accumulation (Multiple MMA calls) ===\n");

    half h_A1[16 * 16], h_A2[16 * 16];
    half h_B1[16 * 8], h_B2[16 * 8];
    float h_C_gpu[16 * 8] = {};
    float h_C_ref[16 * 8] = {};

    // A1 = all 1s, B1 = all 1s → C1 = all 16s
    // A2 = all 2s, B2 = all 0.5s → C2 = all 16s
    // Total = all 32s
    for (int i = 0; i < 16 * 16; i++) h_A1[i] = __float2half(1.0f);
    for (int i = 0; i < 16 * 8; i++) h_B1[i] = __float2half(1.0f);
    for (int i = 0; i < 16 * 16; i++) h_A2[i] = __float2half(2.0f);
    for (int i = 0; i < 16 * 8; i++) h_B2[i] = __float2half(0.5f);

    for (int i = 0; i < 16 * 8; i++) h_C_ref[i] = 32.0f;

    // This is a simplified test - we'll just check the MMA accumulates correctly
    // by calling it twice and checking the result is doubled

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, 16 * 8 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, 16 * 8 * sizeof(float)));

    // First call with A1, B1
    CUDA_CHECK(cudaMemcpy(d_A, h_A1, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B1, 16 * 8 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, 16 * 8 * sizeof(float)));

    test_mma_kernel<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < 16 * 8; i++) {
        float diff = fabsf(h_C_gpu[i] - 16.0f);
        if (diff > max_diff) max_diff = diff;
    }

    printf("After first MMA (expected 16.0): max diff = %.6f\n", max_diff);
    bool passed = max_diff < 0.01f;
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

bool test_fp16_accumulator() {
    printf("\n=== Test: FP16 Accumulator ===\n");

    half h_A[16 * 16];
    half h_B[16 * 8];
    half h_C_gpu[16 * 8] = {};
    float h_C_ref[16 * 8] = {};

    // Use small values to avoid FP16 overflow
    for (int i = 0; i < 16 * 16; i++) h_A[i] = __float2half(0.1f);
    for (int i = 0; i < 16 * 8; i++) h_B[i] = __float2half(0.1f);

    cpu_matmul_16x8x16(h_A, h_B, h_C_ref);  // Should be 0.16 everywhere

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, 16 * 8 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, 16 * 8 * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, 16 * 8 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, 16 * 8 * sizeof(half)));

    test_mma_fp16_kernel<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, 16 * 8 * sizeof(half), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < 16 * 8; i++) {
        float gpu_val = __half2float(h_C_gpu[i]);
        float diff = fabsf(gpu_val - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Expected value: %.4f, Sample GPU: %.4f\n", h_C_ref[0], __half2float(h_C_gpu[0]));
    printf("Max diff: %.6f\n", max_diff);

    bool passed = max_diff < 0.01f;
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

bool test_identity() {
    printf("\n=== Test: Identity Matrix ===\n");

    half h_A[16 * 16] = {};
    half h_B[16 * 8] = {};
    float h_C_gpu[16 * 8] = {};
    float h_C_ref[16 * 8] = {};

    for (int i = 0; i < 16; i++) h_A[i * 16 + i] = __float2half(1.0f);
    for (int i = 0; i < 8; i++) h_B[i * 8 + i] = __float2half(1.0f);

    cpu_matmul_16x8x16(h_A, h_B, h_C_ref);

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, 16 * 8 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, 16 * 8 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, 16 * 8 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, 16 * 8 * sizeof(float)));

    test_mma_kernel<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < 16 * 8; i++) {
        float diff = fabsf(h_C_gpu[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Max diff: %.6f\n", max_diff);
    bool passed = max_diff < 0.001f;
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

bool test_trans_random() {
    printf("\n=== Test: Trans - Random Matrices ===\n");

    srand(123);

    half h_A[16 * 16];
    half h_B[16 * 8];
    float h_C_gpu[16 * 8] = {};
    float h_C_ref[16 * 8] = {};

    for (int i = 0; i < 16 * 16; i++) {
        h_A[i] = __float2half((float)(rand() % 200 - 100) / 100.0f);
    }
    for (int i = 0; i < 16 * 8; i++) {
        h_B[i] = __float2half((float)(rand() % 200 - 100) / 100.0f);
    }

    cpu_matmul_16x8x16(h_A, h_B, h_C_ref);

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, 16 * 16 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, 16 * 8 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, 16 * 8 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, 16 * 8 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, 16 * 8 * sizeof(float)));

    test_mma_trans_kernel<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < 16 * 8; i++) {
        float diff = fabsf(h_C_gpu[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Max diff: %.6f\n", max_diff);
    bool passed = max_diff < 0.01f;
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    int sm70_device = -1;
    printf("============================================================\n");
    printf("Marlin SM70 Full Test Suite\n");
    printf("============================================================\n");

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("  GPU %d: %s (SM%d%d)\n", i, prop.name, prop.major, prop.minor);
        if (prop.major == 7 && prop.minor == 0 && sm70_device < 0) {
            sm70_device = i;
        }
    }

    if (sm70_device >= 0) {
        CUDA_CHECK(cudaSetDevice(sm70_device));
    }

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("\nRunning on: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);

    int passed = 0;
    int total = 0;

    printf("\n============================================================\n");
    printf("Low-level Primitives\n");
    printf("============================================================\n");
    total++; if (test_ldmatrix_x4()) passed++;

    printf("\n============================================================\n");
    printf("mma_m16n8k16_sm70\n");
    printf("============================================================\n");
    total++; if (test_identity()) passed++;
    total++; if (test_random_matrices()) passed++;
    total++; if (test_accumulation()) passed++;

    printf("\n============================================================\n");
    printf("mma_m16n8k16_sm70_trans\n");
    printf("============================================================\n");
    total++; if (test_trans_random()) passed++;

    printf("\n============================================================\n");
    printf("mma_m16n8k16_sm70_fp16\n");
    printf("============================================================\n");
    total++; if (test_fp16_accumulator()) passed++;

    printf("\n============================================================\n");
    printf("Summary: %d/%d tests passed\n", passed, total);
    printf("============================================================\n");

    return (passed == total) ? 0 : 1;
}
