/**
 * Standalone tests for SM70 Marlin MMA functions.
 * Tests mma_m16n8k16_sm70 and mma_m16n8k16_sm70_trans from marlin_mma_sm70.h
 *
 * Compile: nvcc -arch=sm_70 -o test_marlin_sm70_mma test_marlin_sm70_mma.cu
 * Run: ./test_marlin_sm70_mma
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Define the namespace to match Marlin
#define MARLIN_NAMESPACE_NAME marlin_test

// Include the header being tested
#include "../csrc/quantization/marlin/marlin_mma_sm70.h"

using namespace marlin_test;

// =============================================================================
// CPU Reference Implementation
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
// Test Kernel: mma_m16n8k16_sm70
// =============================================================================

__global__ void test_mma_kernel(
    const half* __restrict__ A,  // [16, 16] row-major
    const half* __restrict__ B,  // [16, 8] row-major
    float* __restrict__ C        // [16, 8] row-major
) {
    const int lane = threadIdx.x % 32;

    // Pack A into Marlin FragA layout
    // lane = row*4 + k_pair, where row∈[0,7], k_pair∈[0,3]
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

    // Pack B into Marlin FragB layout
    // lane = col*4 + k_pair, where col∈[0,7], k_pair∈[0,3]
    int col = lane / 4;
    k_pair = lane % 4;

    uint32_t frag_b[2];
    half2 h2_b0 = __halves2half2(B[k_pair * 2 * 8 + col], B[(k_pair * 2 + 1) * 8 + col]);
    half2 h2_b1 = __halves2half2(B[(k_pair * 2 + 8) * 8 + col], B[(k_pair * 2 + 9) * 8 + col]);
    frag_b[0] = *reinterpret_cast<uint32_t*>(&h2_b0);
    frag_b[1] = *reinterpret_cast<uint32_t*>(&h2_b1);

    // Call the function under test
    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_m16n8k16_sm70(frag_a, frag_b, frag_c);

    // Write output - Marlin frag_c layout:
    // lane = row*4 + col_pair, where row∈[0,7], col_pair∈[0,3]
    int marlin_row = lane / 4;
    int marlin_col_pair = lane % 4;

    C[marlin_row * 8 + marlin_col_pair * 2] = frag_c[0];
    C[marlin_row * 8 + marlin_col_pair * 2 + 1] = frag_c[1];
    C[(marlin_row + 8) * 8 + marlin_col_pair * 2] = frag_c[2];
    C[(marlin_row + 8) * 8 + marlin_col_pair * 2 + 1] = frag_c[3];
}

// =============================================================================
// Test Kernel: mma_m16n8k16_sm70_trans
// =============================================================================

__global__ void test_mma_trans_kernel(
    const half* __restrict__ A,   // [16, 16] - weights (marlin_b, marlin_b2)
    const half* __restrict__ B,   // [16, 8] - activations (marlin_a)
    float* __restrict__ C         // [16, 8]
) {
    const int lane = threadIdx.x % 32;

    // Pack activations (B) into marlin_a format
    // marlin_a: lane = col*4 + k_pair
    int col = lane / 4;
    int k_pair = lane % 4;

    uint32_t marlin_a[2];
    half2 h2_a0 = __halves2half2(B[k_pair * 2 * 8 + col], B[(k_pair * 2 + 1) * 8 + col]);
    half2 h2_a1 = __halves2half2(B[(k_pair * 2 + 8) * 8 + col], B[(k_pair * 2 + 9) * 8 + col]);
    marlin_a[0] = *reinterpret_cast<uint32_t*>(&h2_a0);
    marlin_a[1] = *reinterpret_cast<uint32_t*>(&h2_a1);

    // Pack weights (A) into marlin_b/marlin_b2 format
    // marlin_b: lane = row*4 + k_pair for rows 0-7
    // marlin_b2: lane = row*4 + k_pair for rows 8-15
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

    // Call the function under test
    float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_m16n8k16_sm70_trans(marlin_a, marlin_b, marlin_b2, frag_c);

    // Write output
    int marlin_row = lane / 4;
    int marlin_col_pair = lane % 4;

    C[marlin_row * 8 + marlin_col_pair * 2] = frag_c[0];
    C[marlin_row * 8 + marlin_col_pair * 2 + 1] = frag_c[1];
    C[(marlin_row + 8) * 8 + marlin_col_pair * 2] = frag_c[2];
    C[(marlin_row + 8) * 8 + marlin_col_pair * 2 + 1] = frag_c[3];
}

// =============================================================================
// Test Functions
// =============================================================================

bool test_identity() {
    printf("\n=== Test: Identity Matrix ===\n");

    half h_A[16 * 16] = {};
    half h_B[16 * 8] = {};
    float h_C_gpu[16 * 8] = {};
    float h_C_ref[16 * 8] = {};

    // A = 16x16 identity, B = I(8) padded to 16x8
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

    printf("Expected C (diagonal 1s for rows 0-7):\n");
    for (int i = 0; i < 8; i++) {
        printf("  row %d: ", i);
        for (int j = 0; j < 8; j++) printf("%4.1f ", h_C_ref[i * 8 + j]);
        printf("\n");
    }

    printf("\nActual C (rows 0-7):\n");
    for (int i = 0; i < 8; i++) {
        printf("  row %d: ", i);
        for (int j = 0; j < 8; j++) printf("%4.1f ", h_C_gpu[i * 8 + j]);
        printf("\n");
    }

    printf("\nActual C (rows 8-15):\n");
    for (int i = 8; i < 16; i++) {
        printf("  row %d: ", i);
        for (int j = 0; j < 8; j++) printf("%4.1f ", h_C_gpu[i * 8 + j]);
        printf("\n");
    }

    float max_diff = 0.0f;
    int worst_i = -1, worst_j = -1;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            float diff = fabsf(h_C_gpu[i * 8 + j] - h_C_ref[i * 8 + j]);
            if (diff > max_diff) {
                max_diff = diff;
                worst_i = i;
                worst_j = j;
            }
        }
    }

    printf("\nMax diff: %.6f", max_diff);
    if (max_diff > 0.001f) printf(" at [%d,%d]", worst_i, worst_j);
    printf("\n");

    bool passed = max_diff < 0.001f;
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

bool test_all_ones() {
    printf("\n=== Test: All Ones ===\n");

    half h_A[16 * 16];
    half h_B[16 * 8];
    float h_C_gpu[16 * 8] = {};

    for (int i = 0; i < 16 * 16; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < 16 * 8; i++) h_B[i] = __float2half(1.0f);

    // Expected: C[i,j] = 16 for all positions

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

    printf("Expected: all 16.0\n");
    printf("Sample values: C[0,0]=%.1f, C[4,4]=%.1f, C[8,0]=%.1f, C[15,7]=%.1f\n",
           h_C_gpu[0], h_C_gpu[4*8+4], h_C_gpu[8*8+0], h_C_gpu[15*8+7]);

    float max_diff = 0.0f;
    for (int i = 0; i < 16 * 8; i++) {
        float diff = fabsf(h_C_gpu[i] - 16.0f);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Max diff from 16.0: %.6f\n", max_diff);
    bool passed = max_diff < 0.01f;
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

bool test_unique_values() {
    printf("\n=== Test: Unique Row/Col Values ===\n");

    // A[i,k] = i+1 for all k (row i has all (i+1)s)
    // B[k,j] = j+1 for all k (col j has all (j+1)s)
    // C[i,j] = sum(A[i,k]*B[k,j]) = 16 * (i+1) * (j+1)

    half h_A[16 * 16];
    half h_B[16 * 8];
    float h_C_gpu[16 * 8] = {};
    float h_C_ref[16 * 8] = {};

    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 16; k++) {
            h_A[i * 16 + k] = __float2half((float)(i + 1));
        }
    }
    for (int k = 0; k < 16; k++) {
        for (int j = 0; j < 8; j++) {
            h_B[k * 8 + j] = __float2half((float)(j + 1));
        }
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

    printf("Expected C[i,j] = 16*(i+1)*(j+1):\n");
    printf("  C[0,0]=%.0f, C[4,4]=%.0f, C[8,7]=%.0f, C[15,7]=%.0f\n",
           h_C_ref[0], h_C_ref[4*8+4], h_C_ref[8*8+7], h_C_ref[15*8+7]);

    printf("Actual:\n");
    printf("  C[0,0]=%.0f, C[4,4]=%.0f, C[8,7]=%.0f, C[15,7]=%.0f\n",
           h_C_gpu[0], h_C_gpu[4*8+4], h_C_gpu[8*8+7], h_C_gpu[15*8+7]);

    float max_diff = 0.0f;
    int worst_i = -1, worst_j = -1;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            float diff = fabsf(h_C_gpu[i * 8 + j] - h_C_ref[i * 8 + j]);
            if (diff > max_diff) {
                max_diff = diff;
                worst_i = i;
                worst_j = j;
            }
        }
    }

    printf("Max diff: %.6f", max_diff);
    if (max_diff > 0.1f) {
        printf(" at [%d,%d] (expected %.0f, got %.0f)",
               worst_i, worst_j, h_C_ref[worst_i * 8 + worst_j], h_C_gpu[worst_i * 8 + worst_j]);
    }
    printf("\n");

    bool passed = max_diff < 0.1f;
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

bool test_trans_identity() {
    printf("\n=== Test: Trans - Identity Matrix ===\n");

    half h_A[16 * 16] = {};  // weights
    half h_B[16 * 8] = {};   // activations
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

    test_mma_trans_kernel<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Actual C (rows 0-7):\n");
    for (int i = 0; i < 8; i++) {
        printf("  row %d: ", i);
        for (int j = 0; j < 8; j++) printf("%4.1f ", h_C_gpu[i * 8 + j]);
        printf("\n");
    }

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

bool test_trans_unique() {
    printf("\n=== Test: Trans - Unique Values ===\n");

    half h_A[16 * 16];  // weights
    half h_B[16 * 8];   // activations
    float h_C_gpu[16 * 8] = {};
    float h_C_ref[16 * 8] = {};

    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 16; k++) {
            h_A[i * 16 + k] = __float2half((float)(i + 1));
        }
    }
    for (int k = 0; k < 16; k++) {
        for (int j = 0; j < 8; j++) {
            h_B[k * 8 + j] = __float2half((float)(j + 1));
        }
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

    printf("Expected C[i,j] = 16*(i+1)*(j+1):\n");
    printf("  C[0,0]=%.0f, C[4,4]=%.0f, C[8,7]=%.0f\n",
           h_C_ref[0], h_C_ref[4*8+4], h_C_ref[8*8+7]);

    printf("Actual:\n");
    printf("  C[0,0]=%.0f, C[4,4]=%.0f, C[8,7]=%.0f\n",
           h_C_gpu[0], h_C_gpu[4*8+4], h_C_gpu[8*8+7]);

    float max_diff = 0.0f;
    for (int i = 0; i < 16 * 8; i++) {
        float diff = fabsf(h_C_gpu[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Max diff: %.6f\n", max_diff);
    bool passed = max_diff < 0.1f;
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
    // Find SM70 GPU
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    int sm70_device = -1;
    printf("============================================================\n");
    printf("Marlin SM70 MMA Standalone Tests\n");
    printf("============================================================\n");
    printf("Scanning %d GPU(s)...\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("  GPU %d: %s (SM%d%d)\n", i, prop.name, prop.major, prop.minor);
        if (prop.major == 7 && prop.minor == 0 && sm70_device < 0) {
            sm70_device = i;
        }
    }

    if (sm70_device >= 0) {
        printf("\nUsing GPU %d (SM70 V100)\n", sm70_device);
        CUDA_CHECK(cudaSetDevice(sm70_device));
    } else {
        printf("\nWARNING: No SM70 (V100) GPU found!\n");
    }

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Running on: %s (SM%d%d)\n", prop.name, prop.major, prop.minor);

    // Run tests
    int passed = 0;
    int total = 0;

    printf("\n============================================================\n");
    printf("Testing mma_m16n8k16_sm70\n");
    printf("============================================================\n");

    total++; if (test_identity()) passed++;
    total++; if (test_all_ones()) passed++;
    total++; if (test_unique_values()) passed++;

    printf("\n============================================================\n");
    printf("Testing mma_m16n8k16_sm70_trans\n");
    printf("============================================================\n");

    total++; if (test_trans_identity()) passed++;
    total++; if (test_trans_unique()) passed++;

    printf("\n============================================================\n");
    printf("Summary: %d/%d tests passed\n", passed, total);
    printf("============================================================\n");

    return (passed == total) ? 0 : 1;
}
