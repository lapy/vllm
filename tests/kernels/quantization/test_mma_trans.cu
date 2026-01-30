#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

// Test the transposed MMA for SM70
// mma_trans computes: C += weights * activations
// where weights are stored in (b, b2) and activations in (a)

// Reference: CPU computation
void reference_matmul_trans(
    const half* weights_b,   // First half of weights (2 regs per thread)
    const half* weights_b2,  // Second half of weights
    const half* activations_a, // Activations (2 regs per thread)
    float* output_c,          // Output (4 floats per thread)
    int lane)
{
    // Reconstruct the full matrices from fragment layout
    
    // For mma.m16n8k16 transposed:
    // - weights (b, b2) form the A operand: 16x16 matrix
    // - activations (a) form the B operand: 16x8 matrix
    // - output (c) is: 16x8 matrix
    
    // A operand fragment layout (4 regs):
    // Thread t: reg[i] contains A[t/4, (t%4)*2+i*4] and A[t/4, (t%4)*2+i*4+1]
    // Wait, that's not quite right either...
    
    // Let me use the documented PTX m16n8k16 layout:
    // A fragment (row-major): 
    //   Thread groupId = t / 4, localId = t % 4
    //   reg[0] = {A[groupId, localId*2], A[groupId, localId*2+1]}
    //   reg[1] = {A[groupId, localId*2+8], A[groupId, localId*2+9]}
    //   reg[2] = {A[groupId+8, localId*2], A[groupId+8, localId*2+1]}
    //   reg[3] = {A[groupId+8, localId*2+8], A[groupId+8, localId*2+9]}
    //
    // B fragment:
    //   reg[0] = {B[localId*2, groupId], B[localId*2+1, groupId]}
    //   reg[1] = {B[localId*2+8, groupId], B[localId*2+9, groupId]}
    
    int groupId = lane / 4;
    int localId = lane % 4;
    
    // Reconstruct full 16x16 A matrix (weights)
    half A[16][16];
    for (int t = 0; t < 32; t++) {
        int gid = t / 4;
        int lid = t % 4;
        
        // reg[0] from b (thread t's b[0])
        int b_idx = t * 2;  // Each thread contributes 2 halves per reg
        A[gid][lid * 2] = weights_b[b_idx];
        A[gid][lid * 2 + 1] = weights_b[b_idx + 1];
        
        // reg[1] from b (thread t's b[1])
        A[gid][lid * 2 + 8] = weights_b[b_idx + 2];  // Next 2 halves
        A[gid][lid * 2 + 9] = weights_b[b_idx + 3];
        
        // reg[2] from b2 (thread t's b2[0])
        int b2_idx = t * 2;
        A[gid + 8][lid * 2] = weights_b2[b2_idx];
        A[gid + 8][lid * 2 + 1] = weights_b2[b2_idx + 1];
        
        // reg[3] from b2 (thread t's b2[1])
        A[gid + 8][lid * 2 + 8] = weights_b2[b2_idx + 2];
        A[gid + 8][lid * 2 + 9] = weights_b2[b2_idx + 3];
    }
    
    // Reconstruct full 16x8 B matrix (activations)
    half B[16][8];
    for (int t = 0; t < 32; t++) {
        int gid = t / 4;
        int lid = t % 4;
        
        int a_idx = t * 2;
        // reg[0]
        B[lid * 2][gid] = activations_a[a_idx];
        B[lid * 2 + 1][gid] = activations_a[a_idx + 1];
        
        // reg[1]
        B[lid * 2 + 8][gid] = activations_a[a_idx + 2];
        B[lid * 2 + 9][gid] = activations_a[a_idx + 3];
    }
    
    // Compute C = A * B (16x16 * 16x8 = 16x8)
    float C[16][8];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 16; k++) {
                C[i][j] += __half2float(A[i][k]) * __half2float(B[k][j]);
            }
        }
    }
    
    // Extract this thread's output fragment
    // C fragment layout:
    //   c[0] = C[groupId, localId*2]
    //   c[1] = C[groupId, localId*2+1]
    //   c[2] = C[groupId+8, localId*2]
    //   c[3] = C[groupId+8, localId*2+1]
    output_c[0] = C[groupId][localId * 2];
    output_c[1] = C[groupId][localId * 2 + 1];
    output_c[2] = C[groupId + 8][localId * 2];
    output_c[3] = C[groupId + 8][localId * 2 + 1];
}

// The SM70 emulation function - FIXED
__device__ void mma_m16n8k16_sm70_trans(
    const uint32_t* marlin_a,  // activations (2 regs)
    const uint32_t* marlin_b,  // weights part 1 (2 regs)
    const uint32_t* marlin_b2, // weights part 2 (2 regs)
    float* frag_c)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    const int out_row = lane / 4;
    const int col_pair = lane % 4;
    const int out_col0 = col_pair * 2;
    const int out_col1 = col_pair * 2 + 1;
    
    float sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
    
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        int a_k_pair = (k / 2) % 4;
        int a_thread = out_row * 4 + a_k_pair;
        int a_half = k % 2;
        
        // Get A values from weights
        // PTX layout: b[0] = rows 0-7 k<8, b2[0] = rows 8-15 k<8
        //             b[1] = rows 0-7 k>=8, b2[1] = rows 8-15 k>=8
        uint32_t a_top_raw, a_bot_raw;
        if (k < 8) {
            a_top_raw = __shfl_sync(FULL_MASK, marlin_b[0], a_thread);   // rows 0-7
            a_bot_raw = __shfl_sync(FULL_MASK, marlin_b2[0], a_thread);  // rows 8-15
        } else {
            a_top_raw = __shfl_sync(FULL_MASK, marlin_b[1], a_thread);   // rows 0-7
            a_bot_raw = __shfl_sync(FULL_MASK, marlin_b2[1], a_thread);  // rows 8-15
        }
        
        half2 a_top_h2 = *reinterpret_cast<half2*>(&a_top_raw);
        half2 a_bot_h2 = *reinterpret_cast<half2*>(&a_bot_raw);
        
        float a_top = (a_half == 0) ? __half2float(__low2half(a_top_h2)) 
                                    : __half2float(__high2half(a_top_h2));
        float a_bot = (a_half == 0) ? __half2float(__low2half(a_bot_h2))
                                    : __half2float(__high2half(a_bot_h2));
        
        // Get B values from activations
        int b_k_pair = (k / 2) % 4;
        int b_thread0 = out_col0 * 4 + b_k_pair;
        int b_thread1 = out_col1 * 4 + b_k_pair;
        int b_reg_idx = (k < 8) ? 0 : 1;
        int b_half = k % 2;
        
        uint32_t b0_raw = __shfl_sync(FULL_MASK, marlin_a[b_reg_idx], b_thread0);
        uint32_t b1_raw = __shfl_sync(FULL_MASK, marlin_a[b_reg_idx], b_thread1);
        
        half2 b0_h2 = *reinterpret_cast<half2*>(&b0_raw);
        half2 b1_h2 = *reinterpret_cast<half2*>(&b1_raw);
        
        float b0_val = (b_half == 0) ? __half2float(__low2half(b0_h2))
                                     : __half2float(__high2half(b0_h2));
        float b1_val = (b_half == 0) ? __half2float(__low2half(b1_h2))
                                     : __half2float(__high2half(b1_h2));
        
        sum00 += a_top * b0_val;
        sum01 += a_top * b1_val;
        sum10 += a_bot * b0_val;
        sum11 += a_bot * b1_val;
    }
    
    frag_c[0] += sum00;
    frag_c[1] += sum01;
    frag_c[2] += sum10;
    frag_c[3] += sum11;
}

// Native mma_trans (for SM80+)
__global__ void test_mma_trans_native(
    const uint32_t* a,   // activations
    const uint32_t* b,   // weights part 1
    const uint32_t* b2,  // weights part 2
    float* c)
{
    int tid = threadIdx.x;
    
    // Load fragments
    uint32_t frag_a[2] = {a[tid * 2], a[tid * 2 + 1]};
    uint32_t frag_b[2] = {b[tid * 2], b[tid * 2 + 1]};
    uint32_t frag_b2[2] = {b2[tid * 2], b2[tid * 2 + 1]};
    float frag_c[4] = {0, 0, 0, 0};
    
    // Native MMA
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(frag_c[0]), "=f"(frag_c[1]), "=f"(frag_c[2]), "=f"(frag_c[3])
        : "r"(frag_b[0]), "r"(frag_b2[0]), "r"(frag_b[1]), "r"(frag_b2[1]),
          "r"(frag_a[0]), "r"(frag_a[1]),
          "f"(frag_c[0]), "f"(frag_c[1]), "f"(frag_c[2]), "f"(frag_c[3]));
    
    // Store output
    c[tid * 4 + 0] = frag_c[0];
    c[tid * 4 + 1] = frag_c[1];
    c[tid * 4 + 2] = frag_c[2];
    c[tid * 4 + 3] = frag_c[3];
}

// Emulated mma_trans
__global__ void test_mma_trans_emulated(
    const uint32_t* a,
    const uint32_t* b,
    const uint32_t* b2,
    float* c)
{
    int tid = threadIdx.x;
    
    uint32_t frag_a[2] = {a[tid * 2], a[tid * 2 + 1]};
    uint32_t frag_b[2] = {b[tid * 2], b[tid * 2 + 1]};
    uint32_t frag_b2[2] = {b2[tid * 2], b2[tid * 2 + 1]};
    float frag_c[4] = {0, 0, 0, 0};
    
    mma_m16n8k16_sm70_trans(frag_a, frag_b, frag_b2, frag_c);
    
    c[tid * 4 + 0] = frag_c[0];
    c[tid * 4 + 1] = frag_c[1];
    c[tid * 4 + 2] = frag_c[2];
    c[tid * 4 + 3] = frag_c[3];
}

int main() {
    // Test 5: Full test with varying both activations and weights
    uint32_t h_a[64], h_b[64], h_b2[64];
    
    for (int t = 0; t < 32; t++) {
        // activations: sequential
        half a0 = __float2half((float)(t * 4));
        half a1 = __float2half((float)(t * 4 + 1));
        half a2 = __float2half((float)(t * 4 + 2));
        half a3 = __float2half((float)(t * 4 + 3));
        half2 h2_a0 = __halves2half2(a0, a1);
        half2 h2_a1 = __halves2half2(a2, a3);
        h_a[t * 2] = *reinterpret_cast<uint32_t*>(&h2_a0);
        h_a[t * 2 + 1] = *reinterpret_cast<uint32_t*>(&h2_a1);
        
        // weights: different per register
        half b0 = __float2half((float)(t * 10));
        half b1 = __float2half((float)(t * 10 + 1));
        half b2_val = __float2half((float)(t * 10 + 2));
        half b3 = __float2half((float)(t * 10 + 3));
        half2 h2_b0 = __halves2half2(b0, b1);
        half2 h2_b1 = __halves2half2(b2_val, b3);
        h_b[t * 2] = *reinterpret_cast<uint32_t*>(&h2_b0);
        h_b[t * 2 + 1] = *reinterpret_cast<uint32_t*>(&h2_b1);
        
        half b20 = __float2half((float)(t * 10 + 4));
        half b21 = __float2half((float)(t * 10 + 5));
        half b22 = __float2half((float)(t * 10 + 6));
        half b23 = __float2half((float)(t * 10 + 7));
        half2 h2_b20 = __halves2half2(b20, b21);
        half2 h2_b21 = __halves2half2(b22, b23);
        h_b2[t * 2] = *reinterpret_cast<uint32_t*>(&h2_b20);
        h_b2[t * 2 + 1] = *reinterpret_cast<uint32_t*>(&h2_b21);
    }
    
    uint32_t *d_a, *d_b, *d_b2;
    float *d_c_native, *d_c_emulated;
    
    cudaMalloc(&d_a, 64 * sizeof(uint32_t));
    cudaMalloc(&d_b, 64 * sizeof(uint32_t));
    cudaMalloc(&d_b2, 64 * sizeof(uint32_t));
    cudaMalloc(&d_c_native, 32 * 4 * sizeof(float));
    cudaMalloc(&d_c_emulated, 32 * 4 * sizeof(float));
    
    cudaMemcpy(d_a, h_a, 64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, 64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    test_mma_trans_native<<<1, 32>>>(d_a, d_b, d_b2, d_c_native);
    test_mma_trans_emulated<<<1, 32>>>(d_a, d_b, d_b2, d_c_emulated);
    
    cudaDeviceSynchronize();
    
    float h_native[128], h_emulated[128];
    cudaMemcpy(h_native, d_c_native, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_emulated, d_c_emulated, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Testing mma_m16n8k16_sm70_trans\n\n");
    
    int mismatches = 0;
    float max_diff = 0;
    for (int i = 0; i < 128; i++) {
        float diff = fabs(h_native[i] - h_emulated[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.1f) {
            if (mismatches < 20) {
                int lane = i / 4;
                int reg = i % 4;
                printf("MISMATCH lane=%d reg=%d: native=%f emulated=%f diff=%f\n",
                       lane, reg, h_native[i], h_emulated[i], diff);
            }
            mismatches++;
        }
    }
    
    if (mismatches == 0) {
        printf("All 128 values match! Max diff: %f\n", max_diff);
    } else {
        printf("\nTotal mismatches: %d / 128, max_diff: %f\n", mismatches, max_diff);
    }
    
    printf("\n=== First 8 lanes output ===\n");
    for (int lane = 0; lane < 8; lane++) {
        printf("Lane %2d: native=[%8.2f,%8.2f,%8.2f,%8.2f] emulated=[%8.2f,%8.2f,%8.2f,%8.2f]\n",
               lane,
               h_native[lane * 4], h_native[lane * 4 + 1],
               h_native[lane * 4 + 2], h_native[lane * 4 + 3],
               h_emulated[lane * 4], h_emulated[lane * 4 + 1],
               h_emulated[lane * 4 + 2], h_emulated[lane * 4 + 3]);
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_b2);
    cudaFree(d_c_native);
    cudaFree(d_c_emulated);
    
    return 0;
}
