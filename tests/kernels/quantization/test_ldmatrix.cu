#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

// Test ldmatrix.x2 with MARLIN's exact m_block_size_8 pointer pattern
// 
// Marlin's shared memory layout for m_block_size_8:
// - a_sh_stride = 2 (two int4 per row)
// - 8 rows total (M dimension)
// - Each row has 16 halves (K dimension), stored as 2 consecutive int4
//
// Marlin's a_sh_rd (thread-to-index mapping):
//   a_sh_rd = 2 * (tid % 8) + tid / 8    (for tid < 16)
//   tid 0-7:  indices 0, 2, 4, 6, 8, 10, 12, 14 (rows 0-7, K=0-7)
//   tid 8-15: indices 1, 3, 5, 7, 9, 11, 13, 15 (rows 0-7, K=8-15)
//
// Expected output for lane L:
//   row = L / 4
//   k_pair = L % 4
//   reg[0] = A[row, k_pair*2:k_pair*2+1]   from K=0-7 half
//   reg[1] = A[row, k_pair*2+8:k_pair*2+9] from K=8-15 half

__global__ void test_ldmatrix_x2_native(const half* input, uint32_t* output) {
    // input is row-major 8 rows x 16 cols (128 halves)
    // Reorganize into shared memory with Marlin's int4 layout
    __shared__ half smem[16 * 8];  // 16 int4 = 128 halves
    
    int tid = threadIdx.x;
    
    // Copy to smem: int4[row*2 + k_half] = input[row, k_half*8 : k_half*8+7]
    // But smem stores as consecutive int4s, so smem[idx*8 : idx*8+7]
    // idx = row*2 + k_half, so:
    //   smem[(row*2 + k_half) * 8 + col_in_half] = input[row * 16 + k_half*8 + col_in_half]
    // Simplifying: smem[row*16 + k_half*8 + col] = input[row*16 + k_half*8 + col]
    // Which is just identity! Copy directly.
    for (int i = tid; i < 128; i += 32) {
        smem[i] = input[i];
    }
    __syncthreads();
    
    // Compute pointer using Marlin's formula
    int a_sh_stride = 2;  // two int4 per row
    int a_sh_rd = a_sh_stride * (tid % 8) + tid / 8;  // for tid < 16
    if (tid >= 16) {
        a_sh_rd = a_sh_stride * ((tid - 16) % 8) + (tid - 16) / 8;  // wrap around
    }
    
    const void* ptr = &smem[a_sh_rd * 8];  // each int4 = 8 halves
    
    uint32_t result[2];
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(result[0]), "=r"(result[1])
        : "r"(smem_addr));
    
    output[tid * 2 + 0] = result[0];
    output[tid * 2 + 1] = result[1];
}

// SM70 emulation
__device__ __forceinline__ void ldmatrix_m8n8_x2_sm70_current(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    // Each thread loads 4 uint32 (16 bytes = 8 halves) from its address
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t my_words[4] = {row_ptr[0], row_ptr[1], row_ptr[2], row_ptr[3]};
    
    int out_row = lane / 4;      // 0-7
    int k_pair = lane % 4;       // 0-3
    
    // Source lanes: threads 0-7 have K=0-7 data, threads 8-15 have K=8-15 data
    int src_lane_mat0 = out_row;           // Rows 0-7 from threads 0-7
    int src_lane_mat1 = out_row + 8;       // Same rows from threads 8-15
    
    // Get all 4 words from source lanes
    uint32_t w0_mat0 = __shfl_sync(FULL_MASK, my_words[0], src_lane_mat0);
    uint32_t w1_mat0 = __shfl_sync(FULL_MASK, my_words[1], src_lane_mat0);
    uint32_t w2_mat0 = __shfl_sync(FULL_MASK, my_words[2], src_lane_mat0);
    uint32_t w3_mat0 = __shfl_sync(FULL_MASK, my_words[3], src_lane_mat0);
    
    uint32_t w0_mat1 = __shfl_sync(FULL_MASK, my_words[0], src_lane_mat1);
    uint32_t w1_mat1 = __shfl_sync(FULL_MASK, my_words[1], src_lane_mat1);
    uint32_t w2_mat1 = __shfl_sync(FULL_MASK, my_words[2], src_lane_mat1);
    uint32_t w3_mat1 = __shfl_sync(FULL_MASK, my_words[3], src_lane_mat1);
    
    uint32_t arr_mat0[4] = {w0_mat0, w1_mat0, w2_mat0, w3_mat0};
    uint32_t arr_mat1[4] = {w0_mat1, w1_mat1, w2_mat1, w3_mat1};
    
    dst[0] = arr_mat0[k_pair];
    dst[1] = arr_mat1[k_pair];
}

__global__ void test_ldmatrix_x2_emulated(const half* input, uint32_t* output) {
    __shared__ half smem[16 * 8];
    
    int tid = threadIdx.x;
    
    for (int i = tid; i < 128; i += 32) {
        smem[i] = input[i];
    }
    __syncthreads();
    
    int a_sh_stride = 2;
    int a_sh_rd = a_sh_stride * (tid % 8) + tid / 8;
    if (tid >= 16) {
        a_sh_rd = a_sh_stride * ((tid - 16) % 8) + (tid - 16) / 8;
    }
    
    const void* ptr = &smem[a_sh_rd * 8];
    
    uint32_t result[2];
    ldmatrix_m8n8_x2_sm70_current(result, ptr);
    
    output[tid * 2 + 0] = result[0];
    output[tid * 2 + 1] = result[1];
}

int main() {
    // Create test input: row-major 8x16 matrix
    // A[row, col] = row * 100 + col (for easy tracing)
    half h_input[128];
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 16; col++) {
            float val = (float)(row * 100 + col);
            h_input[row * 16 + col] = __float2half(val);
            // Debug: verify conversion
            if (row == 0 && col < 2) {
                printf("Input[%d,%d] = %f, converted = %f\n", row, col, val, 
                       __half2float(h_input[row * 16 + col]));
            }
        }
    }
    
    half* d_input;
    uint32_t* d_output_native;
    uint32_t* d_output_emulated;
    
    cudaMalloc(&d_input, 128 * sizeof(half));
    cudaMalloc(&d_output_native, 32 * 2 * sizeof(uint32_t));
    cudaMalloc(&d_output_emulated, 32 * 2 * sizeof(uint32_t));
    
    cudaMemcpy(d_input, h_input, 128 * sizeof(half), cudaMemcpyHostToDevice);
    
    test_ldmatrix_x2_native<<<1, 32>>>(d_input, d_output_native);
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("Native kernel error: %s\n", cudaGetErrorString(err1));
    }
    test_ldmatrix_x2_emulated<<<1, 32>>>(d_input, d_output_emulated);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("Emulated kernel error: %s\n", cudaGetErrorString(err2));
    }
    
    cudaDeviceSynchronize();
    
    uint32_t h_native[64], h_emulated[64];
    cudaMemcpy(h_native, d_output_native, 64 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_emulated, d_output_emulated, 64 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    printf("Testing ldmatrix.x2 with MARLIN's m_block_size_8 layout\n");
    printf("Input: A[row, col] = row*100 + col (8x16 matrix)\n");
    printf("Expected output for lane L (row=L/4, k_pair=L%%4):\n");
    printf("  reg[0] = A[row, k_pair*2:k_pair*2+1]     (K=0-7 half)\n");
    printf("  reg[1] = A[row, k_pair*2+8:k_pair*2+9]   (K=8-15 half)\n\n");
    
    int mismatches = 0;
    for (int lane = 0; lane < 32; lane++) {
        for (int reg = 0; reg < 2; reg++) {
            uint32_t nat = h_native[lane * 2 + reg];
            uint32_t emu = h_emulated[lane * 2 + reg];
            
            if (nat != emu) {
                half2 nat_h2 = *reinterpret_cast<half2*>(&nat);
                half2 emu_h2 = *reinterpret_cast<half2*>(&emu);
                float nat_lo = __half2float(__low2half(nat_h2));
                float nat_hi = __half2float(__high2half(nat_h2));
                float emu_lo = __half2float(__low2half(emu_h2));
                float emu_hi = __half2float(__high2half(emu_h2));
                
                if (mismatches < 20) {
                    printf("MISMATCH lane=%2d reg=%d: native=(%.0f,%.0f) emulated=(%.0f,%.0f)\n",
                           lane, reg, nat_lo, nat_hi, emu_lo, emu_hi);
                }
                mismatches++;
            }
        }
    }
    
    if (mismatches == 0) {
        printf("All 64 values match!\n");
    } else {
        printf("\nTotal mismatches: %d / 64\n", mismatches);
    }
    
    printf("\n=== VERIFY EXPECTED VALUES ===\n");
    printf("Lane 0 (row=0, k_pair=0): reg[0]=(%.0f,%.0f) expect A[0,0:1]=(0,1)\n",
           __half2float(*reinterpret_cast<half*>(&h_native[0])),
           __half2float(*(reinterpret_cast<half*>(&h_native[0]) + 1)));
    printf("                          reg[1]=(%.0f,%.0f) expect A[0,8:9]=(8,9)\n",
           __half2float(*reinterpret_cast<half*>(&h_native[1])),
           __half2float(*(reinterpret_cast<half*>(&h_native[1]) + 1)));
    printf("Lane 4 (row=1, k_pair=0): reg[0]=(%.0f,%.0f) expect A[1,0:1]=(100,101)\n",
           __half2float(*reinterpret_cast<half*>(&h_native[8])),
           __half2float(*(reinterpret_cast<half*>(&h_native[8]) + 1)));
    printf("                          reg[1]=(%.0f,%.0f) expect A[1,8:9]=(108,109)\n",
           __half2float(*reinterpret_cast<half*>(&h_native[9])),
           __half2float(*(reinterpret_cast<half*>(&h_native[9]) + 1)));
    
    printf("\n=== Native ldmatrix.x2 output (first 16 lanes) ===\n");
    for (int lane = 0; lane < 16; lane++) {
        int row = lane / 4;
        int k_pair = lane % 4;
        printf("Lane %2d (row=%d,k=%d): ", lane, row, k_pair);
        for (int reg = 0; reg < 2; reg++) {
            uint32_t val = h_native[lane * 2 + reg];
            half2 h2 = *reinterpret_cast<half2*>(&val);
            float lo = __half2float(__low2half(h2));
            float hi = __half2float(__high2half(h2));
            printf("[%5.0f,%5.0f] ", lo, hi);
        }
        printf("\n");
    }
    
    cudaFree(d_input);
    cudaFree(d_output_native);
    cudaFree(d_output_emulated);
    
    return 0;
}
