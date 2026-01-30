#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

// Test ldmatrix.x2 - used when m_block_size_8=true (small M)

__global__ void test_ldmatrix_x2_native(const half* input, uint32_t* output) {
    __shared__ half smem[16 * 8];  // 16 rows x 8 columns = 128 halves for x2
    
    int tid = threadIdx.x;
    
    // Load input to shared memory
    for (int i = tid; i < 128; i += 32) {
        smem[i] = input[i];
    }
    __syncthreads();
    
    // Each thread points to its row
    // For ldmatrix.x2: 16 rows, threads 0-15 point to rows 0-15, threads 16-31 also point to rows 0-15
    const void* ptr = &smem[(tid % 16) * 8];
    
    uint32_t result[2];
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(result[0]), "=r"(result[1])
        : "r"(smem_addr));
    
    output[tid * 2 + 0] = result[0];
    output[tid * 2 + 1] = result[1];
}

// SM70 emulation - fixed implementation
__device__ __forceinline__ void ldmatrix_m8n8_x2_sm70_current(
    uint32_t* dst,
    const void* smem_ptr)
{
    const int lane = threadIdx.x % 32;
    const uint32_t FULL_MASK = 0xFFFFFFFF;
    
    // Each thread loads 4 uint32 (16 bytes = 8 halves) from its row
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(smem_ptr);
    uint32_t my_words[4] = {row_ptr[0], row_ptr[1], row_ptr[2], row_ptr[3]};
    
    // For x2: threads 0-15 provide rows 0-15, threads 16-31 wrap around
    // Output mapping: lane -> (src_row, word_idx)
    // reg[0] gets from rows 0-7 (first matrix)
    // reg[1] gets from rows 8-15 (second matrix)
    int out_row = lane / 4;      // 0-7 for 32 threads
    int k_pair = lane % 4;       // Which of 4 words in the row
    
    // Source lanes: first 8 threads (0-7) have rows 0-7, next 8 (8-15) have rows 8-15
    int src_lane_mat0 = out_row;           // Rows 0-7 from threads 0-7
    int src_lane_mat1 = out_row + 8;       // Rows 8-15 from threads 8-15
    
    // Get words from source lanes
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
    
    const void* ptr = &smem[(tid % 16) * 8];
    
    uint32_t result[2];
    ldmatrix_m8n8_x2_sm70_current(result, ptr);
    
    output[tid * 2 + 0] = result[0];
    output[tid * 2 + 1] = result[1];
}

int main() {
    // Create test input: simple pattern to trace
    half h_input[128];
    for (int i = 0; i < 128; i++) {
        int row = i / 8;
        int col = i % 8;
        h_input[i] = __float2half((float)(row * 10 + col));
    }
    
    half* d_input;
    uint32_t* d_output_native;
    uint32_t* d_output_emulated;
    
    cudaMalloc(&d_input, 128 * sizeof(half));
    cudaMalloc(&d_output_native, 32 * 2 * sizeof(uint32_t));
    cudaMalloc(&d_output_emulated, 32 * 2 * sizeof(uint32_t));
    
    cudaMemcpy(d_input, h_input, 128 * sizeof(half), cudaMemcpyHostToDevice);
    
    test_ldmatrix_x2_native<<<1, 32>>>(d_input, d_output_native);
    test_ldmatrix_x2_emulated<<<1, 32>>>(d_input, d_output_emulated);
    
    cudaDeviceSynchronize();
    
    uint32_t h_native[64], h_emulated[64];
    cudaMemcpy(h_native, d_output_native, 64 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_emulated, d_output_emulated, 64 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    printf("Comparing native ldmatrix.x2 vs emulation:\n");
    printf("Input: smem[row * 8 + col] = row * 10 + col (16 rows)\n\n");
    
    int mismatches = 0;
    for (int lane = 0; lane < 32; lane++) {
        for (int reg = 0; reg < 2; reg++) {
            uint32_t nat = h_native[lane * 2 + reg];
            uint32_t emu = h_emulated[lane * 2 + reg];
            
            half2 nat_h2 = *reinterpret_cast<half2*>(&nat);
            half2 emu_h2 = *reinterpret_cast<half2*>(&emu);
            
            float nat_lo = __half2float(__low2half(nat_h2));
            float nat_hi = __half2float(__high2half(nat_h2));
            float emu_lo = __half2float(__low2half(emu_h2));
            float emu_hi = __half2float(__high2half(emu_h2));
            
            if (nat != emu) {
                if (mismatches < 32) {
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
    
    printf("\n=== Native ldmatrix.x2 output ===\n");
    printf("Lane -> [reg0, reg1]\n");
    for (int lane = 0; lane < 16; lane++) {
        printf("Lane %2d: ", lane);
        for (int reg = 0; reg < 2; reg++) {
            uint32_t val = h_native[lane * 2 + reg];
            half2 h2 = *reinterpret_cast<half2*>(&val);
            float lo = __half2float(__low2half(h2));
            float hi = __half2float(__high2half(h2));
            printf("[%3.0f,%3.0f] ", lo, hi);
        }
        printf("\n");
    }
    
    printf("\n=== Emulated ldmatrix.x2 output ===\n");
    printf("Lane -> [reg0, reg1]\n");
    for (int lane = 0; lane < 16; lane++) {
        printf("Lane %2d: ", lane);
        for (int reg = 0; reg < 2; reg++) {
            uint32_t val = h_emulated[lane * 2 + reg];
            half2 h2 = *reinterpret_cast<half2*>(&val);
            float lo = __half2float(__low2half(h2));
            float hi = __half2float(__high2half(h2));
            printf("[%3.0f,%3.0f] ", lo, hi);
        }
        printf("\n");
    }
    
    cudaFree(d_input);
    cudaFree(d_output_native);
    cudaFree(d_output_emulated);
    
    return 0;
}
