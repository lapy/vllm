# SM70 Input Conversion Function Tests

This directory contains tests for the SM70 MMA input conversion function that transforms input fragments into register values for MMA instructions.

## Extracted Function

The `convert_mma_inputs_m16n8k16_sm70` function has been extracted from `mma_m16n8k16_sm70` in `csrc/quantization/marlin/sm70_mma.h`.

### Function Signature
```cpp
__device__ __forceinline__ void convert_mma_inputs_m16n8k16_sm70(
    const uint32_t* a, const uint32_t* b, SM70_MMA_Inputs& inputs);
```

### Input
- `a[0..15]`: 16 uint32_t values containing A matrix fragments
  - `a[0..7]`: Top rows, K pairs 0-1, 2-3, 4-5, 6-7, 8-9, 10-11, 12-13, 14-15
  - `a[8..15]`: Bottom rows, same K pairs
- `b[0..7]`: 8 uint32_t values containing B matrix fragments
  - K pairs: 0-1, 2-3, 4-5, 6-7, 8-9, 10-11, 12-13, 14-15

### Output
- `SM70_MMA_Inputs` structure containing:
  - **A matrix registers (16 total)**:
    - `ak0, ak2, ak4, ak6`: Top rows, K0-7 (direct from `a[0..3]`)
    - `ak8, ak10, ak12, ak14`: Bottom rows, K0-7 (direct from `a[8..11]`)
    - `ab0, ab2, ab4, ab6`: Top rows, K8-15 (shuffled from `a[4..7]` with mask 4)
    - `ab8, ab10, ab12, ab14`: Bottom rows, K8-15 (shuffled from `a[12..15]` with mask 4)
  - **B matrix registers (16 total)**:
    - `bk0_0, bk2_0`: K0-1 (gathered from `b[0]`)
    - `bk0_0_h, bk2_0_h`: K2-3 (gathered from `b[1]`)
    - `bk0_8, bk2_8`: K4-5 (gathered from `b[2]`)
    - `bk0_8_h, bk2_8_h`: K6-7 (gathered from `b[3]`)
    - `bk0_16, bk2_16`: K8-9 (gathered from `b[4]`)
    - `bk0_16_h, bk2_16_h`: K10-11 (gathered from `b[5]`)
    - `bk0_24, bk2_24`: K12-13 (gathered from `b[6]`)
    - `bk0_24_h, bk2_24_h`: K14-15 (gathered from `b[7]`)

## Conversion Logic

### A Matrix Conversion
1. **Direct mapping (K0-7)**:
   - `ak0 = a[0]`, `ak2 = a[1]`, `ak4 = a[2]`, `ak6 = a[3]` (top rows)
   - `ak8 = a[8]`, `ak10 = a[9]`, `ak12 = a[10]`, `ak14 = a[11]` (bottom rows)

2. **Shuffle-based mapping (K8-15)**:
   - Uses `__shfl_xor_sync` with mask 4 to reorganize data for SM70 tensor core layout
   - `ab0 = __shfl_xor_sync(0xffffffff, a[4], 4)` (top, K8-9)
   - `ab2 = __shfl_xor_sync(0xffffffff, a[5], 4)` (top, K10-11)
   - `ab4 = __shfl_xor_sync(0xffffffff, a[6], 4)` (top, K12-13)
   - `ab6 = __shfl_xor_sync(0xffffffff, a[7], 4)` (top, K14-15)
   - Similar for bottom rows from `a[12..15]`

### B Matrix Conversion
1. **Base calculation**:
   - `base = (threadIdx.x & 10) | ((threadIdx.x & 1) << 3)`
   - Determines which threads need which B values

2. **Warp shuffle gathering**:
   - Uses `__shfl_sync` with `SHFL_LANE(base, offset)` to gather B values from different threads
   - Offsets: 0,1 (K0-1), 8,9 (K4-5), 16,17 (K8-9), 24,25 (K12-13)
   - Uses `pack_cols` to combine values from two threads

3. **K-step mapping**:
   - K0-1: from `b[0]` with offsets 0,1
   - K2-3: from `b[1]` with offsets 0,1
   - K4-5: from `b[2]` with offsets 8,9
   - K6-7: from `b[3]` with offsets 8,9
   - K8-9: from `b[4]` with offsets 16,17
   - K10-11: from `b[5]` with offsets 16,17
   - K12-13: from `b[6]` with offsets 24,25
   - K14-15: from `b[7]` with offsets 24,25

## Test Coverage

The test suite (`test_sm70_input_conversion.cu`) verifies:

1. **A Value Extraction**
   - Direct mapping from `a[0..3]` and `a[8..11]` to `ak*` values
   - Correct extraction for multiple threads

2. **B Value Gathering**
   - B values are gathered and packed correctly
   - All K-steps are covered (K0-15)

3. **Input Conversion Consistency**
   - Multiple calls with same inputs produce same outputs
   - Deterministic behavior

4. **Shuffle Effects**
   - Shuffle operations (`__shfl_xor_sync`) are working
   - `ab*` values come from shuffled data

## Running the Tests

### CUDA Test (Direct)
```bash
nvcc -arch=sm_70 -I. -Icsrc/quantization/marlin \
  tests/kernels/quantization/test_sm70_input_conversion.cu \
  -o /tmp/test_sm70_input_conversion && /tmp/test_sm70_input_conversion
```

### Expected Output
```
=== SM70 Input Conversion Tests ===
Device: Tesla V100-SXM2-32GB (SM7.0)
Testing A value extraction...
  PASSED: A value extraction
Testing B value gathering...
  PASSED: B value gathering
Testing input conversion consistency...
  PASSED: input conversion consistency
Testing shuffle effects on A values...
  Detected shuffle effects in 32 threads
  PASSED: shuffle effects test

ALL INPUT CONVERSION TESTS PASSED
```

## Key Properties Verified

1. **Correctness**: A and B values are extracted/gathered correctly
2. **Completeness**: All K-steps (K0-15) are covered
3. **Consistency**: Deterministic conversion across multiple calls
4. **Shuffle Operation**: Warp shuffle operations work as expected

## Integration

This conversion function is used by `mma_m16n8k16_sm70` to prepare input fragments for the MMA instructions. The extracted function allows:
- Independent testing of the conversion logic
- Easier debugging of input preparation issues
- Clear separation of concerns (conversion vs. MMA execution)
