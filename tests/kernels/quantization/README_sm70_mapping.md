# SM70 Mapping Function Tests

This directory contains tests for the SM70 tensor core mapping functions used in the MMA emulation library.

## Extracted Functions

The following mapping functions have been extracted from `csrc/quantization/marlin/sm70_mma.h` and are tested:

### 1. `lane_to_quadpair_logical(int lane, int& qp, int& logical_tid)`
Maps a physical lane ID (0-31) to a quadpair (0-3) and logical thread ID (0-7).

### 2. `get_quadpair_lane(int qp, int logical_tid)`
Reverse mapping: converts quadpair and logical thread ID back to physical lane ID.

### 3. `SM70_8x8_32b_mn(int logical_tid, int value_id, int& m, int& n)`
Maps a logical thread ID and value index to matrix coordinates (m, n) within an 8x8 accumulator tile.
Uses empirically corrected lookup tables for SM70 architecture.

### 4. `get_canonical_row_col_m16n8k16(int src_tid, int i, int& row, int& col)`
Main mapping function for m16n8k16 MMA operations:
- Maps (src_tid, i) to (row, col) in a 16x8 output matrix
- QP0 threads map to rows 0-7
- QP2 threads map to rows 8-15
- QP1 and QP3 threads map to (-1, -1) (invalid)

## Test Coverage

The test suite (`test_sm70_mapping.cu`) verifies:

1. **Quadpair Mapping**
   - Round-trip consistency (lane → (qp, log_tid) → lane)
   - Valid range checks for qp and logical_tid

2. **SM70_8x8_32b_mn Range Checks**
   - All (m, n) outputs are within valid 8x8 range

3. **Canonical Mapping Coverage**
   - All 128 positions (16 rows × 8 cols) are covered exactly once
   - QP0 maps to rows 0-7 (64 positions)
   - QP2 maps to rows 8-15 (64 positions)
   - QP1 and QP3 map to invalid positions (128 positions)
   - No duplicate mappings

4. **Mapping Consistency**
   - Multiple calls with same inputs produce same outputs

5. **Device-Side Mapping**
   - Device-side and host-side implementations produce identical results

## Running the Tests

### CUDA Test (Direct)
```bash
nvcc -arch=sm_70 -I. -Icsrc/quantization/marlin \
  tests/kernels/quantization/test_sm70_mapping.cu \
  -o /tmp/test_sm70_mapping && /tmp/test_sm70_mapping
```

### Python Wrapper
```bash
python tests/kernels/quantization/test_sm70_mapping.py
```

### Expected Output
```
=== SM70 Mapping Function Tests ===
Device: Tesla V100-SXM2-32GB (SM7.0)
Testing quadpair mapping...
  PASSED: quadpair mapping
Testing SM70_8x8_32b_mn...
  PASSED: SM70_8x8_32b_mn range checks
Testing canonical mapping coverage...
  QP0 mappings: 64 (expected 64)
  QP2 mappings: 64 (expected 64)
  Invalid mappings: 128 (expected 128)
  PASSED: canonical mapping coverage
Testing canonical mapping consistency...
  PASSED: canonical mapping consistency
Testing device-side mapping...
  PASSED: device-side mapping

ALL MAPPING TESTS PASSED
```

## Key Properties Verified

1. **Completeness**: Every position in the 16×8 output matrix is mapped exactly once
2. **Correctness**: QP0/QP2 correctly map to top/bottom halves
3. **Uniqueness**: No two (src_tid, i) pairs map to the same (row, col)
4. **Consistency**: Deterministic mapping across multiple calls
5. **Device Compatibility**: Host and device implementations match

## Integration

These mapping functions are critical for the SM70 MMA emulation:
- Used in `sm70_extract_accumulators_fp32_canonical` to gather accumulator values
- Ensures correct reconstruction of the 16×8 output matrix from warp-level accumulators
- Validated against the reference implementation in `tests/validate_sm70_mma.cu`
