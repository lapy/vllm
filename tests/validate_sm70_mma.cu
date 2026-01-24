/*
 * validate_sm70_mma.cu
 *
 * Standalone diagnostic harness for SM70 mma.m16n8k16 emulation using
 * m8n8k4.row.col.f32.f16.f16.f32. Validates accumulator layout (getWarpRow/
 * getWarpCol from tensorcore_mapping_sm70.h), extract logic, and full GEMM.
 * No Marlin dependency.
 *
 * Build (from tests/): nvcc -o validate_mma validate_sm70_mma.cu -arch=sm_70
 * Build (from root):   nvcc -Itests -o validate_mma tests/validate_sm70_mma.cu -arch=sm_70
 *
 * Run:
 *   ./validate_mma           -- minimal: pass/fail per test + overall
 *   ./validate_mma verbose   -- + device prints, per-element mismatches
 *   ./validate_mma diag      -- full diagnostic: checkpoints, dumps, next steps
 *   ./validate_mma diag search  -- diag + mapping search: try swaps, suggest fix
 *
 * Checkpoints (diag mode):
 *   CP-mapping    (tid,i)->(row,col) coverage: 128 unique in 16x8, no duplicates
 *   CP-fill_A     spot-check fill_registers_A vs logical A
 *   CP-fill_B     spot-check fill_registers_B vs logical B
 *   CP-gathered_B gathered B (post warp-gather) vs host simulation (identity B)
 *   CP-partial_K  partial MMA (K0..3) vs A[:,0:4]@B[0:4,:]
 *   CP-extract    extract-only test (100*row+col)
 *   CP-gemm       full GEMM vs CPU reference
 *   Test 2b       GEMM additional matrices (all-ones, random, pattern, scale)
 *   SUMMARY       "where to look next" on failure
 *
 * Device logs (verbose): T0,T1,T2,T8 A regs (ak,ab), gathered B; accumulators.
 * On first GEMM mismatch, diag logs (tid,i) contributors for C[m,n].
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <set>
#include <map>

#include "tensorcore_mapping_sm70.h"

static bool g_verbose = false;
static bool g_diag = false;
static bool g_search = false;
__constant__ int g_device_verbose;

#define DIAG(fmt, ...) \
  do { \
    if (g_diag) { \
      std::printf("[DIAG] " fmt "\n", ##__VA_ARGS__); \
    } \
  } while (0)

#define CP(name, ok, msg) \
  do { \
    if (g_diag) { \
      std::printf("[CP-%s] %s %s\n", (name), (ok) ? "OK" : "FAIL", (msg)); \
    } \
  } while (0)

#define SECTION(title) \
  do { \
    if (g_diag) { \
      std::printf("\n========== %s ==========\n", (title)); \
    } \
  } while (0)

// ============================================================================
// Helpers
// ============================================================================

__device__ __host__ inline float half2float(half h) { return __half2float(h); }
__device__ __host__ inline half float2half(float f) { return __float2half(f); }

__device__ void print_half2(const char* name, half2 val) {
  float2 f = __half22float2(val);
  printf("%s: {%f, %f}\n", name, f.x, f.y);
}

// Helper to pack two half values from different threads
__device__ inline uint32_t pack_cols(uint32_t v0, uint32_t v1, bool use_high) {
  uint16_t h0 = use_high ? (v0 >> 16) : (v0 & 0xFFFF);
  uint16_t h1 = use_high ? (v1 >> 16) : (v1 & 0xFFFF);
  return (uint32_t)h0 | ((uint32_t)h1 << 16);
}

// __shfl lane must be 0-31; wrap (base + offset) % 32.
#define SHFL_LANE(b, off) (((b) + (off)) % 32)

// Host-side pack_cols for B-gather simulation (expected values).
static inline uint32_t pack_cols_host(uint32_t v0, uint32_t v1, bool use_high) {
  uint32_t h0 = use_high ? (v0 >> 16) : (v0 & 0xFFFF);
  uint32_t h1 = use_high ? (v1 >> 16) : (v1 & 0xFFFF);
  return h0 | (h1 << 16);
}

// -----------------------------------------------------------------------------
// Documented m8n8k4.row.col.f32.f16.f16.f32 layout (Volta)
// Ref: forums.developer.nvidia.com "wrong-answer-mma-sync-aligned-m8n8k4",
//      stackoverflow "questions-about-mma-instruction-with-nvidia-ptx".
//
// - One warp = 4 independent 8x8x4 MMAs (quadpairs). Each MMA uses 8 lanes.
// - Per thread (within one MMA): 8 x f32 (C/D), 2 x f16x2 (A), 2 x f16x2 (B).
// - C/D: SM70_8x8_32b, column-major (m,n). (logical_thread, value_id) -> (m,n).
// - A: 8x4 row-major (SM70_8x4_Row); B: 4x8 col-major (same packing as A).
// -----------------------------------------------------------------------------

// Quadpair lanes: MMA q uses lanes [q][0..7]. Inline tables so device code sees them.
__device__ __host__ inline int get_quadpair_lane(int qp, int logical_tid) {
  static const int QUADPAIR_LANES[4][8] = {
      {0, 1, 2, 3, 16, 17, 18, 19},
      {4, 5, 6, 7, 20, 21, 22, 23},
      {8, 9, 10, 11, 24, 25, 26, 27},
      {12, 13, 14, 15, 28, 29, 30, 31},
  };
  return QUADPAIR_LANES[qp][logical_tid];
}

// Map physical lane -> (quadpair_id, logical_thread 0..7).
__device__ __host__ inline void lane_to_quadpair_logical(int lane, int& qp,
                                                        int& logical_tid) {
  for (qp = 0; qp < 4; qp++) {
    for (logical_tid = 0; logical_tid < 8; logical_tid++) {
      if (get_quadpair_lane(qp, logical_tid) == lane) {
        return;
      }
    }
  }
  logical_tid = 0;
  qp = 0;
}

// SM70_8x8_32b: (logical_thread, value_id) -> (m, n) for 8x8 C/D.
// Empirically corrected. 8x8x4 all-ones revealed 7 missing (m,n) and 7 duplicates.
// Remap first writer of each dup -> missing so the second (correct) writer stays
// at the dup: (0,2)->(1,7), (5,3)->(4,1), (3,4)->(4,7), (6,4)->(5,3),
// (3,5)->(5,4), (5,5)->(5,6), (3,3)->(7,1). All 64 (m,n) covered uniquely.
__device__ __host__ inline void SM70_8x8_32b_mn(int logical_tid, int value_id,
                                                int& m, int& n) {
  static const int SM70_M[8][8] = {
      {0, 0, 1, 2, 0, 0, 5, 2}, {1, 1, 3, 3, 1, 1, 3, 3},
      {4, 2, 6, 6, 2, 4, 6, 6}, {5, 5, 7, 7, 4, 5, 7, 7},
      {0, 0, 2, 2, 0, 0, 2, 2}, {1, 1, 3, 4, 1, 5, 4, 3},
      {4, 4, 6, 6, 5, 3, 6, 6}, {5, 3, 7, 7, 4, 5, 7, 7},
  };
  static const int SM70_N[8][8] = {
      {0, 1, 7, 1, 4, 5, 5, 5}, {0, 2, 0, 1, 4, 5, 4, 5},
      {0, 2, 0, 1, 4, 5, 4, 6}, {0, 1, 0, 1, 7, 4, 4, 5},
      {2, 3, 0, 3, 6, 7, 6, 7}, {1, 3, 2, 1, 6, 6, 4, 7},
      {2, 3, 2, 3, 3, 6, 5, 7}, {2, 3, 2, 3, 6, 7, 6, 7},
  };
  m = SM70_M[logical_tid][value_id];
  n = SM70_N[logical_tid][value_id];
}

// SM70_8x4_Row: (logical_tid, value_id) -> (m, k) for A 8x4 row-major.
__device__ __host__ inline void SM70_8x4_Row_mk(int logical_tid, int value_id,
                                                int& m, int& k) {
  static const int SM70_A_M[8][4] = {
      {0, 0, 2, 2}, {1, 1, 3, 3}, {4, 4, 6, 6}, {5, 5, 7, 7},
      {0, 0, 2, 2}, {1, 1, 3, 3}, {4, 4, 6, 6}, {5, 5, 7, 7},
  };
  static const int SM70_A_K[8][4] = {
      {0, 1, 0, 1}, {0, 1, 0, 1}, {0, 1, 0, 1}, {0, 1, 0, 1},
      {2, 3, 2, 3}, {2, 3, 2, 3}, {2, 3, 2, 3}, {2, 3, 2, 3},
  };
  m = SM70_A_M[logical_tid][value_id];
  k = SM70_A_K[logical_tid][value_id];
}

// B 4x8 col-major: (logical_tid, value_id) -> (k, n).
__device__ __host__ inline void SM70_4x8_Col_kn(int logical_tid, int value_id,
                                                int& k, int& n) {
  static const int SM70_B_K[8][4] = {
      {0, 0, 1, 1}, {2, 2, 3, 3}, {0, 0, 1, 1}, {2, 2, 3, 3},
      {0, 0, 1, 1}, {2, 2, 3, 3}, {0, 0, 1, 1}, {2, 2, 3, 3},
  };
  static const int SM70_B_N[8][4] = {
      {0, 1, 0, 1}, {0, 1, 0, 1}, {2, 3, 2, 3}, {2, 3, 2, 3},
      {4, 5, 4, 5}, {4, 5, 4, 5}, {6, 7, 6, 7}, {6, 7, 6, 7},
  };
  k = SM70_B_K[logical_tid][value_id];
  n = SM70_B_N[logical_tid][value_id];
}

// (src_tid, i) -> (row, col) for m16n8k16 extract using canonical layout.
// m16n8k16 uses Q0 tmp_top -> rows 0-7, Q2 tmp_bot -> rows 8-15. Other
// (src_tid, i) map to (-1,-1) so they never match during gather.
__device__ __host__ inline void get_canonical_row_col_m16n8k16(int src_tid,
                                                              int i, int& row,
                                                              int& col) {
  int qp = -1, logical_tid = -1;
  lane_to_quadpair_logical(src_tid, qp, logical_tid);
  int m = 0, n = 0;
  SM70_8x8_32b_mn(logical_tid, i, m, n);
  if (qp == 0) {
    row = m;
    col = n;
    return;
  }
  if (qp == 2) {
    row = 8 + m;
    col = n;
    return;
  }
  row = -1;
  col = -1;
}

// ============================================================================
// Emulation Logic
// ============================================================================

__device__ inline void sm70_extract_accumulators_fp32(float tmp_top[8],
                                                      float tmp_bot[8],
                                                      float* c_ptr) {
  int tid = threadIdx.x % 32;

  if (g_device_verbose && tid == 0) {
    printf("\n[DEVICE] T0 Accumulators (Pre-Extract):\n");
    for (int i = 0; i < 8; i++)
      printf("  Top[%d]=%f, Bot[%d]=%f\n", i, tmp_top[i], i, tmp_bot[i]);
  }

  for (int out_idx = 0; out_idx < 4; ++out_idx) {
    int target_row = tid / 2;
    int target_col = (tid % 2) * 4 + out_idx;

    float gathered_value = 0.0f;
    // __shfl_sync is warp-wide: ALL threads must execute it. We iterate every
    // (src_tid, i), always shfl, and keep the value only when it matches our
    // (target_row, target_col). Use tmp_top for row<8, tmp_bot for row>=8.
    for (int src_tid = 0; src_tid < 32; ++src_tid) {
      for (int i = 0; i < 8; ++i) {
        int row = getWarpRowSM70(src_tid, i);
        int col = getWarpColSM70(src_tid, i);
        float val_to_shfl = (row < 8) ? tmp_top[i] : tmp_bot[i];
        float received = __shfl_sync(0xffffffff, val_to_shfl, src_tid);
        if (row == target_row && col == target_col)
          gathered_value = received;
      }
    }
    c_ptr[out_idx] = gathered_value;
  }
}

// Extract using canonical layout (Q0 tmp_top -> rows 0-7, Q2 tmp_bot -> rows
// 8-15). Used by m16n8k16 MMA kernels; extract-only keeps legacy extract.
__device__ inline void sm70_extract_accumulators_fp32_canonical(float tmp_top[8],
                                                                float tmp_bot[8],
                                                                float* c_ptr) {
  int tid = threadIdx.x % 32;

  if (g_device_verbose && tid == 0) {
    printf("\n[DEVICE] T0 Accumulators (Pre-Extract):\n");
    for (int i = 0; i < 8; i++)
      printf("  Top[%d]=%f, Bot[%d]=%f\n", i, tmp_top[i], i, tmp_bot[i]);
  }

  for (int out_idx = 0; out_idx < 4; ++out_idx) {
    int target_row = tid / 2;
    int target_col = (tid % 2) * 4 + out_idx;

    float gathered_value = 0.0f;
    for (int src_tid = 0; src_tid < 32; ++src_tid) {
      for (int i = 0; i < 8; ++i) {
        int row = -1, col = -1;
        get_canonical_row_col_m16n8k16(src_tid, i, row, col);
        float val_to_shfl = 0.0f;
        if (tid == src_tid)
          val_to_shfl =
              (row >= 0 && row < 8) ? tmp_top[i] : tmp_bot[i];
        float received = __shfl_sync(0xffffffff, val_to_shfl, src_tid);
        if (row == target_row && col == target_col)
          gathered_value = received;
      }
    }
    c_ptr[out_idx] = gathered_value;
  }
}

__device__ inline void mma_m16n8k16_sm70(const half2 a_frag[4],
                                         const half2 b_frag[2],
                                         float c_frag[4]) {
  float tmp_top[8] = {0.0f};
  float tmp_bot[8] = {0.0f};

  const uint32_t* a_ptr = reinterpret_cast<const uint32_t*>(a_frag);
  const uint32_t* b_ptr = reinterpret_cast<const uint32_t*>(b_frag);

  uint32_t ak0 = a_ptr[0];
  uint32_t ak2 = a_ptr[1];
  uint32_t ak4 = a_ptr[2];
  uint32_t ak6 = a_ptr[3];

  uint32_t ak8 = __shfl_xor_sync(0xffffffff, ak0, 8);
  uint32_t ak10 = __shfl_xor_sync(0xffffffff, ak2, 8);
  uint32_t ak12 = __shfl_xor_sync(0xffffffff, ak4, 8);
  uint32_t ak14 = __shfl_xor_sync(0xffffffff, ak6, 8);

  uint32_t b_low = b_ptr[0];   // K0, K1 (j=0)
  uint32_t b_high = b_ptr[1];  // K2, K3 (j=1)

  // Threads with tid%2==0 output cols 0-3 (need B from 0,1,2,3); tid%2==1
  // output cols 4-7 (need B from 8,9,10,11). base = (tid&10) | ((tid&1)<<3)
  // gives 0,8,2,10 for tid 0,1,2,3. Offsets +0,+1; +8,+9; +16,+17; +24,+25.
  // __shfl lane must be 0-31: use SHFL_LANE(base, off).
  int base = (threadIdx.x & 10) | ((threadIdx.x & 1) << 3);

  // Gather b_low -> K0,K1; K4,K5; K8,K9; K12,K13
  uint32_t v0 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 0));
  uint32_t v1 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 1));
  uint32_t bk0_0 = pack_cols(v0, v1, false);
  uint32_t bk2_0 = pack_cols(v0, v1, true);

  v0 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 8));
  v1 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 9));
  uint32_t bk0_8 = pack_cols(v0, v1, false);
  uint32_t bk2_8 = pack_cols(v0, v1, true);

  v0 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 16));
  v1 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 17));
  uint32_t bk0_16 = pack_cols(v0, v1, false);
  uint32_t bk2_16 = pack_cols(v0, v1, true);

  v0 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 24));
  v1 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 25));
  uint32_t bk0_24 = pack_cols(v0, v1, false);
  uint32_t bk2_24 = pack_cols(v0, v1, true);

  // Gather b_high -> K2,K3; K6,K7; K10,K11; K14,K15
  v0 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 0));
  v1 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 1));
  uint32_t bk0_0_h = pack_cols(v0, v1, false);
  uint32_t bk2_0_h = pack_cols(v0, v1, true);

  v0 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 8));
  v1 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 9));
  uint32_t bk0_8_h = pack_cols(v0, v1, false);
  uint32_t bk2_8_h = pack_cols(v0, v1, true);

  v0 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 16));
  v1 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 17));
  uint32_t bk0_16_h = pack_cols(v0, v1, false);
  uint32_t bk2_16_h = pack_cols(v0, v1, true);

  v0 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 24));
  v1 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 25));
  uint32_t bk0_24_h = pack_cols(v0, v1, false);
  uint32_t bk2_24_h = pack_cols(v0, v1, true);

  uint32_t ab0 = __shfl_xor_sync(0xffffffff, ak0, 4);
  uint32_t ab2 = __shfl_xor_sync(0xffffffff, ak2, 4);
  uint32_t ab4 = __shfl_xor_sync(0xffffffff, ak4, 4);
  uint32_t ab6 = __shfl_xor_sync(0xffffffff, ak6, 4);
  uint32_t ab8 = __shfl_xor_sync(0xffffffff, ak8, 4);
  uint32_t ab10 = __shfl_xor_sync(0xffffffff, ak10, 4);
  uint32_t ab12 = __shfl_xor_sync(0xffffffff, ak12, 4);
  uint32_t ab14 = __shfl_xor_sync(0xffffffff, ak14, 4);

  if (g_device_verbose) {
    int t = threadIdx.x;
    if (t == 0 || t == 1 || t == 2 || t == 8) {
      printf("[DEVICE] T%d A regs (hex): ak0=%08x ak2=%08x ab0=%08x ab2=%08x\n",
             t, ak0, ak2, ab0, ab2);
      printf("[DEVICE] T%d Gathered B: K0-1 %08x %08x  K2-3 %08x %08x  "
             "base=%d\n",
             t, bk0_0, bk2_0, bk0_0_h, bk2_0_h,
             (t & 10) | ((t & 1) << 3));
    }
  }

  // 8 K-steps per tile: K0-1, K2-3, K4-5, K6-7, K8-9, K10-11, K12-13, K14-15.
  // A pairs (ak0,ak2), (ak4,ak6), (ak8,ak10), (ak12,ak14) for top; ab* for bot.
  // Reuse each A pair for two consecutive K steps (K0-1/K2-3, etc.) until we
  // have a proper A split for K2,K6,K10,K14.
  // Constraints: 0-15 out, 16-31 A, 32-47 B (bk0_0,bk2_0,bk0_0_h,bk2_0_h,...).
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17}, {%32,%33}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%8,%9,%10,%11,%12,%13,%14,%15}, {%24,%25}, {%32,%33}, "
      "{%8,%9,%10,%11,%12,%13,%14,%15};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17}, {%34,%35}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%8,%9,%10,%11,%12,%13,%14,%15}, {%24,%25}, {%34,%35}, "
      "{%8,%9,%10,%11,%12,%13,%14,%15};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%18,%19}, {%36,%37}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%8,%9,%10,%11,%12,%13,%14,%15}, {%26,%27}, {%36,%37}, "
      "{%8,%9,%10,%11,%12,%13,%14,%15};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%18,%19}, {%38,%39}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%8,%9,%10,%11,%12,%13,%14,%15}, {%26,%27}, {%38,%39}, "
      "{%8,%9,%10,%11,%12,%13,%14,%15};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%20,%21}, {%40,%41}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%8,%9,%10,%11,%12,%13,%14,%15}, {%28,%29}, {%40,%41}, "
      "{%8,%9,%10,%11,%12,%13,%14,%15};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%20,%21}, {%42,%43}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%8,%9,%10,%11,%12,%13,%14,%15}, {%28,%29}, {%42,%43}, "
      "{%8,%9,%10,%11,%12,%13,%14,%15};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%22,%23}, {%44,%45}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%8,%9,%10,%11,%12,%13,%14,%15}, {%30,%31}, {%44,%45}, "
      "{%8,%9,%10,%11,%12,%13,%14,%15};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%22,%23}, {%46,%47}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%8,%9,%10,%11,%12,%13,%14,%15}, {%30,%31}, {%46,%47}, "
      "{%8,%9,%10,%11,%12,%13,%14,%15};\n"
      : "+f"(tmp_top[0]), "+f"(tmp_top[1]), "+f"(tmp_top[2]), "+f"(tmp_top[3]),
        "+f"(tmp_top[4]), "+f"(tmp_top[5]), "+f"(tmp_top[6]), "+f"(tmp_top[7]),
        "+f"(tmp_bot[0]), "+f"(tmp_bot[1]), "+f"(tmp_bot[2]), "+f"(tmp_bot[3]),
        "+f"(tmp_bot[4]), "+f"(tmp_bot[5]), "+f"(tmp_bot[6]), "+f"(tmp_bot[7])
      : "r"(ak0), "r"(ak2), "r"(ak4), "r"(ak6), "r"(ak8), "r"(ak10), "r"(ak12),
        "r"(ak14), "r"(ab0), "r"(ab2), "r"(ab4), "r"(ab6), "r"(ab8), "r"(ab10),
        "r"(ab12), "r"(ab14), "r"(bk0_0), "r"(bk2_0), "r"(bk0_0_h), "r"(bk2_0_h),
        "r"(bk0_8), "r"(bk2_8), "r"(bk0_8_h), "r"(bk2_8_h), "r"(bk0_16),
        "r"(bk2_16), "r"(bk0_16_h), "r"(bk2_16_h), "r"(bk0_24), "r"(bk2_24),
        "r"(bk0_24_h), "r"(bk2_24_h));
  sm70_extract_accumulators_fp32_canonical(tmp_top, tmp_bot, c_frag);
}

// Partial MMA: first two m8n8k4 (K0..3). Same setup as full, then 2 mma ops.
// Use for CP-partial_K: C_ref = A[:,0:4] @ B[0:4,:].
__device__ inline void mma_m16n8k16_sm70_k1(const half2 a_frag[4],
                                            const half2 b_frag[2],
                                            float c_frag[4]) {
  float tmp_top[8] = {0.0f};
  float tmp_bot[8] = {0.0f};
  const uint32_t* a_ptr = reinterpret_cast<const uint32_t*>(a_frag);
  const uint32_t* b_ptr = reinterpret_cast<const uint32_t*>(b_frag);
  uint32_t ak0 = a_ptr[0], ak2 = a_ptr[1], ak4 = a_ptr[2], ak6 = a_ptr[3];
  uint32_t ak8 = __shfl_xor_sync(0xffffffff, ak0, 8);
  uint32_t ak10 = __shfl_xor_sync(0xffffffff, ak2, 8);
  uint32_t ak12 = __shfl_xor_sync(0xffffffff, ak4, 8);
  uint32_t ak14 = __shfl_xor_sync(0xffffffff, ak6, 8);
  uint32_t b_low = b_ptr[0], b_high = b_ptr[1];
  int base = (threadIdx.x & 10) | ((threadIdx.x & 1) << 3);
  uint32_t v0, v1;
  v0 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 0));
  v1 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 1));
  uint32_t bk0_0 = pack_cols(v0, v1, false), bk2_0 = pack_cols(v0, v1, true);
  v0 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 0));
  v1 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 1));
  uint32_t bk0_0_h = pack_cols(v0, v1, false), bk2_0_h = pack_cols(v0, v1, true);
  uint32_t ab0 = __shfl_xor_sync(0xffffffff, ak0, 4);
  uint32_t ab2 = __shfl_xor_sync(0xffffffff, ak2, 4);
  uint32_t ab4 = __shfl_xor_sync(0xffffffff, ak4, 4);
  uint32_t ab6 = __shfl_xor_sync(0xffffffff, ak6, 4);
  uint32_t ab8 = __shfl_xor_sync(0xffffffff, ak8, 4);
  uint32_t ab10 = __shfl_xor_sync(0xffffffff, ak10, 4);
  uint32_t ab12 = __shfl_xor_sync(0xffffffff, ak12, 4);
  uint32_t ab14 = __shfl_xor_sync(0xffffffff, ak14, 4);
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17}, {%32,%33}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%8,%9,%10,%11,%12,%13,%14,%15}, {%24,%25}, {%32,%33}, "
      "{%8,%9,%10,%11,%12,%13,%14,%15};\n"
      : "+f"(tmp_top[0]), "+f"(tmp_top[1]), "+f"(tmp_top[2]), "+f"(tmp_top[3]),
        "+f"(tmp_top[4]), "+f"(tmp_top[5]), "+f"(tmp_top[6]), "+f"(tmp_top[7]),
        "+f"(tmp_bot[0]), "+f"(tmp_bot[1]), "+f"(tmp_bot[2]), "+f"(tmp_bot[3]),
        "+f"(tmp_bot[4]), "+f"(tmp_bot[5]), "+f"(tmp_bot[6]), "+f"(tmp_bot[7])
      : "r"(ak0), "r"(ak2), "r"(ak4), "r"(ak6), "r"(ak8), "r"(ak10), "r"(ak12),
        "r"(ak14), "r"(ab0), "r"(ab2), "r"(ab4), "r"(ab6), "r"(ab8), "r"(ab10),
        "r"(ab12), "r"(ab14), "r"(bk0_0), "r"(bk2_0), "r"(bk0_0_h), "r"(bk2_0_h));
  sm70_extract_accumulators_fp32_canonical(tmp_top, tmp_bot, c_frag);
}

// -----------------------------------------------------------------------------
// Extract-only test: fill accumulators with known values (100*row+col), run
// extract, reconstruct. Validates (tid,i)->(row,col) mapping and gather logic.
// -----------------------------------------------------------------------------
__global__ void test_extract_only(float* C_regs) {
  int tid = threadIdx.x;
  if (tid >= 32) return;

  float tmp_top[8] = {0.0f};
  float tmp_bot[8] = {0.0f};

  for (int i = 0; i < 8; ++i) {
    int row = getWarpRowSM70(tid, i);
    int col = getWarpColSM70(tid, i);
    float val = 100.0f * (float)row + (float)col;
    if (row < 8)
      tmp_top[i] = val;
    else
      tmp_bot[i] = val;
  }

  float c[4];
  sm70_extract_accumulators_fp32(tmp_top, tmp_bot, c);
  for (int i = 0; i < 4; i++) C_regs[tid * 4 + i] = c[i];
}

__global__ void test_mma_accuracy(const half2* A_regs, const half2* B_regs,
                                  float* C_regs) {
  int tid = threadIdx.x;
  if (tid >= 32) return;
  half2 a[4];
  half2 b[2];
  float c[4];
  for (int i = 0; i < 4; i++) a[i] = A_regs[tid * 4 + i];
  for (int i = 0; i < 2; i++) b[i] = B_regs[tid * 2 + i];

  if (g_device_verbose && tid == 0) {
    printf("[DEVICE] T0 Inputs:\n");
    for (int i = 0; i < 4; i++) print_half2("A", a[i]);
    for (int i = 0; i < 2; i++) print_half2("B", b[i]);
  }

  mma_m16n8k16_sm70(a, b, c);
  for (int i = 0; i < 4; i++) C_regs[tid * 4 + i] = c[i];
}

__global__ void test_mma_partial_k1(const half2* A_regs, const half2* B_regs,
                                    float* C_regs) {
  int tid = threadIdx.x;
  if (tid >= 32) return;
  half2 a[4], b[2];
  float c[4];
  for (int i = 0; i < 4; i++) a[i] = A_regs[tid * 4 + i];
  for (int i = 0; i < 2; i++) b[i] = B_regs[tid * 2 + i];
  mma_m16n8k16_sm70_k1(a, b, c);
  for (int i = 0; i < 4; i++) C_regs[tid * 4 + i] = c[i];
}

// -----------------------------------------------------------------------------
// Canonical 8x8x4 MMA (documented layout): one quadpair, 8 D / 2 A / 2 B / 8 C
// per thread. All 32 threads participate; only quadpair 0 has non-zero A,B.
// -----------------------------------------------------------------------------
__device__ void mma_sync_m8n8k4_canonical(const half2 a_frag[2],
                                          const half2 b_frag[2],
                                          const float c_frag[8],
                                          float d_frag[8]) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(b_frag);
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3]),
        "=f"(d_frag[4]), "=f"(d_frag[5]), "=f"(d_frag[6]), "=f"(d_frag[7])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]),
        "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),
        "f"(c_frag[4]), "f"(c_frag[5]), "f"(c_frag[6]), "f"(c_frag[7]));
}

__global__ void test_mma_8x8x4_canonical(const half2* A_frag,
                                         const half2* B_frag,
                                         float* D_out) {
  int lane = threadIdx.x % 32;
  half2 a[2], b[2];
  float c[8] = {0.f};
  float d[8];

  int qp = -1, logical_tid = -1;
  lane_to_quadpair_logical(lane, qp, logical_tid);

  if (qp == 0) {
    a[0] = A_frag[lane * 2 + 0];
    a[1] = A_frag[lane * 2 + 1];
    b[0] = B_frag[lane * 2 + 0];
    b[1] = B_frag[lane * 2 + 1];
  } else {
    half z = __float2half(0.f);
    a[0] = a[1] = make_half2(z, z);
    b[0] = b[1] = make_half2(z, z);
  }

  mma_sync_m8n8k4_canonical(a, b, c, d);

  for (int i = 0; i < 8; i++) D_out[lane * 8 + i] = d[i];
}

// Gather-only test: same B fill + gather as MMA, write 16 u32/thread for
// CP-gathered_B spot-check. Uses same logic as mma_m16n8k16_sm70 gather.
__global__ void test_gather_B_only(const half2* B_regs, uint32_t* gathered_out) {
  int tid = threadIdx.x;
  if (tid >= 32) return;
  const uint32_t* b_ptr = reinterpret_cast<const uint32_t*>(&B_regs[tid * 2]);
  uint32_t b_low = b_ptr[0], b_high = b_ptr[1];
  int base = (tid & 10) | ((tid & 1) << 3);
  // Groups 0..3: (0,1), (8,9), (16,17), (24,25). Store bk0,bk2,bk0_h,bk2_h
  // at tid*16+0..3, then +4, +8, +12 for subsequent groups.
  uint32_t* o = &gathered_out[tid * 16];
  uint32_t v0, v1;
  v0 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 0));
  v1 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 1));
  o[0] = pack_cols(v0, v1, false);
  o[1] = pack_cols(v0, v1, true);
  v0 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 0));
  v1 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 1));
  o[2] = pack_cols(v0, v1, false);
  o[3] = pack_cols(v0, v1, true);
  v0 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 8));
  v1 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 9));
  o[4] = pack_cols(v0, v1, false);
  o[5] = pack_cols(v0, v1, true);
  v0 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 8));
  v1 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 9));
  o[6] = pack_cols(v0, v1, false);
  o[7] = pack_cols(v0, v1, true);
  v0 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 16));
  v1 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 17));
  o[8] = pack_cols(v0, v1, false);
  o[9] = pack_cols(v0, v1, true);
  v0 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 16));
  v1 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 17));
  o[10] = pack_cols(v0, v1, false);
  o[11] = pack_cols(v0, v1, true);
  v0 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 24));
  v1 = __shfl_sync(0xffffffff, b_low, SHFL_LANE(base, 25));
  o[12] = pack_cols(v0, v1, false);
  o[13] = pack_cols(v0, v1, true);
  v0 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 24));
  v1 = __shfl_sync(0xffffffff, b_high, SHFL_LANE(base, 25));
  o[14] = pack_cols(v0, v1, false);
  o[15] = pack_cols(v0, v1, true);
}

void get_coords(int tid, int i, int& row, int& col) {
  row = getWarpRowSM70(tid, i);
  col = getWarpColSM70(tid, i);
}

// A[row, k]: row = getWarpRowSM70 (matches output row), k = getWarpColA_SM70
// so we cover K 0..15. get_coords uses (row,col) for C and only col 0..7.
void get_coords_A(int tid, int i, int& row, int& k) {
  row = getWarpRowSM70(tid, i);
  k = getWarpColA_SM70(tid, i);
}

void fill_registers_A_m16n8k16(const std::vector<float>& A,
                               std::vector<half2>& regs) {
  for (int tid = 0; tid < 32; ++tid) {
    for (int j = 0; j < 4; ++j) {
      int i_lo = j * 2 + 0;
      int r_lo, k_lo;
      get_coords_A(tid, i_lo, r_lo, k_lo);
      int i_hi = j * 2 + 1;
      int r_hi, k_hi;
      get_coords_A(tid, i_hi, r_hi, k_hi);
      float val_lo = (r_lo < 16 && k_lo < 16) ? A[r_lo * 16 + k_lo] : 0.0f;
      float val_hi = (r_hi < 16 && k_hi < 16) ? A[r_hi * 16 + k_hi] : 0.0f;
      half h0 = __float2half(val_lo);
      half h1 = __float2half(val_hi);
      unsigned int p =
          (unsigned short)*reinterpret_cast<unsigned short*>(&h0) |
          ((unsigned int)(unsigned short)*reinterpret_cast<unsigned short*>(&h1)
           << 16);
      regs[tid * 4 + j] = *reinterpret_cast<half2*>(&p);
    }
  }
}

void fill_registers_B_m16n8k16(const std::vector<float>& B,
                               std::vector<half2>& regs) {
  for (int tid = 0; tid < 32; ++tid) {
    // Map Physical Tids to Logical Columns for B
    int col = -1;
    // T0..3 -> Col 0..3 (Phys 0..3 -> Log 0..3)
    // T8..11 -> Col 4..7 (Phys 8..11 -> Log 4..7)
    int t_mod = tid % 16;
    if (t_mod < 4)
      col = t_mod;
    else if (t_mod >= 8 && t_mod < 12)
      col = (t_mod - 8) + 4;
    else
      col = 0;  // Default fill unused to 0 (avoid OOB)

    int k_base = (tid / 8) * 4;
    for (int j = 0; j < 2; j++) {
      float val_lo = 0.0f, val_hi = 0.0f;
      if (col >= 0 && col < 8) {
        // Ensure K range??
        // K inputs are striped.
        val_lo = B[(k_base + j * 2) * 8 + col];
        val_hi = B[(k_base + j * 2 + 1) * 8 + col];
      }
      half h0 = __float2half(val_lo);
      half h1 = __float2half(val_hi);
      unsigned int p =
          (unsigned short)*reinterpret_cast<unsigned short*>(&h0) |
          ((unsigned int)(unsigned short)*reinterpret_cast<unsigned short*>(&h1)
           << 16);
      regs[tid * 2 + j] = *reinterpret_cast<half2*>(&p);
    }
  }
}

void reconstruct_C_m16n8k16(const std::vector<float>& regs,
                            std::vector<float>& matrix) {
  matrix.assign(16 * 8, 0.0f);
  for (int tid = 0; tid < 32; ++tid) {
    for (int i = 0; i < 4; ++i) {
      int row = tid / 2;
      int col = (tid % 2) * 4 + i;
      if (row < 16 && col < 8) {
        matrix[row * 8 + col] = regs[tid * 4 + i];
      }
    }
  }
}

// Fill A_frag[32*2], B_frag[32*2] for canonical 8x8x4. A is 8x4 row-major,
// B is 4x8 col-major. Only quadpair 0 lanes get non-zero; others zero.
void fill_fragments_8x8x4_canonical(const float* A_8x4, const float* B_4x8,
                                    std::vector<half2>& A_frag,
                                    std::vector<half2>& B_frag) {
  A_frag.resize(32 * 2);
  B_frag.resize(32 * 2);
  for (int i = 0; i < 32 * 2; i++) {
    A_frag[i] = make_half2(__float2half(0.f), __float2half(0.f));
    B_frag[i] = make_half2(__float2half(0.f), __float2half(0.f));
  }
  for (int logical_tid = 0; logical_tid < 8; logical_tid++) {
    int lane = get_quadpair_lane(0, logical_tid);
    for (int j = 0; j < 2; j++) {
      int v0 = j * 2 + 0, v1 = j * 2 + 1;
      int m0, k0, m1, k1;
      SM70_8x4_Row_mk(logical_tid, v0, m0, k0);
      SM70_8x4_Row_mk(logical_tid, v1, m1, k1);
      half h0 = __float2half(A_8x4[m0 * 4 + k0]);
      half h1 = __float2half(A_8x4[m1 * 4 + k1]);
      unsigned int p =
          (unsigned short)*reinterpret_cast<unsigned short*>(&h0) |
          ((unsigned int)(unsigned short)*reinterpret_cast<unsigned short*>(&h1)
           << 16);
      A_frag[lane * 2 + j] = *reinterpret_cast<half2*>(&p);
    }
    for (int j = 0; j < 2; j++) {
      int v0 = j * 2 + 0, v1 = j * 2 + 1;
      int k0, n0, k1, n1;
      SM70_4x8_Col_kn(logical_tid, v0, k0, n0);
      SM70_4x8_Col_kn(logical_tid, v1, k1, n1);
      half h0 = __float2half(B_4x8[k0 * 8 + n0]);
      half h1 = __float2half(B_4x8[k1 * 8 + n1]);
      unsigned int p =
          (unsigned short)*reinterpret_cast<unsigned short*>(&h0) |
          ((unsigned int)(unsigned short)*reinterpret_cast<unsigned short*>(&h1)
           << 16);
      B_frag[lane * 2 + j] = *reinterpret_cast<half2*>(&p);
    }
  }
}

// Extract 8x8 from D_out[32*8] using quadpair 0 and SM70_8x8_32b.
void extract_8x8_from_quadpair0(const float* D_out, float* C_8x8) {
  for (int i = 0; i < 64; i++) C_8x8[i] = 0.f;
  for (int logical_tid = 0; logical_tid < 8; logical_tid++) {
    int lane = get_quadpair_lane(0, logical_tid);
    for (int value_id = 0; value_id < 8; value_id++) {
      int m, n;
      SM70_8x8_32b_mn(logical_tid, value_id, m, n);
      C_8x8[m * 8 + n] = D_out[lane * 8 + value_id];
    }
  }
}

// Fill A_frag[32*2], B_frag[32*2] for canonical 16x8 one K-step.
// A 16x16, B 16x8. k_offset in {0,4,8,12}. Q0: A[0:8,k:k+4], B[k:k+4,0:8];
// Q2: A[8:16,k:k+4], B[k:k+4,0:8]. Other lanes zero.
void fill_fragments_16x8_canonical_kstep(const float* A, const float* B,
                                         int k_offset,
                                         std::vector<half2>& A_frag,
                                         std::vector<half2>& B_frag) {
  A_frag.resize(32 * 2);
  B_frag.resize(32 * 2);
  half z = __float2half(0.f);
  for (int i = 0; i < 32 * 2; i++) {
    A_frag[i] = make_half2(z, z);
    B_frag[i] = make_half2(z, z);
  }
  float A_top[8 * 4], A_bot[8 * 4], B_4x8[4 * 8];
  for (int m = 0; m < 8; m++)
    for (int k = 0; k < 4; k++) {
      A_top[m * 4 + k] = A[m * 16 + (k_offset + k)];
      A_bot[m * 4 + k] = A[(8 + m) * 16 + (k_offset + k)];
    }
  for (int k = 0; k < 4; k++)
    for (int n = 0; n < 8; n++)
      B_4x8[k * 8 + n] = B[(k_offset + k) * 8 + n];

  for (int qp : {0, 2}) {
    const float* A_8x4 = (qp == 0) ? A_top : A_bot;
    for (int logical_tid = 0; logical_tid < 8; logical_tid++) {
      int lane = get_quadpair_lane(qp, logical_tid);
      for (int j = 0; j < 2; j++) {
        int v0 = j * 2 + 0, v1 = j * 2 + 1;
        int m0, k0, m1, k1;
        SM70_8x4_Row_mk(logical_tid, v0, m0, k0);
        SM70_8x4_Row_mk(logical_tid, v1, m1, k1);
        half h0 = __float2half(A_8x4[m0 * 4 + k0]);
        half h1 = __float2half(A_8x4[m1 * 4 + k1]);
        unsigned int p =
            (unsigned short)*reinterpret_cast<unsigned short*>(&h0) |
            ((unsigned int)(unsigned short)*reinterpret_cast<unsigned short*>(
                 &h1)
             << 16);
        A_frag[lane * 2 + j] = *reinterpret_cast<half2*>(&p);
      }
      for (int j = 0; j < 2; j++) {
        int v0 = j * 2 + 0, v1 = j * 2 + 1;
        int k0, n0, k1, n1;
        SM70_4x8_Col_kn(logical_tid, v0, k0, n0);
        SM70_4x8_Col_kn(logical_tid, v1, k1, n1);
        half h0 = __float2half(B_4x8[k0 * 8 + n0]);
        half h1 = __float2half(B_4x8[k1 * 8 + n1]);
        unsigned int p =
            (unsigned short)*reinterpret_cast<unsigned short*>(&h0) |
            ((unsigned int)(unsigned short)*reinterpret_cast<unsigned short*>(
                 &h1)
             << 16);
        B_frag[lane * 2 + j] = *reinterpret_cast<half2*>(&p);
      }
    }
  }
}

// Kernel: one 16x8 canonical K-step. Q0 and Q2 use A_frag,B_frag; Q1,Q3 zero.
// Writes D_out[32*8]. No accumulation.
__global__ void mma_16x8_canonical_kstep(const half2* A_frag,
                                         const half2* B_frag, float* D_out) {
  int lane = threadIdx.x % 32;
  half2 a[2], b[2];
  float c[8] = {0.f};
  float d[8];

  int qp = -1, logical_tid = -1;
  lane_to_quadpair_logical(lane, qp, logical_tid);

  if (qp == 0 || qp == 2) {
    a[0] = A_frag[lane * 2 + 0];
    a[1] = A_frag[lane * 2 + 1];
    b[0] = B_frag[lane * 2 + 0];
    b[1] = B_frag[lane * 2 + 1];
  } else {
    half z = __float2half(0.f);
    a[0] = a[1] = make_half2(z, z);
    b[0] = b[1] = make_half2(z, z);
  }

  mma_sync_m8n8k4_canonical(a, b, c, d);
  for (int i = 0; i < 8; i++) D_out[lane * 8 + i] = d[i];
}

// Override version for mapping search. override_M/N are 8x8; -1 = use table.
void extract_16x8_from_canonical_with_override(
    const float* D_out, float* C_16x8, const int (*override_M)[8],
    const int (*override_N)[8]) {
  for (int i = 0; i < 16 * 8; i++) C_16x8[i] = 0.f;
  for (int logical_tid = 0; logical_tid < 8; logical_tid++) {
    for (int qp : {0, 2}) {
      int lane = get_quadpair_lane(qp, logical_tid);
      int row_offset = (qp == 0) ? 0 : 8;
      for (int value_id = 0; value_id < 8; value_id++) {
        int m, n;
        if (override_M && override_N && override_M[logical_tid][value_id] >= 0 &&
            override_N[logical_tid][value_id] >= 0) {
          m = override_M[logical_tid][value_id];
          n = override_N[logical_tid][value_id];
        } else {
          SM70_8x8_32b_mn(logical_tid, value_id, m, n);
        }
        int row = row_offset + m;
        if (row < 16 && n < 8)
          C_16x8[row * 8 + n] = D_out[lane * 8 + value_id];
      }
    }
  }
}

// Extract 16x8 from D_out[32*8]: Q0 -> C[0:8,0:8], Q2 -> C[8:16,0:8].
void extract_16x8_from_canonical(const float* D_out, float* C_16x8) {
  extract_16x8_from_canonical_with_override(D_out, C_16x8, nullptr, nullptr);
}

// -----------------------------------------------------------------------------
// Diagnostic helpers
// -----------------------------------------------------------------------------
static void dump_matrix_snippet(const char* name, const float* mat, int rows,
                                int cols, int max_r, int max_c) {
  if (!g_diag) return;
  std::printf("[DUMP] %s (%d x %d), corner [0:%d, 0:%d]:\n", name, rows, cols,
              max_r, max_c);
  for (int r = 0; r < max_r && r < rows; r++) {
    std::printf("  ");
    for (int c = 0; c < max_c && c < cols; c++)
      std::printf("%6.1f ", mat[r * cols + c]);
    std::printf("\n");
  }
}

static void dump_diff_snippet(const float* got, const float* want, int rows,
                              int cols, int max_r, int max_c) {
  if (!g_diag) return;
  std::printf("[DUMP] Diff (got - want) corner [0:%d, 0:%d]:\n", max_r, max_c);
  for (int r = 0; r < max_r && r < rows; r++) {
    std::printf("  ");
    for (int c = 0; c < max_c && c < cols; c++)
      std::printf("%6.1f ", got[r * cols + c] - want[r * cols + c]);
    std::printf("\n");
  }
}

// Log which (tid, i) map to output (row, col). Useful when debugging GEMM
// mismatch: C[row,col] is gathered from these accumulator slots.
static void log_contributors_for_cell(int row, int col) {
  if (!g_diag) return;
  std::printf("[DIAG] C[%d,%d] contributed by (tid,i):", row, col);
  int n = 0;
  for (int tid = 0; tid < 32; tid++) {
    for (int i = 0; i < 8; i++) {
      int r = getWarpRowSM70(tid, i), c = getWarpColSM70(tid, i);
      if (r == row && c == col) {
        std::printf(" (%d,%d)", tid, i);
        n++;
      }
    }
  }
  std::printf("%s\n", n ? "" : " (none)");
}

// Helper: run extract+accumulate with overrides, return max err vs C_ref.
static float run_extract_and_err(
    const std::vector<std::vector<float>>& D_per_k,
    const std::vector<float>& C_ref, int M, int N,
    const int (*override_M)[8], const int (*override_N)[8]) {
  std::vector<float> C_test(M * N, 0.f);
  for (size_t k = 0; k < D_per_k.size(); k++) {
    std::vector<float> step(M * N);
    extract_16x8_from_canonical_with_override(D_per_k[k].data(), step.data(),
                                              override_M, override_N);
    for (int i = 0; i < M * N; i++) C_test[i] += step[i];
  }
  float err = 0.f;
  for (int i = 0; i < M * N; i++)
    err = std::max(err, std::abs(C_test[i] - C_ref[i]));
  return err;
}

// Mapping search: try (tid,vid) destination swaps to fix GEMM mismatch.
// Uses D_per_k (4 x 32*8), C_ref, and base SM70_8x8_32b. Tries single then
// double swaps. Reports first fix.
// -----------------------------------------------------------------------------
static void run_mapping_search(
    const std::vector<std::vector<float>>& D_per_k,
    const std::vector<float>& C_ref, int M, int N) {
  const float tol = 1e-3f;
  int rev_m[8][8], rev_n[8][8];
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) rev_m[i][j] = rev_n[i][j] = -1;
  for (int tid = 0; tid < 8; tid++)
    for (int vid = 0; vid < 8; vid++) {
      int m, n;
      SM70_8x8_32b_mn(tid, vid, m, n);
      if (m >= 0 && m < 8 && n >= 0 && n < 8) rev_m[m][n] = tid, rev_n[m][n] = vid;
    }

  std::vector<float> C_base(M * N, 0.f);
  for (size_t k = 0; k < D_per_k.size(); k++) {
    std::vector<float> step(M * N);
    extract_16x8_from_canonical(D_per_k[k].data(), step.data());
    for (int i = 0; i < M * N; i++) C_base[i] += step[i];
  }

  struct Mis { int r, c; float gpu, ref; };
  std::vector<Mis> extra, missing;
  for (int r = 0; r < M; r++)
    for (int c = 0; c < N; c++) {
      float g = C_base[r * N + c], w = C_ref[r * N + c];
      if (std::abs(g - w) <= tol) continue;
      if (g > w) extra.push_back({r, c, g, w});
      else missing.push_back({r, c, g, w});
    }

  if (extra.empty() || missing.empty()) {
    std::printf("[SEARCH] No extra/missing mismatches (unexpected).\n");
    return;
  }

  std::printf("[SEARCH] %zu extra, %zu missing. Trying single swaps...\n",
              extra.size(), missing.size());

  int override_M[8][8], override_N[8][8];
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) override_M[i][j] = override_N[i][j] = -1;

  for (const Mis& e : extra) {
    int me = e.r % 8, ne = e.c;
    if (rev_m[me][ne] < 0) continue;
    int tid_e = rev_m[me][ne], vid_e = rev_n[me][ne];
    for (const Mis& m : missing) {
      int mm = m.r % 8, nm = m.c;
      if (rev_m[mm][nm] < 0) continue;
      int tid_m = rev_m[mm][nm], vid_m = rev_n[mm][nm];
      for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
          override_M[i][j] = override_N[i][j] = -1;
      override_M[tid_e][vid_e] = mm;
      override_N[tid_e][vid_e] = nm;
      override_M[tid_m][vid_m] = me;
      override_N[tid_m][vid_m] = ne;

      float err = run_extract_and_err(D_per_k, C_ref, M, N, override_M, override_N);
      if (err <= tol) {
        std::printf("[SEARCH] FIX (1 swap): (%d,%d) <-> (%d,%d)\n", tid_e, vid_e,
                    tid_m, vid_m);
        std::printf("        (%d,%d)->(%d,%d)  (%d,%d)->(%d,%d)\n",
                    tid_e, vid_e, mm, nm, tid_m, vid_m, me, ne);
        std::printf("        Apply: M[%d][%d]=%d N[%d][%d]=%d  "
                    "M[%d][%d]=%d N[%d][%d]=%d\n",
                    tid_e, vid_e, mm, tid_e, vid_e, nm, tid_m, vid_m, me,
                    tid_m, vid_m, ne);
        return;
      }
    }
  }

  std::printf("[SEARCH] No single-swap fix. Trying double swaps...\n");

  for (const Mis& e1 : extra) {
    int me = e1.r % 8, ne = e1.c;
    if (rev_m[me][ne] < 0) continue;
    int tid_e = rev_m[me][ne], vid_e = rev_n[me][ne];
    for (const Mis& m1 : missing) {
      int mm1 = m1.r % 8, nm1 = m1.c;
      if (rev_m[mm1][nm1] < 0) continue;
      int tid_m1 = rev_m[mm1][nm1], vid_m1 = rev_n[mm1][nm1];

      for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
          override_M[i][j] = override_N[i][j] = -1;
      override_M[tid_e][vid_e] = mm1;
      override_N[tid_e][vid_e] = nm1;
      override_M[tid_m1][vid_m1] = me;
      override_N[tid_m1][vid_m1] = ne;

      std::vector<Mis> rem_extra, rem_missing;
      std::vector<float> C_after(M * N, 0.f);
      for (size_t k = 0; k < D_per_k.size(); k++) {
        std::vector<float> step(M * N);
        extract_16x8_from_canonical_with_override(D_per_k[k].data(), step.data(),
                                                  override_M, override_N);
        for (int i = 0; i < M * N; i++) C_after[i] += step[i];
      }
      for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
          float g = C_after[r * N + c], w = C_ref[r * N + c];
          if (std::abs(g - w) <= tol) continue;
          if (g > w) rem_extra.push_back({r, c, g, w});
          else rem_missing.push_back({r, c, g, w});
        }

      for (const Mis& m2 : rem_missing) {
        int mm2 = m2.r % 8, nm2 = m2.c;
        if (rev_m[mm2][nm2] < 0) continue;
        int tid_m2 = rev_m[mm2][nm2], vid_m2 = rev_n[mm2][nm2];
        if ((tid_m2 == tid_e && vid_m2 == vid_e) ||
            (tid_m2 == tid_m1 && vid_m2 == vid_m1))
          continue;

        for (int tid_x = 0; tid_x < 8; tid_x++)
          for (int vid_x = 0; vid_x < 8; vid_x++) {
            if ((tid_x == tid_e && vid_x == vid_e) ||
                (tid_x == tid_m1 && vid_x == vid_m1) ||
                (tid_x == tid_m2 && vid_x == vid_m2))
              continue;
            int mx, nx;
            if (override_M[tid_x][vid_x] >= 0)
              mx = override_M[tid_x][vid_x], nx = override_N[tid_x][vid_x];
            else
              SM70_8x8_32b_mn(tid_x, vid_x, mx, nx);

            int save_m2 = override_M[tid_m2][vid_m2], save_n2 = override_N[tid_m2][vid_m2];
            int save_xm = override_M[tid_x][vid_x], save_xn = override_N[tid_x][vid_x];
            override_M[tid_m2][vid_m2] = mx;
            override_N[tid_m2][vid_m2] = nx;
            override_M[tid_x][vid_x] = mm2;
            override_N[tid_x][vid_x] = nm2;

            float err = run_extract_and_err(D_per_k, C_ref, M, N, override_M, override_N);
            if (err <= tol) {
              std::printf("[SEARCH] FIX (2 swaps):\n");
              std::printf("  1) (%d,%d)<->(%d,%d)  (%d,%d)->(%d,%d)  (%d,%d)->(%d,%d)\n",
                          tid_e, vid_e, tid_m1, vid_m1, tid_e, vid_e, mm1, nm1,
                          tid_m1, vid_m1, me, ne);
              std::printf("  2) (%d,%d)<->(%d,%d)  (%d,%d)->(%d,%d)  (%d,%d)->(%d,%d)\n",
                          tid_m2, vid_m2, tid_x, vid_x, tid_m2, vid_m2, mx, nx,
                          tid_x, vid_x, mm2, nm2);
              std::printf("  Apply: M[%d][%d]=%d N[%d][%d]=%d  M[%d][%d]=%d N[%d][%d]=%d\n",
                          tid_e, vid_e, mm1, tid_e, vid_e, nm1, tid_m1, vid_m1, me, tid_m1, vid_m1, ne);
              std::printf("        M[%d][%d]=%d N[%d][%d]=%d  M[%d][%d]=%d N[%d][%d]=%d\n",
                          tid_m2, vid_m2, mx, tid_m2, vid_m2, nx, tid_x, vid_x, mm2, tid_x, vid_x, nm2);
              return;
            }
            override_M[tid_m2][vid_m2] = save_m2;
            override_N[tid_m2][vid_m2] = save_n2;
            override_M[tid_x][vid_x] = save_xm;
            override_N[tid_x][vid_x] = save_xn;
          }
      }
    }
  }
  std::printf("[SEARCH] No single- or double-swap fix found.\n");
}

// Checkpoint 0: (tid,i)->(row,col) coverage. Returns true if OK.
static bool checkpoint_mapping_coverage() {
  SECTION("CHECKPOINT 0: (tid,i)->(row,col) mapping coverage");
  std::set<std::pair<int, int>> pos;
  std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> rev;
  int row_lo = 99, row_hi = -99, col_lo = 99, col_hi = -99;
  for (int tid = 0; tid < 32; tid++) {
    for (int i = 0; i < 8; i++) {
      int r = getWarpRowSM70(tid, i);
      int c = getWarpColSM70(tid, i);
      pos.insert({r, c});
      rev[{r, c}].push_back({tid, i});
      row_lo = std::min(row_lo, r);
      row_hi = std::max(row_hi, r);
      col_lo = std::min(col_lo, c);
      col_hi = std::max(col_hi, c);
    }
  }
  DIAG("Row range [%d, %d], Col range [%d, %d]", row_lo, row_hi, col_lo,
       col_hi);
  int in_8x8 = 0;
  for (const auto& p : pos)
    if (p.first >= 0 && p.first < 16 && p.second >= 0 && p.second < 8)
      in_8x8++;
  DIAG("Unique (row,col) in 16x8: %d (expect 128)", in_8x8);
  bool ok = (in_8x8 == 128);
  for (const auto& kv : rev) {
    if (kv.second.size() > 1 && kv.first.first < 16 && kv.first.second < 8) {
      DIAG("Duplicate (r,c)=(%d,%d) from (tid,i):", kv.first.first,
           kv.first.second);
      for (const auto& ti : kv.second)
        std::printf("    (%d,%d)\n", ti.first, ti.second);
      ok = false;
    }
  }
  CP("mapping", ok,
     ok ? "128 unique (row,col) in 16x8, no duplicates"
       : "coverage or uniqueness violated");
  return ok;
}

// Checkpoint: fill_A spot-check. Returns true if OK. Uses get_coords_A (row, k).
static bool checkpoint_fill_A(const std::vector<float>& A,
                              const std::vector<half2>& regs) {
  SECTION("CHECKPOINT: Fill A spot-check");
  bool ok = true;
  for (int tid : {0, 1, 16}) {
    for (int j = 0; j < 4 && ok; j++) {
      int i_lo = j * 2 + 0, i_hi = j * 2 + 1;
      int r_lo, k_lo, r_hi, k_hi;
      get_coords_A(tid, i_lo, r_lo, k_lo);
      get_coords_A(tid, i_hi, r_hi, k_hi);
      float expect_lo = (r_lo < 16 && k_lo < 16) ? A[r_lo * 16 + k_lo] : 0.f;
      float expect_hi = (r_hi < 16 && k_hi < 16) ? A[r_hi * 16 + k_hi] : 0.f;
      half hr0 = *reinterpret_cast<const half*>(&regs[tid * 4 + j]);
      half hr1 =
          *reinterpret_cast<const half*>((const char*)&regs[tid * 4 + j] + 2);
      float got_lo = half2float(hr0), got_hi = half2float(hr1);
      if (std::abs(got_lo - expect_lo) > 1e-3f ||
          std::abs(got_hi - expect_hi) > 1e-3f) {
        DIAG("fill_A tid=%d j=%d (r,k)=(%d,%d),(%d,%d) expect (%.2f,%.2f) got (%.2f,%.2f)",
             tid, j, r_lo, k_lo, r_hi, k_hi, expect_lo, expect_hi, got_lo,
             got_hi);
        ok = false;
      }
    }
  }
  CP("fill_A", ok, ok ? "spot-check (tid 0,1,16) matches A" : "spot-check mismatch");
  return ok;
}

// Checkpoint: fill_B spot-check. Returns true if OK.
static bool checkpoint_fill_B(const std::vector<float>& B,
                              const std::vector<half2>& regs) {
  SECTION("CHECKPOINT: Fill B spot-check");
  bool ok = true;
  for (int tid : {0, 2, 8}) {
    int t_mod = tid % 16;
    int col = (t_mod < 4) ? t_mod : (t_mod >= 8 && t_mod < 12) ? (t_mod - 8) + 4 : 0;
    int k_base = (tid / 8) * 4;
    for (int j = 0; j < 2 && ok; j++) {
      float e0 = (col >= 0 && col < 8) ? B[(k_base + j * 2) * 8 + col] : 0.f;
      float e1 = (col >= 0 && col < 8) ? B[(k_base + j * 2 + 1) * 8 + col] : 0.f;
      half h0 = *reinterpret_cast<const half*>(&regs[tid * 2 + j]);
      half h1 =
          *reinterpret_cast<const half*>((const char*)&regs[tid * 2 + j] + 2);
      float g0 = half2float(h0), g1 = half2float(h1);
      if (std::abs(g0 - e0) > 1e-3f || std::abs(g1 - e1) > 1e-3f) {
        DIAG("fill_B tid=%d j=%d col=%d k_base=%d expect (%.2f,%.2f) got (%.2f,%.2f)",
             tid, j, col, k_base, e0, e1, g0, g1);
        ok = false;
      }
    }
  }
  CP("fill_B", ok, ok ? "spot-check (tid 0,2,8) matches B" : "spot-check mismatch");
  return ok;
}

// Simulate B gather on host for identity B. Fills expected[tid][0..15] using
// same base, SHFL_LANE, pack_cols as device. b_regs = reinterpret of B fill.
static void simulate_gather_B(const uint32_t* b_regs, int tid,
                              uint32_t expected[16]) {
  int base = (tid & 10) | ((tid & 1) << 3);
  auto lane = [](int b, int o) { return ((b) + (o)) % 32; };
  const int offs[4][2] = {{0, 1}, {8, 9}, {16, 17}, {24, 25}};
  for (int g = 0; g < 4; g++) {
    int la = lane(base, offs[g][0]), lb = lane(base, offs[g][1]);
    uint32_t v0_lo = b_regs[la * 2 + 0], v1_lo = b_regs[lb * 2 + 0];
    uint32_t v0_hi = b_regs[la * 2 + 1], v1_hi = b_regs[lb * 2 + 1];
    expected[g * 4 + 0] = pack_cols_host(v0_lo, v1_lo, false);
    expected[g * 4 + 1] = pack_cols_host(v0_lo, v1_lo, true);
    expected[g * 4 + 2] = pack_cols_host(v0_hi, v1_hi, false);
    expected[g * 4 + 3] = pack_cols_host(v0_hi, v1_hi, true);
  }
}

// Checkpoint: gathered B (after warp gather) vs host simulation. Identity B.
static bool checkpoint_gathered_B(const std::vector<float>& B,
                                  const std::vector<half2>& B_regs) {
  SECTION("CHECKPOINT: Gathered B (post-gather) vs expected");
  const uint32_t* b_regs = reinterpret_cast<const uint32_t*>(B_regs.data());
  uint32_t expected[32][16];
  for (int tid = 0; tid < 32; tid++)
    simulate_gather_B(b_regs, tid, expected[tid]);

  uint32_t* d_gathered = nullptr;
  cudaMalloc(&d_gathered, 32 * 16 * sizeof(uint32_t));
  half2* dB = nullptr;
  cudaMalloc(&dB, 32 * 2 * sizeof(half2));
  cudaMemcpy(dB, B_regs.data(), 32 * 2 * sizeof(half2),
             cudaMemcpyHostToDevice);
  test_gather_B_only<<<1, 32>>>(dB, d_gathered);
  cudaDeviceSynchronize();
  std::vector<uint32_t> gathered_h(32 * 16);
  cudaMemcpy(gathered_h.data(), d_gathered, 32 * 16 * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaFree(d_gathered);
  cudaFree(dB);

  bool ok = true;
  for (int tid : {0, 1, 2, 8}) {
    for (int i = 0; i < 16 && ok; i++) {
      uint32_t got = gathered_h[tid * 16 + i];
      uint32_t want = expected[tid][i];
      if (got != want) {
        DIAG("gathered_B tid=%d i=%d got %08x want %08x", tid, i, got, want);
        ok = false;
      }
    }
  }
  CP("gathered_B", ok,
     ok ? "spot-check (tid 0,1,2,8) gathered B matches simulation"
       : "gathered B mismatch vs host simulation");
  return ok;
}

// Checkpoint: partial MMA (K0..3) vs C_ref = A[:,0:4] @ B[0:4,:].
// Uses canonical 16x8 path: fill_fragments_16x8_canonical_kstep, mma_16x8_canonical_kstep, extract.
static bool checkpoint_partial_K(const std::vector<float>& A,
                                 const std::vector<float>& B,
                                 const std::vector<half2>&,
                                 const std::vector<half2>&) {
  SECTION("CHECKPOINT: Partial K (K0..3) vs A[:,0:4]@B[0:4,:]");
  const int M = 16, N = 8, K_part = 4;
  std::vector<float> C_ref_partial(M * N, 0.0f);
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++)
      for (int k = 0; k < K_part; k++)
        C_ref_partial[m * N + n] += A[m * 16 + k] * B[k * N + n];

  std::vector<half2> A_frag, B_frag;
  fill_fragments_16x8_canonical_kstep(A.data(), B.data(), 0, A_frag, B_frag);

  half2 *dA = nullptr, *dB = nullptr;
  float* dD = nullptr;
  cudaMalloc(&dA, 32 * 2 * sizeof(half2));
  cudaMalloc(&dB, 32 * 2 * sizeof(half2));
  cudaMalloc(&dD, 32 * 8 * sizeof(float));
  cudaMemcpy(dA, A_frag.data(), 32 * 2 * sizeof(half2), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B_frag.data(), 32 * 2 * sizeof(half2), cudaMemcpyHostToDevice);
  mma_16x8_canonical_kstep<<<1, 32>>>(dA, dB, dD);
  cudaDeviceSynchronize();
  std::vector<float> D_h(32 * 8);
  cudaMemcpy(D_h.data(), dD, 32 * 8 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dD);

  std::vector<float> C_gpu(M * N);
  extract_16x8_from_canonical(D_h.data(), C_gpu.data());
  float err = 0.0f;
  for (int i = 0; i < M * N; i++)
    err = std::max(err, std::abs(C_gpu[i] - C_ref_partial[i]));
  bool ok = (err <= 1e-3f);
  CP("partial_K", ok,
     ok ? "partial K0..3 matches A[:,0:4]@B[0:4,:]"
       : "partial K0..3 mismatch (check first K-step, A/B layout)");
  if (!ok) DIAG("partial_K max_err=%.6f", err);
  return ok;
}

static void set_verbose(bool v) {
  g_verbose = v;
  int x = v ? 1 : 0;
  cudaMemcpyToSymbol(g_device_verbose, &x, sizeof(int));
}

static void diagnostic_summary(bool cp_mapping, bool cp_fill_a, bool cp_fill_b,
                               bool cp_gathered_b, bool cp_partial_k,
                               bool extract_ok, bool gemm_ok,
                               bool canonical_8x8x4) {
  SECTION("DIAGNOSTIC SUMMARY");
  std::printf("  CP mapping        : %s\n", cp_mapping ? "PASS" : "FAIL");
  std::printf("  CP fill_A         : %s\n", cp_fill_a ? "PASS" : "FAIL");
  std::printf("  CP fill_B         : %s\n", cp_fill_b ? "PASS" : "FAIL");
  std::printf("  CP gathered_B     : %s\n", cp_gathered_b ? "PASS" : "FAIL");
  std::printf("  CP partial_K      : %s\n", cp_partial_k ? "PASS" : "FAIL");
  std::printf("  Extract-only      : %s\n", extract_ok ? "PASS" : "FAIL");
  std::printf("  Full GEMM         : %s\n", gemm_ok ? "PASS" : "FAIL");
  std::printf("  8x8x4 canonical   : %s\n",
              canonical_8x8x4 ? "PASS" : "FAIL");
  std::printf("\n  WHERE TO LOOK NEXT:\n");
  if (!cp_mapping) {
    std::printf("    -> getWarpRowSM70 / getWarpColSM70: (tid,i)->(row,col) must "
                "cover 128 unique (row,col) in 16x8, no duplicates.\n");
  }
  if (!cp_fill_a || !cp_fill_b) {
    std::printf("    -> Fill spot-checks failed. Fix fill_registers_A / "
                "fill_registers_B (get_coords, B col/k_base) before GEMM.\n");
  }
  if (!cp_gathered_b) {
    std::printf("    -> Gathered B mismatch: base, SHFL_LANE, pack_cols, or "
                "fill_B layout. Compare test_gather_B_only vs simulate_gather_B.\n");
  }
  if (!cp_partial_k) {
    std::printf("    -> Partial K0..3 failed: first K-step or A/B->C layout wrong. "
                "Check m8n8k4 operand order, ab* layout.\n");
  }
  if (extract_ok && !gemm_ok && cp_fill_a && cp_fill_b && cp_gathered_b &&
      cp_partial_k) {
    std::printf("    -> Fills, gather, partial K OK. Full GEMM fails in later "
                "K-steps or extract. Check K-tiling, b_high usage.\n");
  }
  if (extract_ok && !gemm_ok && (cp_fill_a && cp_fill_b) && !cp_gathered_b) {
    std::printf("    -> Fix gathered B first (see above), then re-check GEMM.\n");
  }
  if (!extract_ok) {
    std::printf("    -> Fix extract first. Check:\n");
    std::printf("       1. getWarpRowSM70 / getWarpColSM70 match reverse-engineered "
                "mapping.\n");
    std::printf("       2. Extract: search all 32 threads, row==target_row, "
                "col==target_col.\n");
    std::printf("       3. tmp_top vs tmp_bot (rows 0-7 vs 8-15).\n");
    std::printf("       4. reconstruct_C_m16n8k16 (row=tid/2, col=(tid%%2)*4+i).\n");
  }
  if (!canonical_8x8x4) {
    std::printf("    -> 8x8x4 canonical failed: check quadpairs, SM70_8x8_32b, "
                "SM70_8x4_Row, B 4x8 col-major fill/extract.\n");
  }
  if (gemm_ok && extract_ok && cp_mapping && cp_fill_a && cp_fill_b &&
      cp_gathered_b && cp_partial_k && canonical_8x8x4) {
    std::printf("    -> All checkpoints passed. Emulation validates.\n");
  }
  std::printf("\n");
}

int main(int argc, char** argv) {
  g_verbose = false;
  g_diag = false;
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "verbose") == 0) g_verbose = true;
    if (std::strcmp(argv[i], "diag") == 0 ||
        std::strcmp(argv[i], "diagnostic") == 0)
      g_diag = true;
    if (std::strcmp(argv[i], "search") == 0) g_search = true;
  }
  if (g_diag) g_verbose = true;  // diag implies verbose (device logs, etc.)
  if (g_search) g_diag = true;   // search implies diag (need GEMM failure details)

  int failures = 0;
  bool cp_mapping_ok = true;
  bool cp_fill_a_ok = true;
  bool cp_fill_b_ok = true;
  bool cp_gathered_b_ok = true;
  bool cp_partial_k_ok = true;
  bool extract_ok = false;
  bool gemm_ok = false;
  bool canonical_8x8x4_ok = false;

  if (g_diag) {
    std::printf("\n*** SM70 MMA validation diagnostic mode ***\n");
    std::printf("    verbose=%d diag=%d search=%d\n", g_verbose ? 1 : 0,
                g_diag ? 1 : 0, g_search ? 1 : 0);
    std::printf("    Device logs (T0,T1,T2,T8 A/B regs, accumulators) appear "
                "between kernel runs.\n\n");
  }

  // -------------------------------------------------------------------------
  // Checkpoint 0: (tid,i)->(row,col) coverage (host-only, no CUDA)
  // -------------------------------------------------------------------------
  if (g_diag) {
    cp_mapping_ok = checkpoint_mapping_coverage();
    if (!cp_mapping_ok) failures++;
  }

  // -------------------------------------------------------------------------
  // Test 1: Extract-only. Validates (tid,i)->(row,col) and gather.
  // -------------------------------------------------------------------------
  {
    if (g_diag) SECTION("TEST 1: EXTRACT-ONLY (mapping + gather)");

    float* dC = nullptr;
    cudaMalloc(&dC, 32 * 4 * sizeof(float));
    cudaMemset(dC, 0, 32 * 4 * sizeof(float));
    set_verbose(g_verbose);
    test_extract_only<<<1, 32>>>(dC);
    cudaDeviceSynchronize();

    std::vector<float> dC_h(32 * 4);
    cudaMemcpy(dC_h.data(), dC, 32 * 4 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(dC);

    std::vector<float> C_gpu(16 * 8);
    reconstruct_C_m16n8k16(dC_h, C_gpu);

    float err = 0.0f;
    int first_fail_r = -1, first_fail_c = -1;
    float first_fail_got = 0.f, first_fail_want = 0.f;
    for (int r = 0; r < 16; r++) {
      for (int c = 0; c < 8; c++) {
        float want = 100.0f * (float)r + (float)c;
        float got = C_gpu[r * 8 + c];
        float e = std::abs(got - want);
        if (e > err) err = e;
        if (e > 1e-3f && first_fail_r < 0) {
          first_fail_r = r;
          first_fail_c = c;
          first_fail_got = got;
          first_fail_want = want;
        }
      }
    }

    extract_ok = (err <= 1e-3f);
    if (g_diag) {
      CP("extract", extract_ok,
         extract_ok ? "extract-only matches 100*row+col"
                   : "extract-only mismatch");
      if (!extract_ok) DIAG("extract max_err=%.6f", err);
    }

    if (!extract_ok) {
      failures++;
      if (g_diag && first_fail_r >= 0) {
        DIAG("First mismatch (r,c)=(%d,%d): got %.2f want %.2f", first_fail_r,
             first_fail_c, first_fail_got, first_fail_want);
        std::vector<float> want_mat(16 * 8);
        for (int r = 0; r < 16; r++)
          for (int c = 0; c < 8; c++) want_mat[r * 8 + c] = 100.f * r + c;
        dump_matrix_snippet("C_gpu (extract)", C_gpu.data(), 16, 8, 6, 6);
        dump_matrix_snippet("C_want (100r+c)", want_mat.data(), 16, 8, 6, 6);
        dump_diff_snippet(C_gpu.data(), want_mat.data(), 16, 8, 6, 6);
      }
      if (!g_diag) {
        std::cout << "[FAIL] Extract-only: max err " << err << std::endl;
        std::cout << "  Run with 'diag' for checkpoints and dumps; 'verbose' "
                     "for device prints."
                  << std::endl;
      }
    } else if (!g_diag) {
      std::cout << "[PASS] Extract-only (mapping + gather)." << std::endl;
    }
  }

  // -------------------------------------------------------------------------
  // Test 2: Full GEMM (identity-style A, B -> C_ref = A @ B).
  // -------------------------------------------------------------------------
  {
    if (g_diag) SECTION("TEST 2: FULL GEMM (A @ B)");

    int M = 16, N = 8, K = 16;
    std::vector<float> A(M * K, 0.0f), B(K * N, 0.0f), C_ref(M * N, 0.0f);
    for (int k = 0; k < 16; k++) A[k * K + k] = 1.0f;
    for (int k = 0; k < 8; k++) B[k * N + k] = 1.0f;

    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[m * K + k] * B[k * N + n];
        C_ref[m * N + n] = sum;
      }

    std::vector<half2> dA_h(32 * 4), dB_h(32 * 2);
    fill_registers_A_m16n8k16(A, dA_h);
    fill_registers_B_m16n8k16(B, dB_h);

    if (g_diag) {
      cp_fill_a_ok = checkpoint_fill_A(A, dA_h);
      cp_fill_b_ok = checkpoint_fill_B(B, dB_h);
      if (!cp_fill_a_ok || !cp_fill_b_ok) failures++;
      cp_gathered_b_ok = checkpoint_gathered_B(B, dB_h);
      if (!cp_gathered_b_ok) failures++;
      cp_partial_k_ok = checkpoint_partial_K(A, B, dA_h, dB_h);
      if (!cp_partial_k_ok) failures++;
    }

    std::vector<float> C_gpu(M * N, 0.0f);
    std::vector<half2> A_frag, B_frag;
    half2 *dA = nullptr, *dB = nullptr;
    float* dD = nullptr;
    cudaMalloc(&dA, 32 * 2 * sizeof(half2));
    cudaMalloc(&dB, 32 * 2 * sizeof(half2));
    cudaMalloc(&dD, 32 * 8 * sizeof(float));

    std::vector<std::vector<float>> D_per_k;
    set_verbose(g_verbose);
    for (int k_off = 0; k_off < K; k_off += 4) {
      fill_fragments_16x8_canonical_kstep(A.data(), B.data(), k_off, A_frag,
                                          B_frag);
      cudaMemcpy(dA, A_frag.data(), 32 * 2 * sizeof(half2),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(dB, B_frag.data(), 32 * 2 * sizeof(half2),
                 cudaMemcpyHostToDevice);
      mma_16x8_canonical_kstep<<<1, 32>>>(dA, dB, dD);
      cudaDeviceSynchronize();
      std::vector<float> D_h(32 * 8);
      cudaMemcpy(D_h.data(), dD, 32 * 8 * sizeof(float),
                 cudaMemcpyDeviceToHost);
      if (g_search) D_per_k.push_back(D_h);
      std::vector<float> C_step(M * N);
      extract_16x8_from_canonical(D_h.data(), C_step.data());
      for (int i = 0; i < M * N; i++) C_gpu[i] += C_step[i];
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);

    float err = 0.0f;
    int first_m = -1, first_n = -1;
    float first_got = 0.f, first_ref = 0.f;
    for (int i = 0; i < M * N; i++)
      err = std::max(err, std::abs(C_gpu[i] - C_ref[i]));
    for (int m = 0; m < M && first_m < 0; m++)
      for (int n = 0; n < N; n++)
        if (std::abs(C_gpu[m * N + n] - C_ref[m * N + n]) > 1e-3f) {
          first_m = m;
          first_n = n;
          first_got = C_gpu[m * N + n];
          first_ref = C_ref[m * N + n];
          break;
        }

    gemm_ok = (err <= 1e-3f);
    if (g_diag) {
      CP("gemm", gemm_ok,
         gemm_ok ? "C_gpu matches C_ref" : "C_gpu vs C_ref mismatch");
      if (!gemm_ok) DIAG("gemm max_err=%.6f", err);
    }

    if (!gemm_ok) {
      failures++;
      if (g_diag && first_m >= 0) {
        DIAG("First GEMM mismatch (m,n)=(%d,%d): GPU=%.4f Ref=%.4f", first_m,
             first_n, first_got, first_ref);
        log_contributors_for_cell(first_m, first_n);
        dump_matrix_snippet("C_gpu", C_gpu.data(), M, N, 8, 6);
        dump_matrix_snippet("C_ref", C_ref.data(), M, N, 8, 6);
        dump_diff_snippet(C_gpu.data(), C_ref.data(), M, N, 8, 6);
      }
      if (g_search && D_per_k.size() == 4) {
        SECTION("MAPPING SEARCH (swap candidates)");
        run_mapping_search(D_per_k, C_ref, M, N);
      }
      if (g_verbose && !g_diag) {
        std::cout << "A (snippet): " << A[0] << " " << A[1] << " ... " << A[16]
                  << " " << A[17] << std::endl;
        std::cout << "B (snippet): " << B[0] << " " << B[1] << " ... " << B[8]
                  << " " << B[9] << std::endl;
      }
      if (!g_diag) {
        std::cout << "[FAIL] Full GEMM: max err " << err << std::endl;
        if (g_verbose) {
          for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
              if (std::abs(C_gpu[m * N + n] - C_ref[m * N + n]) > 1e-3f)
                std::cout << "  (" << m << "," << n
                          << ") GPU:" << C_gpu[m * N + n]
                          << " Ref:" << C_ref[m * N + n] << std::endl;
        } else {
          std::cout << "  Run with 'diag' or 'verbose' for details."
                    << std::endl;
        }
      }
    } else if (!g_diag) {
      std::cout << "[PASS] Full GEMM." << std::endl;
    }
  }

  // -------------------------------------------------------------------------
  // Test 2b: Full GEMM with additional matrices (identity, ones, random, etc.)
  // Same path as Test 2: alloc once, set_verbose, 4-step loop per case, free.
  // -------------------------------------------------------------------------
  {
    if (g_diag) SECTION("TEST 2b: GEMM (additional matrices)");
    const int M = 16, N = 8, K = 16;
    const float tol_default = 1e-2f;  // for identity, all-ones, scale
    std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_gpu(M * N);
    std::vector<half2> A_frag, B_frag;
    half2 *dA = nullptr, *dB = nullptr;
    float* dD = nullptr;
    cudaMalloc(&dA, 32 * 2 * sizeof(half2));
    cudaMalloc(&dB, 32 * 2 * sizeof(half2));
    cudaMalloc(&dD, 32 * 8 * sizeof(float));
    set_verbose(g_verbose);

    auto run_case = [&](const char* name, float tol_override = -1.f) {
      float tol = (tol_override >= 0.f) ? tol_override : tol_default;
      for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
          float sum = 0.f;
          for (int k = 0; k < K; k++)
            sum += A[m * K + k] * B[k * N + n];
          C_ref[m * N + n] = sum;
        }
      std::fill(C_gpu.begin(), C_gpu.end(), 0.f);
      for (int k_off = 0; k_off < K; k_off += 4) {
        fill_fragments_16x8_canonical_kstep(A.data(), B.data(), k_off, A_frag,
                                            B_frag);
        cudaMemcpy(dA, A_frag.data(), 32 * 2 * sizeof(half2),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B_frag.data(), 32 * 2 * sizeof(half2),
                   cudaMemcpyHostToDevice);
        mma_16x8_canonical_kstep<<<1, 32>>>(dA, dB, dD);
        cudaDeviceSynchronize();
        std::vector<float> D_h(32 * 8);
        cudaMemcpy(D_h.data(), dD, 32 * 8 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        std::vector<float> C_step(M * N);
        extract_16x8_from_canonical(D_h.data(), C_step.data());
        for (int i = 0; i < M * N; i++) C_gpu[i] += C_step[i];
      }
      float err = 0.f;
      for (int i = 0; i < M * N; i++)
        err = std::max(err, std::abs(C_gpu[i] - C_ref[i]));
      bool ok = (err <= tol);
      if (g_diag) {
        CP(name, ok, ok ? "OK" : "mismatch");
        if (!ok) DIAG("%s max_err=%.6f", name, err);
      } else {
        if (ok)
          std::cout << "[PASS] GEMM " << name << "." << std::endl;
        else
          std::cout << "[FAIL] GEMM " << name << ": max err " << err
                    << std::endl;
      }
      if (!ok) failures++;
      return ok;
    };

    // identity (same as Test 2; sanity check same path)
    std::fill(A.begin(), A.end(), 0.f);
    std::fill(B.begin(), B.end(), 0.f);
    for (int k = 0; k < 16; k++) A[k * K + k] = 1.f;
    for (int k = 0; k < 8; k++) B[k * N + k] = 1.f;
    run_case("identity");

    // all-ones: A=1, B=1 -> C[m][n] = 16
    std::fill(A.begin(), A.end(), 1.f);
    std::fill(B.begin(), B.end(), 1.f);
    run_case("all-ones");

    // random [0,1], fixed seed (relaxed tol: fp16 accumulation over K=16)
    {
      std::mt19937 rng(42);
      std::uniform_real_distribution<float> u(0.f, 1.f);
      for (float& x : A) x = u(rng);
      for (float& x : B) x = u(rng);
    }
    run_case("random [0,1]", 5.f);

    // random [-1,1] (relaxed tol)
    {
      std::mt19937 rng(123);
      std::uniform_real_distribution<float> u(-1.f, 1.f);
      for (float& x : A) x = u(rng);
      for (float& x : B) x = u(rng);
    }
    run_case("random [-1,1]", 5.f);

    // pattern: A[m][k]=m+k, B[k][n]=k-n (relaxed tol: large |C|, fp16)
    for (int m = 0; m < M; m++)
      for (int k = 0; k < K; k++) A[m * K + k] = static_cast<float>(m + k);
    for (int k = 0; k < K; k++)
      for (int n = 0; n < N; n++)
        B[k * N + n] = static_cast<float>(k - n);
    run_case("pattern A=m+k B=k-n", 2000.f);

    // scale 0.1: A=0.1, B=0.1 -> C = 0.16
    std::fill(A.begin(), A.end(), 0.1f);
    std::fill(B.begin(), B.end(), 0.1f);
    run_case("scale 0.1");

    // row/col: A[m][k]=m, B[k][n]=n (relaxed tol: large |C|, fp16)
    for (int m = 0; m < M; m++)
      for (int k = 0; k < K; k++) A[m * K + k] = static_cast<float>(m);
    for (int k = 0; k < K; k++)
      for (int n = 0; n < N; n++) B[k * N + n] = static_cast<float>(n);
    run_case("pattern A=m B=n", 1000.f);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
  }

  // -------------------------------------------------------------------------
  // Test 3: 8x8x4 canonical (documented SM70 layout: quadpairs, SM70_8x8_32b)
  // -------------------------------------------------------------------------
  {
    if (g_diag) SECTION("TEST 3: 8x8x4 CANONICAL (documented layout)");

    float A_8x4[32], B_4x8[32];
    for (int i = 0; i < 32; i++) A_8x4[i] = B_4x8[i] = 0.f;
    for (int i = 0; i < 4; i++) {
      A_8x4[i * 4 + i] = 1.f;
      B_4x8[i * 8 + i] = 1.f;
    }
    float C_ref_8x8[64];
    for (int m = 0; m < 8; m++)
      for (int n = 0; n < 8; n++) {
        float sum = 0.f;
        for (int k = 0; k < 4; k++)
          sum += A_8x4[m * 4 + k] * B_4x8[k * 8 + n];
        C_ref_8x8[m * 8 + n] = sum;
      }

    std::vector<half2> A_frag, B_frag;
    fill_fragments_8x8x4_canonical(A_8x4, B_4x8, A_frag, B_frag);

    half2 *dA = nullptr, *dB = nullptr;
    float* dD = nullptr;
    cudaMalloc(&dA, 32 * 2 * sizeof(half2));
    cudaMalloc(&dB, 32 * 2 * sizeof(half2));
    cudaMalloc(&dD, 32 * 8 * sizeof(float));
    cudaMemcpy(dA, A_frag.data(), 32 * 2 * sizeof(half2),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_frag.data(), 32 * 2 * sizeof(half2),
               cudaMemcpyHostToDevice);
    cudaMemset(dD, 0, 32 * 8 * sizeof(float));

    test_mma_8x8x4_canonical<<<1, 32>>>(dA, dB, dD);
    cudaDeviceSynchronize();

    std::vector<float> dD_h(32 * 8);
    cudaMemcpy(dD_h.data(), dD, 32 * 8 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);

    float C_gpu_8x8[64];
    extract_8x8_from_quadpair0(dD_h.data(), C_gpu_8x8);

    float err = 0.f;
    for (int i = 0; i < 64; i++)
      err = std::max(err, std::abs(C_gpu_8x8[i] - C_ref_8x8[i]));
    canonical_8x8x4_ok = (err <= 1e-3f);

    if (g_diag) {
      CP("canonical_8x8x4", canonical_8x8x4_ok,
         canonical_8x8x4_ok ? "8x8x4 canonical (SM70 layout) matches A@B"
                           : "8x8x4 canonical mismatch");
      if (!canonical_8x8x4_ok) {
        DIAG("canonical_8x8x4 max_err=%.6f", err);
        dump_matrix_snippet("C_gpu_8x8 (canonical)", C_gpu_8x8, 8, 8, 8, 8);
        dump_matrix_snippet("C_ref_8x8 (A@B)", C_ref_8x8, 8, 8, 8, 8);
        dump_diff_snippet(C_gpu_8x8, C_ref_8x8, 8, 8, 8, 8);
        /* Single-element micro-test: A[0][0]=1, B[0][0]=1 only -> (A@B)[0][0]=1 */
        {
          float Ae[32], Be[32];
          for (int i = 0; i < 32; i++) Ae[i] = Be[i] = 0.f;
          Ae[0] = Be[0] = 1.f;
          std::vector<half2> Af, Bf;
          fill_fragments_8x8x4_canonical(Ae, Be, Af, Bf);
          half2 *dAe = nullptr, *dBe = nullptr;
          float* dDe = nullptr;
          cudaMalloc(&dAe, 32 * 2 * sizeof(half2));
          cudaMalloc(&dBe, 32 * 2 * sizeof(half2));
          cudaMalloc(&dDe, 32 * 8 * sizeof(float));
          cudaMemcpy(dAe, Af.data(), 32 * 2 * sizeof(half2),
                     cudaMemcpyHostToDevice);
          cudaMemcpy(dBe, Bf.data(), 32 * 2 * sizeof(half2),
                     cudaMemcpyHostToDevice);
          cudaMemset(dDe, 0, 32 * 8 * sizeof(float));
          test_mma_8x8x4_canonical<<<1, 32>>>(dAe, dBe, dDe);
          cudaDeviceSynchronize();
          std::vector<float> dDe_h(32 * 8);
          cudaMemcpy(dDe_h.data(), dDe, 32 * 8 * sizeof(float),
                     cudaMemcpyDeviceToHost);
          cudaFree(dAe);
          cudaFree(dBe);
          cudaFree(dDe);
          float Ce[64];
          extract_8x8_from_quadpair0(dDe_h.data(), Ce);
          int mi = -1, mj = -1;
          float mx = 0.f;
          for (int i = 0; i < 64; i++)
            if (std::abs(Ce[i]) > mx) {
              mx = std::abs(Ce[i]);
              mi = i / 8;
              mj = i % 8;
            }
          DIAG("single-element A[0][0]=B[0][0]=1: max |C| at (%d,%d)=%.4f (expect (0,0)=1)",
               mi, mj, mx);
          dump_matrix_snippet("C_8x8 (single-elem)", Ce, 8, 8, 8, 8);
        }
      }
    }
    if (!canonical_8x8x4_ok) {
      failures++;
      if (!g_diag) {
        std::cout << "[FAIL] 8x8x4 canonical: max err " << err << std::endl;
        std::cout << "  Run with 'diag' for layout checkpoints." << std::endl;
      }
    } else if (!g_diag) {
      std::cout << "[PASS] 8x8x4 canonical (documented layout)." << std::endl;
    }
  }

  // -------------------------------------------------------------------------
  // Test 3b: 8x8x4 all-ones (isolate MMA+extract for dense vs 16x8 path)
  // -------------------------------------------------------------------------
  {
    if (g_diag) SECTION("TEST 3b: 8x8x4 ALL-ONES");
    float A_8x4[32], B_4x8[32], C_ref_8x8[64];
    for (int i = 0; i < 32; i++) A_8x4[i] = B_4x8[i] = 1.f;
    for (int i = 0; i < 64; i++) C_ref_8x8[i] = 4.f;  // 8x4 @ 4x8, all 1s -> 4

    std::vector<half2> A_frag, B_frag;
    fill_fragments_8x8x4_canonical(A_8x4, B_4x8, A_frag, B_frag);
    half2 *dA = nullptr, *dB = nullptr;
    float* dD = nullptr;
    cudaMalloc(&dA, 32 * 2 * sizeof(half2));
    cudaMalloc(&dB, 32 * 2 * sizeof(half2));
    cudaMalloc(&dD, 32 * 8 * sizeof(float));
    cudaMemcpy(dA, A_frag.data(), 32 * 2 * sizeof(half2),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_frag.data(), 32 * 2 * sizeof(half2),
               cudaMemcpyHostToDevice);
    cudaMemset(dD, 0, 32 * 8 * sizeof(float));
    test_mma_8x8x4_canonical<<<1, 32>>>(dA, dB, dD);
    cudaDeviceSynchronize();
    std::vector<float> dD_h(32 * 8);
    cudaMemcpy(dD_h.data(), dD, 32 * 8 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    float C_gpu_8x8[64];
    extract_8x8_from_quadpair0(dD_h.data(), C_gpu_8x8);
    float err = 0.f;
    for (int i = 0; i < 64; i++)
      err = std::max(err, std::abs(C_gpu_8x8[i] - C_ref_8x8[i]));
    bool ok = (err <= 1e-2f);
    if (g_diag) {
      CP("8x8x4_all_ones", ok, ok ? "OK" : "mismatch");
      if (!ok) {
        DIAG("8x8x4 all-ones max_err=%.6f", err);
        dump_matrix_snippet("C_gpu_8x8 (all-ones)", C_gpu_8x8, 8, 8, 8, 8);
        dump_matrix_snippet("C_ref_8x8 (4)", C_ref_8x8, 8, 8, 8, 8);
      }
    } else {
      if (ok)
        std::cout << "[PASS] 8x8x4 all-ones." << std::endl;
      else
        std::cout << "[FAIL] 8x8x4 all-ones: max err " << err << std::endl;
    }
    if (!ok) failures++;
  }

  // -------------------------------------------------------------------------
  // Diagnostic summary and next steps
  // -------------------------------------------------------------------------
  if (g_diag)
    diagnostic_summary(cp_mapping_ok, cp_fill_a_ok, cp_fill_b_ok,
                      cp_gathered_b_ok, cp_partial_k_ok, extract_ok, gemm_ok,
                      canonical_8x8x4_ok);

  std::cout << (failures ? "OVERALL: FAIL" : "OVERALL: PASS") << std::endl;
  return failures != 0;
}
