# SM70 Marlin GEMM Debugging Tracker

## Current Status: IMPLEMENTATION COMPLETE - READY TO TEST

## Bugs Fixed (Feb 1-2, 2026)

### Bug #1: pipe=-1 when b_sh_wr_iters=1
- **Location:** `marlin_template.h:1899`, `marlin_moe_wna16/marlin_template.h:2046`
- **Problem:** When M is small, `b_sh_wr_iters=1`, causing condition `k >= b_sh_wr_iters - 2` to always be true
- **Impact:** `matmul()` received `pipe=-1`, causing invalid memory access → NaN output
- **Fix:** Changed `pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0)` to `pipe - (b_sh_wr_iters > 1 && k >= b_sh_wr_iters - 2 ? 1 : 0)`

### Bug #2: s_tb_groups calculation for SM70
- **Location:** `marlin_template.h:582`, `marlin_moe_wna16/marlin_template.h:714`
- **Problem:** SM70 with `thread_k_blocks=1` needs different group indexing than SM75+
- **Fix:** Changed `div_ceil(group_blocks, thread_k_blocks)` to `1` for SM70

### Bug #3: Missing SM70 U4 dequantization
- **Location:** `dequant.h:145-175`
- **Problem:** SM70's m8n8k4 has different fragment layout than SM75+ m16n8k16
- **Fix:** Added SM70-specific U4 dequant with direct LOP3 extraction

### Bug #4: Debug printf statements
- **Location:** `marlin_template.h`
- **Fix:** Removed all 15 debug printf statements

### Bug #5: MoE Marlin thread config mismatch
- **Location:** `csrc/moe/marlin_moe_wna16/ops.cu:124-140`, `csrc/quantization/marlin/marlin.cu:128-143`
- **Problem:** Generated SM70 kernels use `thread_k_blocks=1,2` (thread_k=16,32) but runtime configs only had thread_k=64,128
- **Impact:** "Invalid thread config: thread_k = -1, thread_n = -1" error
- **Fix:** Added SM70-compatible thread configs:

**MoE Marlin (ops.cu):**
```cpp
thread_config_t small_batch_thread_configs[] = {
    {128, 128, 256}, {64, 128, 128}, {128, 64, 128},
    // SM70 configs
    {16, 256, 128}, {16, 512, 128}};

thread_config_t large_batch_thread_configs[] = {
    {64, 256, 256}, {64, 128, 128}, {128, 64, 128},
    // SM70 configs
    {32, 512, 256}, {32, 1024, 256}};
```

**Standard Marlin (marlin.cu):**
```cpp
thread_config_t small_batch_thread_configs[] = {
    {128, 128, 256}, {64, 128, 128}, {128, 64, 128},
    // SM70 configs
    {16, 256, 128}, {16, 384, 128}, {16, 512, 128}};

thread_config_t large_batch_thread_configs[] = {
    {64, 256, 256}, {64, 128, 128}, {128, 64, 128},
    // SM70 configs
    {16, 512, 256}, {16, 768, 256}, {16, 1024, 256}};
```

## Files Modified
1. `csrc/quantization/marlin/marlin_template.h` - pipe=-1 fix, s_tb_groups fix, printf removal
2. `csrc/moe/marlin_moe_wna16/marlin_template.h` - pipe=-1 fix, s_tb_groups fix
3. `csrc/quantization/marlin/dequant.h` - SM70 U4 dequant implementation
4. `csrc/moe/marlin_moe_wna16/ops.cu` - Thread config arrays for SM70 MoE
5. `csrc/quantization/marlin/marlin.cu` - Thread config arrays for SM70 standard Marlin

## SM70 Architecture Constraints
- Only supports m8n8k4 instruction (emulates m16n8k16 with 8 operations)
- `thread_k_blocks = 1` for small batch, `thread_k_blocks = 2` for large batch
- `stages = 2` (not 4 like SM80+)
- FP16 only (no BF16, no FP8 activations, no INT8 activations)
- No native ldmatrix - emulated via shuffles

## Build Instructions
```bash
# On remote server (root@10.0.0.129), in container stupefied_nash
source /workspace/vllm/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 CCACHE_NOHASHDIR=true TORCH_CUDA_ARCH_LIST="7.0" pip install -e . --no-build-isolation
```

## Test Commands
```bash
# Per-channel quantization
python tests/kernels/quantization/debug_marlin_sm70.py --M 16 --N 256 --K 128 --group_size 128

# Grouped quantization
python tests/kernels/quantization/debug_marlin_sm70.py --M 16 --N 256 --K 128 --group_size 32

# Full test suite
pytest tests/kernels/quantization/test_marlin_gemm.py -v -k sm70

# MoE test with GLM model
# Model: /workspace/safetensors/MidnightPhreaker-GLM-4.5-Air-REAP-82B-GPTQ-FP16
```

## Test Status
| Test | Status |
|------|--------|
| Per-channel (group_blocks=-1) | ✅ Works |
| Grouped quantization (group_blocks≥0) | 🔄 Pending test after fixes |
| MoE GLM-4.5-Air-82B-GPTQ | 🔄 Pending test after thread config fix |

## Key Files Reference
- `csrc/quantization/marlin/marlin_mma_sm70.h` - MMA emulation functions
- `csrc/quantization/marlin/marlin_template.h` - Main kernel template
- `csrc/moe/marlin_moe_wna16/marlin_template.h` - MoE kernel template
- `csrc/quantization/marlin/dequant.h` - Dequantization functions
- `csrc/moe/marlin_moe_wna16/kernel_selector.h` - Generated kernel dispatch
