#!/usr/bin/env python3
"""R3 Patch: Enable LPDDR5X UMA L2 prefetch, MMQ-few min_ncols=2, MMVQ UMA table, FP32 acc, and FA whole tiles for GGML_CUDA_CC_THOR (1100)."""
from pathlib import Path
import sys

root = Path(sys.argv[1])

# 1. mmvq.cu: enable L2 prefetch + GB10-class UMA table on Thor (CC 1100)
mmvq = root / "ggml/src/ggml-cuda/mmvq.cu"
txt = mmvq.read_text()
repls = [
    ("#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10\n// returns true only for those quants",
     "#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10 || __CUDA_ARCH__ == GGML_CUDA_CC_THOR\n// returns true only for those quants"),
    ("#elif __CUDA_ARCH__ == GGML_CUDA_CC_GB10\n    return MMVQ_PARAMETERS_GB10;",
     "#elif __CUDA_ARCH__ == GGML_CUDA_CC_GB10 || __CUDA_ARCH__ == GGML_CUDA_CC_THOR\n    return MMVQ_PARAMETERS_GB10;"),
    ("if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_GB10) {",
     "if (GGML_CUDA_CC_IS_NVIDIA(cc) && (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_GB10 || ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_THOR)) {"),
    ("if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_GB10) {",
     "if (GGML_CUDA_CC_IS_NVIDIA(cc) && (cc == GGML_CUDA_CC_GB10 || cc == GGML_CUDA_CC_THOR)) {"),
    ("#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10\n        // start the next iterations",
     "#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10 || __CUDA_ARCH__ == GGML_CUDA_CC_THOR\n        // start the next iterations"),
]
for old, new in repls:
    if old in txt:
        txt = txt.replace(old, new, 1)
mmvq.write_text(txt)
print("Patched mmvq.cu for GGML_CUDA_CC_THOR (1100)")

# 2. mmq-few.cu: enable LPDDR5X L2 prefetch and min_ncols=2 on Thor (CC 1100)
mmqf = root / "ggml/src/ggml-cuda/mmq-few.cu"
txt = mmqf.read_text()
repls2 = [
    ("#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10\n    // DGX Spark DRAM",
     "#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10 || __CUDA_ARCH__ == GGML_CUDA_CC_THOR\n    // DGX Spark / Jetson Thor LPDDR5X DRAM"),
    ("if (cc == GGML_CUDA_CC_GB10 && ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_GB10) {",
     "if ((cc == GGML_CUDA_CC_GB10 && ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_GB10) || (cc == GGML_CUDA_CC_THOR && ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_THOR)) {"),
    ("if (cc == GGML_CUDA_CC_GB10) { // tuned on DGX Spark GB10, also for MUL_MAT_ID",
     "if (cc == GGML_CUDA_CC_GB10 || cc == GGML_CUDA_CC_THOR) { // tuned on LPDDR5X UMA (GB10 & Thor), also for MUL_MAT_ID"),
]
for old, new in repls2:
    if old in txt:
        txt = txt.replace(old, new, 1)
mmqf.write_text(txt)
print("Patched mmq-few.cu for GGML_CUDA_CC_THOR (1100)")

# 3. fattn-common.cuh: enable prefer_whole_tiles on Thor (CC 1100)
fac = root / "ggml/src/ggml-cuda/fattn-common.cuh"
txt = fac.read_text()
old_fa = "const bool prefer_whole_tiles = GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_GB10 && async_kv_preload && scan_mask;"
new_fa = "const bool prefer_whole_tiles = GGML_CUDA_CC_IS_NVIDIA(cc) && (cc == GGML_CUDA_CC_GB10 || cc == GGML_CUDA_CC_THOR) && async_kv_preload && scan_mask;"
if old_fa in txt:
    txt = txt.replace(old_fa, new_fa, 1)
    fac.write_text(txt)
    print("Patched fattn-common.cuh for GGML_CUDA_CC_THOR (1100)")

# 4. ggml-cuda.cu: should_use_fp32_acc for Thor (CC 1100)
gc = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
txt = gc.read_text()
old_gc = "        if (cc == GGML_CUDA_CC_GB10) {\n            return true;\n        }"
new_gc = "        if (cc == GGML_CUDA_CC_GB10 || cc == GGML_CUDA_CC_THOR) {\n            return true;\n        }"
if old_gc in txt:
    txt = txt.replace(old_gc, new_gc, 1)
    gc.write_text(txt)
    print("Patched ggml-cuda.cu should_use_fp32_acc for GGML_CUDA_CC_THOR (1100)")
