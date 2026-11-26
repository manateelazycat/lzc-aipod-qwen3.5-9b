#!/usr/bin/env python3
"""R2 Patch: Define GGML_CUDA_CC_THOR (1100) and unlock Blackwell Tensor Core MMA Gated DeltaNet on Thor."""
from pathlib import Path
import sys

root = Path(sys.argv[1])

# 1. common.cuh: add GGML_CUDA_CC_THOR = 1100
common = root / "ggml/src/ggml-cuda/common.cuh"
txt = common.read_text()
if "GGML_CUDA_CC_THOR" not in txt:
    target = "#define GGML_CUDA_CC_HOPPER          900\n"
    assert target in txt, "HOPPER anchor not found"
    txt = txt.replace(target, target + "#define GGML_CUDA_CC_THOR            1100 // Jetson AGX Thor (Blackwell SM110)\n", 1)
    common.write_text(txt)
    print("Patched common.cuh: added GGML_CUDA_CC_THOR=1100")

# 2. gated-delta-net-mma.cu: whitelist GGML_CUDA_CC_THOR (1100)
gdn = root / "ggml/src/ggml-cuda/gated-delta-net-mma.cu"
txt = gdn.read_text()
old = """        (info.cc != GGML_CUDA_CC_AMPERE_GA10X && info.cc != GGML_CUDA_CC_ADA_LOVELACE &&
         info.cc != GGML_CUDA_CC_BLACKWELL && info.cc != GGML_CUDA_CC_GB10)) {"""
new = """        (info.cc != GGML_CUDA_CC_AMPERE_GA10X && info.cc != GGML_CUDA_CC_ADA_LOVELACE &&
         info.cc != GGML_CUDA_CC_THOR && info.cc != GGML_CUDA_CC_BLACKWELL && info.cc != GGML_CUDA_CC_GB10)) {"""
if "GGML_CUDA_CC_THOR" not in txt:
    assert old in txt, "gdn whitelist anchor not found"
    txt = txt.replace(old, new, 1)
    gdn.write_text(txt)
    print("Patched gated-delta-net-mma.cu: enabled Tensor Core MMA GDN on GGML_CUDA_CC_THOR (1100)")
