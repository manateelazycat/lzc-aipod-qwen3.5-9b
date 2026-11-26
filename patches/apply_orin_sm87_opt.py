#!/usr/bin/env python3
"""Orin sm_87 Hardware-Specific Optimization Patch (Protocol §20).

Applies Jetson AGX Orin 64GB (sm_87 Ampere: 16 SM, 4 MB L2, 164 KB SMEM/SM,
204.8 GB/s LPDDR5, 61.4 GiB UMA, 12-core Cortex-A78AE, JetPack 7 / CUDA 13.2)
optimizations on top of R1-R19:

  1. `common.cuh`: Define `GGML_CUDA_CC_ORIN 870` (sm_87).
  2. `gated-delta-net-mma.cu`: Upstream missed `sm_87` (only allowed `GA10X=860`!),
     causing all 24 GDN layers during Prefill to fall back to slow scalar `gdn_path::ar`.
     Unlock `GGML_CUDA_CC_ORIN` for Ampere Tensor Core MMA `gdn_single`!
  3. `mmf.cuh`: Extend R7b `__ldcs` streaming loads to `GGML_CUDA_CC_ORIN` so the
     2,034 MiB BF16 `output.weight` lm_head does NOT evict Orin's tiny 4 MB L2 cache (§20.4).
  4. `mmq-few.cu`: Unlock `mmq-few min_ncols=2` on `GGML_CUDA_CC_ORIN` (Ampere INT8 MMA
     + cp.async), while deliberately keeping Thor's full-row upfront L2 prefetch OFF on
     Orin to protect its 4 MB L2 (§20.3).
  5. `mmvq.cu`: Enable `rows_per_block=2` (for ultra-compact 334 MiB `Q2_K` / `MXFP4`
     Draft `lm_head`) and LPDDR5 UMA MMVQ tuning on `GGML_CUDA_CC_ORIN` (§20.4/§20.5).
  6. `fattn-common.cuh`: Enable `prefer_whole_tiles` on `GGML_CUDA_CC_ORIN` UMA.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch_file(p: Path, tag: str, replacements: list[tuple[str, str]]) -> None:
    text = p.read_text(encoding="utf-8")
    if tag in text:
        print(f"[ORIN] already patched {p}")
        return
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"[ORIN] anchor not found in {p}: {old!r}")
        text = text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8")
    print(f"[ORIN] patched {p}")


def apply_all(src_root: Path) -> None:
    # 1. common.cuh
    patch_file(
        src_root / "ggml/src/ggml-cuda/common.cuh",
        "GGML_CUDA_CC_ORIN",
        [(
            "#define GGML_CUDA_CC_THOR            1100 // Jetson AGX Thor (Blackwell SM110)",
            "#define GGML_CUDA_CC_ORIN            870  // Jetson AGX Orin (Ampere SM87)\n"
            "#define GGML_CUDA_CC_THOR            1100 // Jetson AGX Thor (Blackwell SM110)",
        )],
    )

    # 2. gated-delta-net-mma.cu: unlock Ampere sm_87 Tensor Core MMA GDN
    patch_file(
        src_root / "ggml/src/ggml-cuda/gated-delta-net-mma.cu",
        "GGML_CUDA_CC_ORIN",
        [(
            "info.cc != GGML_CUDA_CC_THOR && info.cc != GGML_CUDA_CC_BLACKWELL && info.cc != GGML_CUDA_CC_GB10)) {",
            "info.cc != GGML_CUDA_CC_ORIN && info.cc != GGML_CUDA_CC_THOR && info.cc != GGML_CUDA_CC_BLACKWELL && info.cc != GGML_CUDA_CC_GB10)) {",
        )],
    )

    # 3. mmf.cuh: extend __ldcs streaming load on BF16 lm_head to Orin 4 MB L2
    patch_file(
        src_root / "ggml/src/ggml-cuda/mmf.cuh",
        "GGML_CUDA_CC_ORIN",
        [(
            "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_THOR",
            "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_THOR || __CUDA_ARCH__ == GGML_CUDA_CC_ORIN)",
        )],
    )

    # 4. mmq-few.cu: unlock min_ncols=2 on Orin (DO NOT enable Thor full-row L2 prefetch on 4 MB L2!)
    patch_file(
        src_root / "ggml/src/ggml-cuda/mmq-few.cu",
        "GGML_CUDA_CC_ORIN",
        [(
            "if (cc == GGML_CUDA_CC_GB10 || cc == GGML_CUDA_CC_THOR) { // tuned on LPDDR5X UMA (GB10 & Thor), also for MUL_MAT_ID",
            "if (cc == GGML_CUDA_CC_GB10 || cc == GGML_CUDA_CC_THOR || cc == GGML_CUDA_CC_ORIN) { // tuned on LPDDR5/5X UMA (GB10, Thor & Orin)",
        )],
    )

    # 5. mmvq.cu: enable wide 2-row MMVQ (Q2_K/MXFP4 Draft lm_head) on Orin sm_87
    patch_file(
        src_root / "ggml/src/ggml-cuda/mmvq.cu",
        "GGML_CUDA_CC_ORIN",
        [
            (
                "#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10 || __CUDA_ARCH__ == GGML_CUDA_CC_THOR",
                "#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10 || __CUDA_ARCH__ == GGML_CUDA_CC_THOR || __CUDA_ARCH__ == GGML_CUDA_CC_ORIN",
            ),
            (
                "#elif __CUDA_ARCH__ == GGML_CUDA_CC_GB10 || __CUDA_ARCH__ == GGML_CUDA_CC_THOR",
                "#elif __CUDA_ARCH__ == GGML_CUDA_CC_GB10 || __CUDA_ARCH__ == GGML_CUDA_CC_THOR || __CUDA_ARCH__ == GGML_CUDA_CC_ORIN",
            ),
            (
                "if (GGML_CUDA_CC_IS_NVIDIA(cc) && (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_GB10 || ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_THOR)) {",
                "if (GGML_CUDA_CC_IS_NVIDIA(cc) && (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_GB10 || ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_THOR || ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_ORIN)) {",
            ),
            (
                "if (GGML_CUDA_CC_IS_NVIDIA(cc) && (cc == GGML_CUDA_CC_GB10 || cc == GGML_CUDA_CC_THOR)) {",
                "if (GGML_CUDA_CC_IS_NVIDIA(cc) && (cc == GGML_CUDA_CC_GB10 || cc == GGML_CUDA_CC_THOR || cc == GGML_CUDA_CC_ORIN)) {",
            ),
            (
                "#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10 || __CUDA_ARCH__ == GGML_CUDA_CC_THOR",
                "#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10 || __CUDA_ARCH__ == GGML_CUDA_CC_THOR || __CUDA_ARCH__ == GGML_CUDA_CC_ORIN",
            ),
        ],
    )

    # 6. fattn-common.cuh: whole tiles on Orin UMA
    patch_file(
        src_root / "ggml/src/ggml-cuda/fattn-common.cuh",
        "GGML_CUDA_CC_ORIN",
        [(
            "const bool prefer_whole_tiles = GGML_CUDA_CC_IS_NVIDIA(cc) && (cc == GGML_CUDA_CC_GB10 || cc == GGML_CUDA_CC_THOR) && async_kv_preload && scan_mask;",
            "const bool prefer_whole_tiles = GGML_CUDA_CC_IS_NVIDIA(cc) && (cc == GGML_CUDA_CC_GB10 || cc == GGML_CUDA_CC_THOR || cc == GGML_CUDA_CC_ORIN) && async_kv_preload && scan_mask;",
        )],
    )


def main() -> int:
    src_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/nvidia/qwen35-9b-work/llama.cpp")
    apply_all(src_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
