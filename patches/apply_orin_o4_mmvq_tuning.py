#!/usr/bin/env python3
"""Orin O4 Patch: MMVQ 1-Row-Per-Warp for Ultra-Compact Draft LM Heads (Q2_K/Q3_K/MXFP4) on sm_87.

In O3 Nsight Systems (`o3_c1.cuda_gpu_kern_sum.csv`), `mul_mat_vec_q<Q2_K, 1>` (Draft `lm_head`,
334 MiB) took 2,572.6 us on AGX Orin (`130.0 GB/s` = only 63.5% of Orin's 204.8 GB/s LPDDR5 peak),
while `mul_mat_f<bf16>` achieved 184.4 GB/s (90.0% of peak).

Root cause in `ggml/src/ggml-cuda/mmvq.cu` (`MMVQ_PARAMETERS_ORIN`):
  `Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0` already use `return nwarps;` (1 row per warp = 4 rows/block,
  zero cross-warp `__syncthreads()`, maximum memory-level parallelism on Orin's 16 SMs),
  whereas `Q2_K / Q3_K / MXFP4` fell through to `rows_per_block = 2`, paying cross-warp
  shared-memory reduction + `__syncthreads()` overhead and cutting in-flight loads in half.

This patch adds `GGML_TYPE_Q2_K`, `GGML_TYPE_Q3_K`, `GGML_TYPE_MXFP4` to the
`MMVQ_PARAMETERS_ORIN` `return nwarps;` switch for `ncols_dst == 1`.
"""
from __future__ import annotations

import sys
from pathlib import Path


def apply_patch(src_root: Path) -> None:
    p = src_root / "ggml/src/ggml-cuda/mmvq.cu"
    text = p.read_text(encoding="utf-8")
    tag = "ORIN_O4_DRAFT_NWARPS"
    if tag in text:
        print(f"[ORIN-O4] already patched {p}")
        return

    anchor = (
        "                    case GGML_TYPE_Q6_K:\n"
        "                    case GGML_TYPE_IQ4_NL:\n"
        "                        return nwarps;"
    )
    repl = (
        "                    case GGML_TYPE_Q6_K:\n"
        "                    case GGML_TYPE_IQ4_NL:\n"
        "                    // ORIN_O4_DRAFT_NWARPS: 1 row/warp (zero __syncthreads) for Q2_K/Q3_K/MXFP4 Draft lm_head on Orin 16 SM\n"
        "                    case GGML_TYPE_Q2_K:\n"
        "                    case GGML_TYPE_Q3_K:\n"
        "                    case GGML_TYPE_MXFP4:\n"
        "                        return nwarps;"
    )
    if anchor not in text:
        raise RuntimeError(f"[ORIN-O4] anchor not found in {p}")
    text = text.replace(anchor, repl, 1)
    p.write_text(text, encoding="utf-8")
    print(f"[ORIN-O4] patched {p}")


def main() -> int:
    src_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/nvidia/qwen35-9b-work/llama.cpp")
    apply_patch(src_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
