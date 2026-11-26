#!/usr/bin/env python3
"""R17 Patch: Thor 20-SM / Orin 16-SM mmq-few.cu Occupancy & 3-Stage cp.async Pipeline Tuning.

Problem diagnosed from R15 nsys & ncu (r15_c1 & r15_ncu_sol.csv):
  - `mul_mat_q_few<Q4_K>` + `<Q6_K>` accounts for 49.6% of all C=1 GPU time (2,573 ms).
  - R15 `ncu` revealed `Achieved Occupancy` of only 14.5% (Q4_K) and 16.0% (Q6_K) because
    `__launch_bounds__(128, 1)` permits up to 255 regs/thread (limiting to ~2 blocks/SM)
    while `mmq_few_nstages(Q4_K) == 2` leaves only 1 cp.async group in flight on high-latency
    LPDDR5X UMA.

Solution (R17):
  - Specialize `__launch_bounds__(128, mmq_few_min_blocks<type, ncg, nmt>())` to target
    higher SM occupancy (up to 3-4 blocks/SM for ncg<=2, nmt==1) while respecting zero-spill
    constraints (verified with ptxas -v / ncu).
  - Add configurable `MMQ_FEW_Q4K_STAGES` compile-time knob (default 3 on Thor/Orin) for
    deeper 2-in-flight cp.async ring pipelining on LPDDR5X UMA.
  - Keep `GGML_CUDA_MMQ_FEW_NO_PREFETCH` runtime/compile check clean and preserve all
    R13 `MMQ_FEW_MAX_NCG=4` unlocks.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch_mmq_few(src_root: Path, min_blocks: int = 3, q4k_stages: int = 3) -> None:
    p = src_root / "ggml/src/ggml-cuda/mmq-few.cu"
    text = p.read_text(encoding="utf-8")

    # Idempotent reset if re-tuning parameters
    if "R17_MMQ_FEW_OCC_PIPE" in text:
        import re
        text = re.sub(
            r"static constexpr int R17_MIN_BLOCKS = \d+;",
            f"static constexpr int R17_MIN_BLOCKS = {min_blocks};",
            text,
        )
        text = re.sub(
            r"static constexpr int R17_Q4K_STAGES = \d+;",
            f"static constexpr int R17_Q4K_STAGES = {q4k_stages};",
            text,
        )
        p.write_text(text, encoding="utf-8")
        print(f"[R17] updated {p} (min_blocks={min_blocks}, q4k_stages={q4k_stages})")
        return

    target_stages = """// A third ring stage only makes Q6_K faster, the other types get slower on large matrices.
static constexpr __host__ __device__ int mmq_few_nstages(const ggml_type type) {
    return type == GGML_TYPE_Q6_K ? 3 : 2;
}"""

    repl_stages = f"""// R17_MMQ_FEW_OCC_PIPE: Thor 20-SM / Orin 16-SM LPDDR5X pipeline & occupancy tuning
static constexpr int R17_MIN_BLOCKS = {min_blocks};
static constexpr int R17_Q4K_STAGES = {q4k_stages};

static constexpr __host__ __device__ int mmq_few_nstages(const ggml_type type) {{
    if (type == GGML_TYPE_Q6_K) {{
        return 3;
    }}
    if (type == GGML_TYPE_Q4_K) {{
        return R17_Q4K_STAGES;
    }}
    return 2;
}}

template <ggml_type type, int ncg, int nmt>
static constexpr int mmq_few_min_blocks_per_sm() {{
    if constexpr (ncg <= 2 && nmt == 1 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K)) {{
        return R17_MIN_BLOCKS;
    }}
    if constexpr (ncg <= 2 && nmt == 2 && type == GGML_TYPE_Q4_K) {{
        return R17_MIN_BLOCKS > 2 ? 2 : R17_MIN_BLOCKS;
    }}
    return 1;
}}"""

    if target_stages not in text:
        raise RuntimeError(f"Could not find mmq_few_nstages anchor in {p}")
    text = text.replace(target_stages, repl_stages, 1)

    target_lb = "__launch_bounds__(MMQ_FEW_NWARPS*ggml_cuda_get_physical_warp_size(), 1)"
    repl_lb = "__launch_bounds__(MMQ_FEW_NWARPS*ggml_cuda_get_physical_warp_size(), mmq_few_min_blocks_per_sm<type, ncg, nmt>())"
    if target_lb not in text:
        raise RuntimeError(f"Could not find __launch_bounds__ anchor in {p}")
    text = text.replace(target_lb, repl_lb, 1)

    p.write_text(text, encoding="utf-8")
    print(f"[R17] patched {p} (min_blocks={min_blocks}, q4k_stages={q4k_stages})")


def main() -> int:
    src_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/nvidia/qwen35-9b-work/llama.cpp")
    min_blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    q4k_stages = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    patch_mmq_few(src_root, min_blocks=min_blocks, q4k_stages=q4k_stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
