#!/usr/bin/env python3
"""R5 Patch: Runtime-tunable MMQ -> cuBLAS Tensor Core batch crossover for Thor (sm_110a).

In upstream ggml-cuda/mmq.cu, ggml_cuda_should_use_mmq() returns true unconditionally
when turing_mma_available(cc) is true (including Blackwell sm_110a).
This patch adds an env-controlled threshold GGML_CUDA_MMQ_MAX_BATCH (default -1 = upstream
behavior) so we can A/B benchmark MMQ INT8-MMA vs cuBLAS FP16/BF16 Tensor Core prefill
at large batch sizes (e.g. ne11 = 512..2048) without recompiling.
"""
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <llama.cpp-dir>")
        return 2
    root = pathlib.Path(sys.argv[1])
    path = root / "ggml/src/ggml-cuda/mmq.cu"
    text = path.read_text(encoding="utf-8")

    marker = "GGML_CUDA_MMQ_MAX_BATCH"
    if marker in text:
        print(f"[R5] already patched: {path}")
        return 0

    target = """    if (turing_mma_available(cc)) {
        return true;
    }"""
    if target not in text:
        raise RuntimeError(f"Anchor not found in {path}")

    replacement = """    // R5: Allow runtime tuning of MMQ vs cuBLAS Tensor Core crossover batch size on Thor (sm_110a).
    static const int64_t mmq_max_batch = []() -> int64_t {
        const char * env = getenv("GGML_CUDA_MMQ_MAX_BATCH");
        if (!env || !env[0]) {
            return -1;
        }
        return atoll(env);
    }();
    if (mmq_max_batch >= 0 && n_experts == 0 && ne11 > mmq_max_batch) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }"""

    text = text.replace(target, replacement, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[R5] patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
