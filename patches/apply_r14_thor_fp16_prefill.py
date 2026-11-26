#!/usr/bin/env python3
"""R14 Patch: Selective Blackwell sm_110a FP16 Tensor Core prefill for quantized weights.

Problem diagnosed from R13 nsys (r13_prefill4k.cuda_gpu_kern_sum.csv, total=1389.29 ms):
  - nvjet_sm110_tss_* (BF16 inputs + CUBLAS_COMPUTE_32F accumulate) consumes 498.34 ms (35.9%).
  - In R3, setting `prefer_fp32_acc = true` for Thor forced quantized weights (Q4_K / Q6_K)
    to dequantize to BF16 and run CUBLAS_COMPUTE_32F (half peak Blackwell Tensor Core TFLOPS
    compared to FP16 CUBLAS_COMPUTE_16F).
  - Setting upstream `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` globally would also override native
    BF16 weights (`output.weight` lm_head, 2,034 MiB), forcing a catastrophic 2 GiB BF16->FP16
    runtime conversion on every C=8 / prefill lm_head GEMM!

Solution (R14):
  1. Add `GGML_CUDA_CUBLAS_COMPUTE_TYPE=quant_f16` mode in `ggml_cuda_mul_mat_cublas`:
     - Quantized weights (`ggml_is_quantized(src0->type)` or FP8): dequantize directly to FP16
       and execute Blackwell `CUBLAS_COMPUTE_16F` Tensor Core GEMM (2x peak TFLOPS), unless the
       op explicitly requires `GGML_PREC_F32` or `GGML_PREC_BF16`.
     - Native BF16 weights (`output.weight` lm_head): remain native `GGML_TYPE_BF16` with
       zero extra weight conversion and full BF16/FP32 logit precision.
  2. Add optional `GGML_CUDA_F16_PREFER_F32_OUT=1` toggle to A/B test `CUBLAS_COMPUTE_16F`
     (FP16 accum + `to_fp32_cuda`) vs `CUBLAS_COMPUTE_32F` (direct FP32 output) on Thor.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch_ggml_cuda(src_root: Path) -> None:
    p = src_root / "ggml/src/ggml-cuda/ggml-cuda.cu"
    text = p.read_text(encoding="utf-8")

    if "R14_QUANT_F16_PREFILL" in text:
        print(f"[R14] already patched {p}")
        return

    # 1. Optional F32 output toggle for F16 cuBLAS path
    target_out = """    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {"""

    repl_out = """    if (compute_type == GGML_TYPE_F16) {
        static const bool r14_f16_f32_out = []() {
            const char * v = getenv("GGML_CUDA_F16_PREFER_F32_OUT");
            return v != nullptr && atoi(v) > 0;
        }();
        prefer_f32_output = r14_f16_f32_out || cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {"""

    if target_out not in text:
        raise RuntimeError(f"Could not find prefer_f32_output anchor in {p}")
    text = text.replace(target_out, repl_out, 1)

    # 2. Add `quant_f16` mode in GGML_CUDA_CUBLAS_COMPUTE_TYPE (cached getenv!)
    target_env = """    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }"""

    repl_env = """    // R14_QUANT_F16_PREFILL: cache env lookup and support selective quant_f16 (FP16 TC for quantized weights, native BF16 for lm_head)
    enum r14_cublas_mode_t { R14_CUBLAS_AUTO = 0, R14_CUBLAS_F32, R14_CUBLAS_F16, R14_CUBLAS_BF16, R14_CUBLAS_QUANT_F16 };
    static const r14_cublas_mode_t r14_cublas_mode = []() {
        const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
        if (env_c == nullptr) {
            return R14_CUBLAS_AUTO;
        }
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            return R14_CUBLAS_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            return R14_CUBLAS_F16;
        } else if (env_cpp == "bf16") {
            return R14_CUBLAS_BF16;
        } else if (env_cpp == "quant_f16" || env_cpp == "qf16") {
            return R14_CUBLAS_QUANT_F16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
        return R14_CUBLAS_AUTO;
    }();
    if (r14_cublas_mode == R14_CUBLAS_F32) {
        compute_type = GGML_TYPE_F32;
    } else if (r14_cublas_mode == R14_CUBLAS_F16) {
        compute_type = GGML_TYPE_F16;
    } else if (r14_cublas_mode == R14_CUBLAS_BF16) {
        compute_type = GGML_TYPE_BF16;
    } else if (r14_cublas_mode == R14_CUBLAS_QUANT_F16) {
        if ((ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F8_E4M3) &&
            prec != GGML_PREC_F32 && prec != GGML_PREC_BF16 && fast_fp16_hardware_available(cc)) {
            compute_type = GGML_TYPE_F16;
        }
    }"""

    if target_env not in text:
        raise RuntimeError(f"Could not find GGML_CUDA_CUBLAS_COMPUTE_TYPE anchor in {p}")
    text = text.replace(target_env, repl_env, 1)

    p.write_text(text, encoding="utf-8")
    print(f"[R14] patched {p}")


def main() -> int:
    src_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/nvidia/qwen35-9b-work/llama.cpp")
    patch_ggml_cuda(src_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
