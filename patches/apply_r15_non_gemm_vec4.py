#!/usr/bin/env python3
"""R15 Patch: Zero-division 128-bit float4 vectorized SwiGLU (unary_gated_op_kernel).

Problem diagnosed from R13/R14 nsys (prefill4k.cuda_gpu_kern_sum.csv):
  - `unary_gated_op_kernel<&op_silu, float>` consumes 108.94 ms (7.8% of 4K prefill, 138 calls,
    avg 789.4 us/call) at only ~65% of Thor's 273 GB/s LPDDR5X bandwidth ceiling.
  - Root cause in ggml/src/ggml-cuda/unary.cu:
      1. Scalar 32-bit float LDG/STG per thread instead of 128-bit float4 vectorized memory ops.
      2. Two 64-bit integer divisions/modulos (`i / n`, `i % n`) executed per scalar element even
         when gate/up tensors are contiguous (`o0 == n && o1 == n`) or row-strided with 16B alignment.

Solution (R15):
  - Add two bitwise-exact 128-bit `float4` kernels for `T == float` in `unary.cu`:
      1. `unary_gated_op_kernel_f32_vec4<op>`: contiguous fast path (`o0 == n && o1 == n`,
         16B-aligned x/g/dst, `k % 4 == 0`), 1D grid over `k/4`, ZERO integer divisions.
      2. `unary_gated_op_kernel_f32_strided_vec4<op>`: packed/strided fast path (`n % 4 == 0`,
         `o0 % 4 == 0`, `o1 % 4 == 0`, `nrows <= 65535`, 16B-aligned x/g/dst), 2D grid
         `dim3(blocks_per_row, nrows)`, ZERO integer divisions.
  - Runtime-toggleable via `GGML_CUDA_GLU_VEC4=0` (default enabled = 1) for clean A/B validation.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch_unary(src_root: Path) -> None:
    p = src_root / "ggml/src/ggml-cuda/unary.cu"
    text = p.read_text(encoding="utf-8")

    if "R15_GLU_FLOAT4_VEC" in text:
        print(f"[R15] already patched {p}")
        return

    target = """template <float (*op)(float), typename T>
static void unary_gated_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_gated_op_kernel<op, T>, launch_params, x, g, dst, k, n, o0, o1);
}"""

    replacement = """// R15_GLU_FLOAT4_VEC: zero-division 128-bit float4 SwiGLU/GEGLU kernels for Thor UMA bandwidth
template <float (*op)(float)>
static __global__ void unary_gated_op_kernel_f32_vec4(
        const float * __restrict__ x, const float * __restrict__ g, float * __restrict__ dst,
        const int64_t k4) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= k4) {
        return;
    }
    ggml_cuda_pdl_sync();
    const float4 vx = reinterpret_cast<const float4 *>(x)[i];
    const float4 vg = reinterpret_cast<const float4 *>(g)[i];
    float4 vd;
    vd.x = op(vx.x) * vg.x;
    vd.y = op(vx.y) * vg.y;
    vd.z = op(vx.z) * vg.z;
    vd.w = op(vx.w) * vg.w;
    reinterpret_cast<float4 *>(dst)[i] = vd;
}

template <float (*op)(float)>
static __global__ void unary_gated_op_kernel_f32_strided_vec4(
        const float * __restrict__ x, const float * __restrict__ g, float * __restrict__ dst,
        const int64_t n4, const int64_t o0_4, const int64_t o1_4) {
    ggml_cuda_pdl_lc();
    const int64_t col4 = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (col4 >= n4) {
        return;
    }
    const int64_t row = blockIdx.y;
    ggml_cuda_pdl_sync();
    const float4 vx = reinterpret_cast<const float4 *>(x)[row * o0_4 + col4];
    const float4 vg = reinterpret_cast<const float4 *>(g)[row * o1_4 + col4];
    float4 vd;
    vd.x = op(vx.x) * vg.x;
    vd.y = op(vx.y) * vg.y;
    vd.z = op(vx.z) * vg.z;
    vd.w = op(vx.w) * vg.w;
    reinterpret_cast<float4 *>(dst)[row * n4 + col4] = vd;
}

template <float (*op)(float), typename T>
static void unary_gated_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, cudaStream_t stream) {
    if constexpr (std::is_same<T, float>::value) {
        static const bool r15_glu_vec4 = []() {
            const char * v = getenv("GGML_CUDA_GLU_VEC4");
            return v == nullptr || atoi(v) != 0;
        }();
        const bool aligned16 =
            (((uintptr_t) x & 15) == 0) &&
            (((uintptr_t) g & 15) == 0) &&
            (((uintptr_t) dst & 15) == 0);
        if (r15_glu_vec4 && aligned16) {
            if (o0 == n && o1 == n && (k & 3) == 0) {
                const int64_t k4 = k >> 2;
                const int64_t num_blocks = (k4 + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
                if (num_blocks <= INT32_MAX) {
                    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
                    ggml_cuda_kernel_launch(unary_gated_op_kernel_f32_vec4<op>, lp, x, g, dst, k4);
                    return;
                }
            } else if (n > 0 && (n & 3) == 0 && (o0 & 3) == 0 && (o1 & 3) == 0 && (k % n) == 0) {
                const int64_t nrows = k / n;
                const int64_t n4 = n >> 2;
                const int64_t blocks_x = (n4 + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
                if (nrows > 0 && nrows <= 65535 && blocks_x <= INT32_MAX) {
                    const dim3 grid((unsigned int)blocks_x, (unsigned int)nrows, 1);
                    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid, CUDA_GLU_BLOCK_SIZE, 0, stream);
                    ggml_cuda_kernel_launch(unary_gated_op_kernel_f32_strided_vec4<op>, lp, x, g, dst, n4, o0 >> 2, o1 >> 2);
                    return;
                }
            }
        }
    }
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_gated_op_kernel<op, T>, launch_params, x, g, dst, k, n, o0, o1);
}"""

    if target not in text:
        raise RuntimeError(f"Could not find unary_gated_cuda anchor in {p}")
    text = text.replace(target, replacement, 1)
    p.write_text(text, encoding="utf-8")
    print(f"[R15] patched {p}")


def main() -> int:
    src_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/nvidia/qwen35-9b-work/llama.cpp")
    patch_unary(src_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
