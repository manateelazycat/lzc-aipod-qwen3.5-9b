#!/usr/bin/env python3
"""R11 Patch: 128-bit Vectorized Streaming (__ldcs uint4) BF16 LM Head Kernel + Dispatch Tuning.

Nsight Systems (R10) showed that Target BF16 lm_head (`mul_mat_f<nv_bfloat162, 32, 3, 8>`,
2,034 MiB) is the #2 hotspot in C=1 (20.5%, 8,120 us/call -> 250.5 GB/s), paying shared-memory
tile_xy + ldmatrix + MMA overhead for only M=3..4 columns while loading 4 bytes per thread.

This patch:
1. Vectorizes the nv_bfloat16 inner loop in `ggml/src/ggml-cuda/mmvf.cu` to 128-bit `uint4`
   cache-streaming `__ldcs` loads (8 BF16 weights per instruction, 2 iterations per 256-thread
   block for K=4096) with float4 activation loads, eliminating L2 thrashing and cutting load
   instruction count by 4x.
2. Adds runtime env `GGML_CUDA_MMVF_BF16_MAX_BATCH` in `ggml_cuda_should_use_mmvf()` and
   `GGML_CUDA_MMF_MAX_BATCH` in `ggml_cuda_should_use_mmf()` to A/B test mmvf-vec128 vs mmf
   vs cuBLAS on Thor sm_110a for BF16 lm_head across M=2..24 without recompiling.
"""
import pathlib
import sys


def patch_mmvf(root: pathlib.Path) -> None:
    path = root / "ggml/src/ggml-cuda/mmvf.cu"
    text = path.read_text(encoding="utf-8")
    if "R11_BF16_VEC128" in text:
        print("[R11] mmvf.cu already patched")
        return

    old_loop = """        const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
        [[maybe_unused]] const nv_bfloat162 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const nv_bfloat162 *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const nv_bfloat162 tmpx = x2[col2];
            [[maybe_unused]] nv_bfloat162 tmpx_gate;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }"""

    if old_loop not in text:
        raise RuntimeError(f"BF16 loop anchor not found in {path}")

    new_loop = """        // R11_BF16_VEC128: 128-bit streaming (__ldcs uint4) BF16 weights + float4 activations on Thor
        if constexpr (!has_fusion) {
            const bool aligned16 = ((ncols2 & 3) == 0) &&
                (((uintptr_t) x & 15) == 0) && (((uintptr_t) y & 15) == 0) &&
                ((stride_col_y2 & 3) == 0);
            if (aligned16) {
                const uint4 * x8 = reinterpret_cast<const uint4 *>(x);
                const float4 * y4 = reinterpret_cast<const float4 *>(y);
                const int ncols8 = ncols2 >> 2;
                const int stride_col_y4 = stride_col_y2 >> 1;
                for (int col8 = tid; col8 < ncols8; col8 += block_size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_THOR
                    const uint4 raw_x = __ldcs(x8 + col8);
#else
                    const uint4 raw_x = x8[col8];
#endif
                    const nv_bfloat162 bx0 = *reinterpret_cast<const nv_bfloat162 *>(&raw_x.x);
                    const nv_bfloat162 bx1 = *reinterpret_cast<const nv_bfloat162 *>(&raw_x.y);
                    const nv_bfloat162 bx2 = *reinterpret_cast<const nv_bfloat162 *>(&raw_x.z);
                    const nv_bfloat162 bx3 = *reinterpret_cast<const nv_bfloat162 *>(&raw_x.w);
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const float4 ya = y4[j * stride_col_y4 + 2 * col8 + 0];
                        const float4 yb = y4[j * stride_col_y4 + 2 * col8 + 1];
                        ggml_cuda_mad(sumf[j], bx0.x, ya.x);
                        ggml_cuda_mad(sumf[j], bx0.y, ya.y);
                        ggml_cuda_mad(sumf[j], bx1.x, ya.z);
                        ggml_cuda_mad(sumf[j], bx1.y, ya.w);
                        ggml_cuda_mad(sumf[j], bx2.x, yb.x);
                        ggml_cuda_mad(sumf[j], bx2.y, yb.y);
                        ggml_cuda_mad(sumf[j], bx3.x, yb.z);
                        ggml_cuda_mad(sumf[j], bx3.y, yb.w);
                    }
                }
            } else {
                const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
                for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                    const nv_bfloat162 tmpx = x2[col2];
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const float2 tmpy = y2[j*stride_col_y2 + col2];
                        ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                        ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
                    }
                }
            }
        } else {
            const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
            const nv_bfloat162 * gate_x2 = use_gate ? (const nv_bfloat162 *) gate_x : nullptr;
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const nv_bfloat162 tmpx = x2[col2];
                nv_bfloat162 tmpx_gate{};
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }"""

    text = text.replace(old_loop, new_loop, 1)

    disp_anchor = """        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);"""
    if disp_anchor not in text:
        raise RuntimeError(f"BF16 dispatch anchor not found in {path}")

    disp_repl = """        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                static const int64_t env_bf16_mmvf_max = []() -> int64_t {
                    const char * e = getenv("GGML_CUDA_MMVF_BF16_MAX_BATCH");
                    return (e && e[0]) ? atoll(e) : -1;
                }();
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                if (env_bf16_mmvf_max >= 0 && src0_small) {
                    return ne11 <= std::min<int64_t>(env_bf16_mmvf_max, MMVF_MAX_BATCH_SIZE);
                }"""
    text = text.replace(disp_anchor, disp_repl, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[R11] patched {path}")


def patch_mmf(root: pathlib.Path) -> None:
    path = root / "ggml/src/ggml-cuda/mmf.cu"
    text = path.read_text(encoding="utf-8")
    if "GGML_CUDA_MMF_MAX_BATCH" in text:
        print("[R11] mmf.cu already patched")
        return

    anchor = """        } else if (src1_ncols > 16) {
            return false;
        }"""
    if anchor not in text:
        raise RuntimeError(f"mmf max batch anchor not found in {path}")
    repl = """        } else {
            static const int64_t env_mmf_max = []() -> int64_t {
                const char * e = getenv("GGML_CUDA_MMF_MAX_BATCH");
                return (e && e[0]) ? atoll(e) : 16;
            }();
            if (src1_ncols > env_mmf_max) {
                return false;
            }
        }"""
    text = text.replace(anchor, repl, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[R11] patched {path}")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <llama.cpp-dir>")
        return 2
    root = pathlib.Path(sys.argv[1])
    patch_mmvf(root)
    patch_mmf(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
