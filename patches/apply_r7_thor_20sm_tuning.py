#!/usr/bin/env python3
"""R7 Patch (v2 - R7b): Thor 20-SM Kernel Micro-Tuning driven by R6 & R7a Nsight Systems analysis.

R7a Finding (Rejected):
  Promoting GGML_TYPE_MXFP4 to 2x-nwarps (halve_iters=1) in MMVQ_PARAMETERS_GB10 regressed
  mul_mat_vec_q<MXFP4, M=1> from 2,723 us -> 3,634 us (+33%) because 248,320 rows with 4-warp
  blocks doubled shared-memory reduction overhead per row on Thor's 20 SMs.

R7b Optimization (Applied):
1. ggml/src/ggml-cuda/mmvq.cu:
   - Keep MXFP4 at 2 warps/block (halve_iters=0), and set rows_per_block=2 for GGML_TYPE_MXFP4
     at ncols_dst=1 on MMVQ_PARAMETERS_GB10 (Thor CC 1100), sharing q8_1 activation loads across
     2 output rows and halving CTA count from 248,320 -> 124,160 while preserving 2-warp low
     reduction overhead.

2. ggml/src/ggml-cuda/mmf.cuh:
   - Kept from R7a (verified -104 us/call in R7a nsys: 7,888 us -> 7,783 us): on GGML_CUDA_CC_THOR
     (__CUDA_ARCH__ == 1100), stream BF16/F16 lm_head tiles with __ldcs (L2 evict-first) so the
     2,034 MiB linear scan does not thrash Thor's 32 MiB L2 cache.
"""
import pathlib
import sys


def patch_mmvq(root: pathlib.Path) -> None:
    path = root / "ggml/src/ggml-cuda/mmvq.cu"
    text = path.read_text(encoding="utf-8")

    # Revert R7a 2x-nwarps promotion if present
    r7a_block = """                // R7_THOR_MXFP4_PROMOTE: double nwarps for MXFP4/NVFP4 on Thor/GB10 UMA
                case GGML_TYPE_MXFP4:
                case GGML_TYPE_NVFP4:
"""
    if r7a_block in text:
        text = text.replace(r7a_block, "", 1)
        print("[R7b] reverted R7a MXFP4 2x-nwarps promotion in mmvq.cu")

    if "R7B_THOR_MXFP4_2ROWS" not in text:
        target = """                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }"""
        if target not in text:
            raise RuntimeError(f"MMVQ_PARAMETERS_GB10 rows_per_block anchor not found in {path}")
        repl = """                if (table_id == MMVQ_PARAMETERS_GB10) {
                    // R7B_THOR_MXFP4_2ROWS: share q8_1 loads across 2 rows for 4.25-bit MXFP4 lm_head
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_MXFP4 ? 2 : 1;
                }"""
        text = text.replace(target, repl, 1)
        print(f"[R7b] patched {path} (MXFP4 rows_per_block=2)")
    else:
        print(f"[R7b] mmvq.cu already patched")

    path.write_text(text, encoding="utf-8")


def patch_mmf(root: pathlib.Path) -> None:
    path = root / "ggml/src/ggml-cuda/mmf.cuh"
    text = path.read_text(encoding="utf-8")
    if "R7_THOR_MMF_LDCS" in text:
        print(f"[R7b] mmf.cuh already patched (__ldcs kept)")
        return

    target = """#pragma unroll
            for (int i = 0; i < tile_A::I; ++i) {
                tile_xy[i*tile_k_padded + threadIdx.x] = x[(itA*tile_A::I + i)*stride_row  + col];
            }"""
    if target not in text:
        raise RuntimeError(f"mul_mat_f x-load anchor not found in {path}")

    repl = """#pragma unroll
            for (int i = 0; i < tile_A::I; ++i) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_THOR
                // R7_THOR_MMF_LDCS: stream large BF16/F16 lm_head with cache-streaming __ldcs to preserve L2
                if constexpr (sizeof(T) == sizeof(uint32_t)) {
                    const uint32_t raw_x = __ldcs(reinterpret_cast<const uint32_t *>(&x[(itA*tile_A::I + i)*stride_row + col]));
                    tile_xy[i*tile_k_padded + threadIdx.x] = *reinterpret_cast<const T *>(&raw_x);
                } else {
                    tile_xy[i*tile_k_padded + threadIdx.x] = x[(itA*tile_A::I + i)*stride_row + col];
                }
#else
                tile_xy[i*tile_k_padded + threadIdx.x] = x[(itA*tile_A::I + i)*stride_row  + col];
#endif
            }"""
    text = text.replace(target, repl, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[R7b] patched {path} (__ldcs)")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <llama.cpp-dir>")
        return 2
    root = pathlib.Path(sys.argv[1])
    patch_mmvq(root)
    patch_mmf(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
