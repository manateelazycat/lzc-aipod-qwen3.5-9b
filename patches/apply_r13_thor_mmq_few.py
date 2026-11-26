#!/usr/bin/env python3
"""R13 Patch: Thor 20-SM mmq-few.cu Deep Micro-Architecture Tuning + NCG=4 (M=32) Unlock.

Nsight Systems (R10) showed:
  1. `mul_mat_q_few` is the #1 kernel in BOTH C=1 (49.5%, 2,606 ms) and C=8 (43.4%, 1,817 ms).
  2. Upstream `MMQ_FEW_MAX_NCG = 3` caps mmq-few at 24 columns (`3 * 8`), causing `K=3` at C=8
     (`8 * 4 = 32` verify tokens) to fall off a cliff into slower `mul_mat_q M=32` (dropping C=8
     from 220 -> 176 tok/s).
  3. On Thor (20 SMs @ 273 GB/s = 13.65 GB/s/SM vs GB10 48 SMs @ 5.7 GB/s/SM), Q4_K benefits from
     testing:
     - `MMQ_FEW_MAX_NCG = 4` (unlocking M=25..32 for K=3 @ C=8!)
     - Env-tunable `GGML_CUDA_MMQ_FEW_NCG1_TALL=1` (`nmt=2`, 32 rows/CTA for `ncg==1`, halving
       src1 loads per weight byte and CTA launch overhead on 4096/12288-row Q4_K layers!)
     - Env-tunable `GGML_CUDA_MMQ_FEW_MAX_NCG` (3 vs 4) for clean A/B comparison.
"""
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <llama.cpp-dir>")
        return 2
    root = pathlib.Path(sys.argv[1])
    path = root / "ggml/src/ggml-cuda/mmq-few.cu"
    text = path.read_text(encoding="utf-8")

    if "R13_THOR_MMQ_FEW" in text:
        print("[R13] mmq-few.cu already patched")
        return 0

    # 1. Expand MMQ_FEW_MAX_NCG from 3 -> 4 (unlocks M=25..32)
    a1 = "static constexpr int MMQ_FEW_MAX_NCG = 3;  // max. number of column groups"
    if a1 not in text:
        raise RuntimeError(f"MMQ_FEW_MAX_NCG anchor not found in {path}")
    text = text.replace(a1, "static constexpr int MMQ_FEW_MAX_NCG = 4;  // R13_THOR_MMQ_FEW: unlock M=25..32 (K=3 @ C=8)", 1)

    # 2. Allow tall (nmt=2, 32-row tile) for ncg==1 on Thor via env GGML_CUDA_MMQ_FEW_NCG1_TALL=1
    a2 = """static int mmq_few_nmt(const ggml_type type, const int ncg, const int64_t nrows_x) {
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    return mmq_few_tall_supported(type) && ncg == 2 && nrows_x >= int64_t(nsm)*2*MMQ_FEW_NROWS ? 2 : 1;
}"""
    if a2 not in text:
        raise RuntimeError(f"mmq_few_nmt anchor not found in {path}")
    r2 = """static int mmq_few_nmt(const ggml_type type, const int ncg, const int64_t nrows_x) {
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    static const bool env_ncg1_tall = []() {
        const char * e = getenv("GGML_CUDA_MMQ_FEW_NCG1_TALL");
        return e && atoi(e) != 0;
    }();
    const bool allow_ncg = (ncg == 2) || (env_ncg1_tall && ncg == 1);
    return mmq_few_tall_supported(type) && allow_ncg && nrows_x >= int64_t(nsm)*2*MMQ_FEW_NROWS ? 2 : 1;
}"""
    text = text.replace(a2, r2, 1)

    # 3. Support ncg=1 tall + ncg=4 in launch_mul_mat_q_few_switch_ncg
    a3 = """    switch (ncg) {
        case 1:
            launch_mul_mat_q_few<type, 1, 1>(src0, src1, dst, src1_q, stream);
            break;
        case 2:
            launch_mul_mat_q_few_tall<type>(src0, src1, dst, src1_q, stream);
            break;
        case 3:
            launch_mul_mat_q_few<type, 3, 1>(src0, src1, dst, src1_q, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }"""
    if a3 not in text:
        raise RuntimeError(f"switch (ncg) anchor not found in {path}")
    r3 = """    switch (ncg) {
        case 1:
            if constexpr (mmq_few_tall_supported(type)) {
                if (mmq_few_nmt(type, 1, src0->ne[1]) == 2) {
                    launch_mul_mat_q_few<type, 1, 2>(src0, src1, dst, src1_q, stream);
                    break;
                }
            }
            launch_mul_mat_q_few<type, 1, 1>(src0, src1, dst, src1_q, stream);
            break;
        case 2:
            launch_mul_mat_q_few_tall<type>(src0, src1, dst, src1_q, stream);
            break;
        case 3:
            launch_mul_mat_q_few<type, 3, 1>(src0, src1, dst, src1_q, stream);
            break;
        case 4:
            launch_mul_mat_q_few<type, 4, 1>(src0, src1, dst, src1_q, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }"""
    text = text.replace(a3, r3, 1)

    # 4. Runtime env GGML_CUDA_MMQ_FEW_MAX_NCG (default 4, set 3 to A/B against upstream)
    a4 = "    if (min_ncols == 0 || ncols < min_ncols || ncols > MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS) {"
    if a4 not in text:
        raise RuntimeError(f"should_use_mmq_few max_ncols anchor not found in {path}")
    r4 = """    static const int env_max_ncg = []() {
        const char * e = getenv("GGML_CUDA_MMQ_FEW_MAX_NCG");
        int v = (e && e[0]) ? atoi(e) : MMQ_FEW_MAX_NCG;
        return std::max(1, std::min(v, MMQ_FEW_MAX_NCG));
    }();
    if (min_ncols == 0 || ncols < min_ncols || ncols > env_max_ncg*MMQ_FEW_NCOLS) {"""
    text = text.replace(a4, r4, 1)

    path.write_text(text, encoding="utf-8")
    print(f"[R13] patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
