#!/usr/bin/env python3
"""R12 Patch: Ultra-Compact Lossless Draft LM Head Types (Q3_K, Q2_K, Q2_0, IQ4_NL, IQ4_XS).

Extends R6's `LLAMA_MTP_DRAFT_HEAD_TYPE` (in `src/llama-model.cpp`) to support ultra-compact
2-bit / 3-bit / IQ4 quant types that require no imatrix:
  - q3_k   (3.44 bpw = 437 MiB, natively supported by mmq-few for C=8 M=8!)
  - q2_k   (2.63 bpw = 334 MiB, 38% smaller than MXFP4 540 MiB)
  - q2_0   (2.50 bpw = 318 MiB, 41% smaller than MXFP4)
  - iq4_nl (4.50 bpw = 572 MiB, natively supported by mmq-few)
  - iq4_xs (4.25 bpw = 540 MiB, natively supported by mmq-few)
Target verification STILL runs on exact 16-bit BF16 `output.weight` -> 100% lossless!

Also updates `mmvq.cu` `MMVQ_PARAMETERS_GB10` (Thor CC 1100) `calc_rows_per_block` at `ncols_dst=1`
so `Q2_K`, `Q3_K`, `Q2_0` share `q8_1` activation loads across 2 rows (`rows_per_block=2`),
matching R7b's `-15.4%` win on `MXFP4`.
"""
import pathlib
import sys


def patch_llama_model(root: pathlib.Path) -> None:
    path = root / "src/llama-model.cpp"
    text = path.read_text(encoding="utf-8")
    if "GGML_TYPE_Q3_K" in text and "LLAMA_MTP_DRAFT_HEAD_TYPE" in text:
        print("[R12] src/llama-model.cpp already patched")
        return

    anchor = '        else if (mode == "f8_e4m3") draft_type = GGML_TYPE_F8_E4M3;'
    if anchor not in text:
        raise RuntimeError(f"R6 draft_type anchor not found in {path}")
    repl = (
        anchor
        + '\n        else if (mode == "q3_k")    draft_type = GGML_TYPE_Q3_K;'
        + '\n        else if (mode == "q2_k")    draft_type = GGML_TYPE_Q2_K;'
        + '\n        else if (mode == "q2_0")    draft_type = GGML_TYPE_Q2_0;'
        + '\n        else if (mode == "iq4_nl")  draft_type = GGML_TYPE_IQ4_NL;'
        + '\n        else if (mode == "iq4_xs")  draft_type = GGML_TYPE_IQ4_XS;'
    )
    text = text.replace(anchor, repl, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[R12] patched {path}")


def patch_mmvq(root: pathlib.Path) -> None:
    path = root / "ggml/src/ggml-cuda/mmvq.cu"
    text = path.read_text(encoding="utf-8")
    if "R12_THOR_LOWBIT_2ROWS" in text:
        print("[R12] mmvq.cu already patched")
        return

    anchor = "return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_MXFP4 ? 2 : 1;"
    if anchor not in text:
        raise RuntimeError(f"R7b MXFP4 2-rows anchor not found in {path}")
    repl = (
        "// R12_THOR_LOWBIT_2ROWS: share q8_1 loads across 2 rows for 2-4 bit draft heads on Thor\n"
        "                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_MXFP4 ||\n"
        "                           type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q2_0 ? 2 : 1;"
    )
    text = text.replace(anchor, repl, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[R12] patched {path}")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <llama.cpp-dir>")
        return 2
    root = pathlib.Path(sys.argv[1])
    patch_llama_model(root)
    patch_mmvq(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
