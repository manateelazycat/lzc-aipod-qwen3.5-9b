#!/usr/bin/env python3
"""R16 Patch: Concurrency-Adaptive Dynamic MTP Draft Depth (LLAMA_MTP_ADAPTIVE).

Problem diagnosed from R12-R15 + slim-0.2.0 benchmarks:
  - Fixed K=3 maximizes C=1 decode (62.2-64.0 tok/s, peak 72.2 tok/s) using Q2_K draft head,
    but at C=8 runs 3 sequential draft steps (3 x 3.86 ms Q2_K lm_head) + M=32 target verify,
    capping C=8 at ~200.5 tok/s.
  - Fixed K=2 / K=1 maximizes high-concurrency C=4 / C=8 throughput (210-240+ tok/s) by
    eliminating excess draft passes and keeping verify batch in mmq-few's ncg=2 sweet spot,
    but sacrifices C=1 peak speed.

Solution (R16):
  - Right before `common_speculative_draft(spec.get())` in `tools/server/server-context.cpp`,
    dynamically cap each drafting slot's `n_max` based on the live active generating slot count
    (`generating.size()`):
      * n_active <= 2 : K_low  (default 3, env `LLAMA_MTP_ADAPTIVE_K_LOW`)
      * 3 <= n_active <= 5 : K_mid  (default 2, env `LLAMA_MTP_ADAPTIVE_K_MID`)
      * n_active >= 6 : K_high (default 1, env `LLAMA_MTP_ADAPTIVE_K_HIGH`)
  - Controlled by `LLAMA_MTP_ADAPTIVE` (default 1; set 0 to disable for A/B).
  - 100% mathematically lossless: Target verification remains exact BF16 and speculative
    rejection sampling is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch_server_context(src_root: Path) -> None:
    p = src_root / "tools/server/server-context.cpp"
    text = p.read_text(encoding="utf-8")

    if "R16_ADAPTIVE_MTP" in text:
        print(f"[R16] already patched {p}")
        return

    target = """        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }"""

    replacement = """        // R16_ADAPTIVE_MTP: concurrency-aware dynamic MTP draft depth
        if (!drafting.empty()) {
            static const int r16_adaptive = []() {
                const char * v = getenv("LLAMA_MTP_ADAPTIVE");
                return v == nullptr ? 1 : atoi(v);
            }();
            if (r16_adaptive > 0) {
                static const int r16_k_low = []() {
                    const char * v = getenv("LLAMA_MTP_ADAPTIVE_K_LOW");
                    return v != nullptr ? atoi(v) : 3;
                }();
                static const int r16_k_mid = []() {
                    const char * v = getenv("LLAMA_MTP_ADAPTIVE_K_MID");
                    return v != nullptr ? atoi(v) : 2;
                }();
                static const int r16_k_high = []() {
                    const char * v = getenv("LLAMA_MTP_ADAPTIVE_K_HIGH");
                    return v != nullptr ? atoi(v) : 1;
                }();
                const size_t n_act = generating.size();
                const int cap = n_act <= 2 ? r16_k_low : (n_act <= 5 ? r16_k_mid : r16_k_high);
                if (cap <= 0) {
                    for (server_slot * sp : drafting) {
                        common_speculative_get_draft_params(spec.get(), sp->id).drafting = false;
                    }
                    drafting.clear();
                } else {
                    for (server_slot * sp : drafting) {
                        auto & dp = common_speculative_get_draft_params(spec.get(), sp->id);
                        dp.n_max = std::min(dp.n_max, cap);
                    }
                }
            }
        }

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }"""

    if target not in text:
        raise RuntimeError(f"Could not find common_speculative_draft anchor in {p}")
    text = text.replace(target, replacement, 1)
    p.write_text(text, encoding="utf-8")
    print(f"[R16] patched {p}")


def main() -> int:
    src_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/nvidia/qwen35-9b-work/llama.cpp")
    patch_server_context(src_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
