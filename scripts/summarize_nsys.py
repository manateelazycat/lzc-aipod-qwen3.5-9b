#!/usr/bin/env python3
"""Parse and categorize nsys cuda_gpu_kern_sum CSV files for Qwen3.5-9B."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def categorize(name: str) -> str:
    n = name.lower()
    if "mul_mat_vec_q" in n:
        return "MMVQ (M=1..N quant GEMV)"
    if "mul_mat_q_few" in n:
        return "MMQ_FEW (small-M quant GEMM)"
    if "mul_mat_q" in n:
        return "MMQ (large-M quant GEMM)"
    if "mul_mat_vec_f" in n or "mul_mat_f" in n or "nvjet" in n or "cutlass" in n or "gemm" in n:
        return "FP16/BF16_GEMM (lm_head / cuBLAS)"
    if "gdn_single" in n:
        return "GDN_MMA (Tensor Core Gated DeltaNet)"
    if "gated_delta_net" in n:
        return "GDN_AR (scalar Gated DeltaNet)"
    if "flash_attn" in n:
        return "FLASH_ATTN (Full Attention)"
    if "ssm_conv" in n:
        return "SSM_CONV1D"
    if "quantize" in n:
        return "QUANTIZE_ACT"
    if "get_rows" in n:
        return "GET_ROWS"
    if "rms_norm" in n or "norm_f32" in n or "l2_norm" in n:
        return "NORM"
    if "cpy" in n or "concat" in n or "set_rows" in n:
        return "MEM_CPY_CONCAT"
    return "OTHER_ELEMWISE"


def summarize(csv_path: Path) -> None:
    lines = [l for l in csv_path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    start = next((i for i, l in enumerate(lines) if l.startswith("Time (%)")), None)
    if start is None:
        print(f"{csv_path.name}: no CSV table found")
        return
    reader = csv.DictReader(lines[start:])
    cats: dict[str, float] = defaultdict(float)
    total_ns = 0.0
    total_inst = 0
    top = []
    for row in reader:
        ns = float(row["Total Time (ns)"])
        inst = int(row["Instances"])
        nm = row["Name"]
        total_ns += ns
        total_inst += inst
        cats[categorize(nm)] += ns
        if len(top) < 8:
            top.append((float(row["Time (%)"]), ns / 1e6, inst, float(row["Avg (ns)"]) / 1e3, nm[:85]))
    print(f"=== {csv_path.name}: total={total_ns/1e6:.2f} ms, launches={total_inst:,} ===")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:38s}: {v/1e6:8.2f} ms ({100.0*v/max(total_ns,1):5.1f}%)")
    print("  Top kernels:")
    for pct, ms, inst, avg_us, nm in top:
        print(f"    {pct:5.1f}% {ms:8.2f} ms ({inst:6d}x, avg {avg_us:7.1f} us) {nm}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(Path(p))
