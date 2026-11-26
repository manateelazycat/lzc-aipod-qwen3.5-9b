#!/usr/bin/env python3
"""Steady-state windowed NVIDIA Nsight Systems (`nsys`) & Nsight Compute (`ncu`)
profiler for Qwen3.5-9B `llama-server` on Jetson AGX Thor (`sm_110a`).

Captures steady-state windows (`c1`, `prefill4k`, `c8`) with `--cuda-graph-trace=node`
so CUDA Graph kernels are expanded to individual kernel timings, exports CSV tables,
and immediately removes bulky `.nsys-rep` / `.sqlite` files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_qwen35_9b as bench  # noqa: E402


REPORTS = (
    "cuda_gpu_kern_sum",
    "cuda_api_sum",
    "cuda_gpu_mem_time_sum",
    "osrt_sum",
)


def resolve_nsys(explicit: str) -> str:
    if explicit:
        return explicit
    for cand in (
        "/opt/nvidia/nsight-systems/2026.3.1/bin/nsys",
        "/opt/nvidia/nsight-systems/2025.6.3/bin/nsys",
        "/usr/local/bin/nsys",
    ):
        if Path(cand).is_file():
            return cand
    found = shutil.which("nsys")
    if not found:
        raise RuntimeError("nsys not found")
    return found


def resolve_ncu(explicit: str) -> str:
    if explicit:
        return explicit
    for cand in (
        "/opt/nvidia/nsight-compute/2026.1.1/ncu",
        "/usr/local/cuda/bin/ncu",
    ):
        if Path(cand).is_file():
            return cand
    found = shutil.which("ncu")
    if not found:
        raise RuntimeError("ncu not found")
    return found


def wait_ready(base_url: str, timeout: int = 300) -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=3) as r:
                d = json.loads(r.read().decode("utf-8", errors="replace"))
                if d.get("status") == "ok":
                    return time.monotonic() - t0
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server not ready within {timeout}s")


def run_nsys(args: argparse.Namespace) -> None:
    nsys = resolve_nsys(args.nsys_bin)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    session = f"qwen35_9b_{os.getpid()}"

    llama_args = [
        args.server_bin,
        "-m", args.model_path,
        "--alias", args.model_id,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "-c", str(args.context),
        "-np", str(args.concurrency),
        "-ngl", "999",
        "--jinja",
        "--metrics",
    ]
    if args.mmproj_path:
        llama_args += ["--mmproj", args.mmproj_path]
    if args.extra_args:
        llama_args += shlex.split(args.extra_args)

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    bin_dir = str(Path(args.server_bin).resolve().parent)
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia:" + env.get("LD_LIBRARY_PATH", "")
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v

    base_url = f"http://127.0.0.1:{args.port}"
    log_f = (out_dir / f"{args.prefix}.server.log").open("w", encoding="utf-8")
    launch_cmd = [
        nsys, "launch",
        f"--session-new={session}",
        "--trace=cuda,nvtx,osrt",
        "--cuda-graph-trace=node",
        *llama_args,
    ]
    print(f"[nsys] launching session={session}", flush=True)
    proc = subprocess.Popen(launch_cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        ready_s = wait_ready(base_url, timeout=300)
        print(f"[nsys] server ready in {ready_s:.2f}s", flush=True)

        # Warmup outside capture
        bench.stream_chat(base_url, args.model_id, [{"role": "user", "content": "Say ok."}], max_tokens=32, enable_thinking=False, temperature=0.0)
        wdoc, _ = bench.make_filler(seed=101, target_tokens=1500)
        bench.stream_chat(base_url, args.model_id, [{"role": "user", "content": wdoc + "\nOne word."}], max_tokens=1, enable_thinking=False, temperature=0.0)

        def w_c1() -> dict[str, Any]:
            r1 = bench.stream_chat(base_url, args.model_id, [{"role": "user", "content": f"[Nsys:{args.prefix}:1] " + bench.PROMPT_PROSE}], max_tokens=128, enable_thinking=False, temperature=0.0, ignore_eos=True)
            r2 = bench.stream_chat(base_url, args.model_id, [{"role": "user", "content": f"[Nsys:{args.prefix}:2] " + bench.PROMPT_CODE}], max_tokens=128, enable_thinking=False, temperature=0.0, ignore_eos=True)
            return {"runs": [r1, r2]}

        def w_prefill4k() -> dict[str, Any]:
            doc, _ = bench.make_filler(seed=7777, target_tokens=4096)
            r = bench.stream_chat(base_url, args.model_id, [{"role": "user", "content": doc + "\nOne word."}], max_tokens=1, enable_thinking=False, temperature=0.0)
            return {"request": r}

        def w_c8() -> dict[str, Any]:
            t0 = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futs = [
                    pool.submit(
                        bench.stream_chat,
                        base_url,
                        args.model_id,
                        [{"role": "user", "content": f"[Nsys:{args.prefix}:c8:{i}] " + (bench.PROMPT_PROSE if i % 2 == 0 else bench.PROMPT_CODE)}],
                        96,
                        False,
                        0.0,
                        600.0,
                        ignore_eos=True,
                    )
                    for i in range(8)
                ]
                reqs = [f.result() for f in futs]
            wall = time.monotonic() - t0
            toks = sum(r["completion_tokens"] for r in reqs)
            return {"wall_s": round(wall, 3), "total_tokens": toks, "aggregate_tok_s": round(toks / max(wall, 1e-6), 2)}

        wmap = {"c1": w_c1, "prefill4k": w_prefill4k, "c8": w_c8}
        for win in [w.strip() for w in args.windows.split(",") if w.strip()]:
            wname = f"{args.prefix}_{win}"
            rep_base = out_dir / wname
            rep_file = out_dir / f"{wname}.nsys-rep"
            sqlite_file = out_dir / f"{wname}.sqlite"
            before = bench.fetch_metrics(base_url)
            subprocess.run(
                [nsys, "start", f"--session={session}", "--sample=none", "-o", str(rep_base), "--force-overwrite=true"],
                check=True,
            )
            t0 = time.monotonic()
            try:
                details = wmap[win]()
            finally:
                wall_s = time.monotonic() - t0
                subprocess.run([nsys, "stop", f"--session={session}"], check=True)
            after = bench.fetch_metrics(base_url)
            delta = {k: round(after.get(k, 0.0) - before.get(k, 0.0), 4) for k in set(before) | set(after)}
            (out_dir / f"{wname}.driver.json").write_text(
                json.dumps({"window": wname, "wallSec": round(wall_s, 4), "metricsDelta": delta, "details": details}, indent=2) + "\n",
                encoding="utf-8",
            )
            for _ in range(40):
                if rep_file.is_file() and rep_file.stat().st_size > 0:
                    break
                time.sleep(0.5)
            for rep in REPORTS:
                res = subprocess.run(
                    [nsys, "stats", "--report", rep, "--format", "csv", "--output", "-", str(rep_file)],
                    text=True, capture_output=True, check=False,
                )
                (out_dir / f"{wname}.{rep}.csv").write_text(res.stdout or "", encoding="utf-8")
            rep_file.unlink(missing_ok=True)
            sqlite_file.unlink(missing_ok=True)
            print(f"[nsys] {wname} wall={wall_s:.2f}s done", flush=True)
    finally:
        subprocess.run([nsys, "shutdown", f"--session={session}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_f.close()


def run_ncu(args: argparse.Namespace) -> None:
    ncu = resolve_ncu(args.ncu_bin)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / f"{args.prefix}_ncu_sol.csv"
    bin_dir = str(Path(args.completion_bin).resolve().parent)
    ld_path = f"{bin_dir}:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia"
    target_cmd = [
        args.completion_bin,
        "-m", args.model_path,
        "-c", "4096",
        "-ngl", "999",
        "-n", "4",
        "-no-cnv",
        "--no-warmup",
        "-p", "Explain unified memory bandwidth on NVIDIA Thor.",
    ]
    if args.extra_args:
        target_cmd += shlex.split(args.extra_args)

    cmd = [
        "sudo", "-S", "env",
        f"LD_LIBRARY_PATH={ld_path}",
        "CUDA_VISIBLE_DEVICES=0",
        "GGML_CUDA_DISABLE_GRAPHS=1",
        ncu,
        "--target-processes", "all",
        "--cache-control", "none",
        "--launch-skip", str(args.launch_skip),
        "--launch-count", str(args.launch_count),
        "--section", "SpeedOfLight",
        "--section", "MemoryWorkloadAnalysis",
        "--section", "Occupancy",
        "--section", "SchedulerStats",
        "--csv",
        "--log-file", str(csv_out),
        *target_cmd,
    ]
    print(f"[ncu] capturing {args.launch_count} kernels (skip={args.launch_skip}) -> {csv_out}", flush=True)
    subprocess.run(cmd, input=f"{args.sudo_pass}\n", text=True, check=True)
    print(f"[ncu] done -> {csv_out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp1 = sub.add_parser("nsys")
    sp1.add_argument("--server-bin", required=True)
    sp1.add_argument("--model-path", required=True)
    sp1.add_argument("--mmproj-path", default="")
    sp1.add_argument("--model-id", default="qwen3.5-9b-defiant-fable")
    sp1.add_argument("--out-dir", required=True)
    sp1.add_argument("--prefix", required=True)
    sp1.add_argument("--port", type=int, default=18156)
    sp1.add_argument("--context", type=int, default=131072)
    sp1.add_argument("--concurrency", type=int, default=8)
    sp1.add_argument("--extra-args", default="")
    sp1.add_argument("--windows", default="c1,prefill4k,c8")
    sp1.add_argument("--env", action="append", default=[])
    sp1.add_argument("--nsys-bin", default="")

    sp2 = sub.add_parser("ncu")
    sp2.add_argument("--completion-bin", required=True)
    sp2.add_argument("--model-path", required=True)
    sp2.add_argument("--out-dir", required=True)
    sp2.add_argument("--prefix", required=True)
    sp2.add_argument("--extra-args", default="")
    sp2.add_argument("--launch-skip", type=int, default=400)
    sp2.add_argument("--launch-count", type=int, default=80)
    sp2.add_argument("--ncu-bin", default="")
    sp2.add_argument("--sudo-pass", default="nvidia")

    args = p.parse_args()
    if args.cmd == "nsys":
        run_nsys(args)
    else:
        run_ncu(args)


if __name__ == "__main__":
    main()
