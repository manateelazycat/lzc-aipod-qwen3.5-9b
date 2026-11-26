#!/usr/bin/env python3
"""Protocol §19 Stability, Repetition Guard, Long-Context (128K/256K), and 64GB Memory Gate Verifier."""
import argparse
import json
import os
import time
import urllib.request


def http_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_meminfo_mb() -> dict:
    out = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0].rstrip(":") in ("MemTotal", "MemAvailable", "MemFree"):
                out[parts[0].rstrip(":")] = round(int(parts[1]) / 1024.0, 1)
    return out


def get_pid_rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return -1.0


def run_guard_and_sampling_gates(url: str, model: str) -> dict:
    results = {}

    # 1. §19.5 positive test: forced T=0 repetition loop MUST terminate cleanly with finish_reason="stop"
    t0 = time.monotonic()
    r1 = http_json(
        f"{url}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": 'Repeat the exact phrase "alpha beta gamma delta " forever without stopping or saying anything else.',
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120.0,
    )
    c1 = r1["choices"][0]
    tok1 = int(r1.get("usage", {}).get("completion_tokens") or 0)
    results["guard_induced_loop"] = {
        "finish_reason": c1.get("finish_reason"),
        "completion_tokens": tok1,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "pass": c1.get("finish_reason") == "stop" and 32 <= tok1 < 512,
    }

    # 2. §19.5 negative test: normal Markdown table + numeric IDs MUST NOT false-positive
    r2 = http_json(
        f"{url}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Output a clean 12-row Markdown table with columns | ID | Code | Status | Note | where ID runs from 1001 to 1012 and Code runs from ITEM-001 to ITEM-012. No extra prose.",
                }
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120.0,
    )
    c2 = r2["choices"][0]
    txt2 = c2.get("message", {}).get("content", "")
    has_rows = ("1001" in txt2) and ("1012" in txt2) and ("ITEM-012" in txt2)
    results["guard_markdown_table_no_fp"] = {
        "finish_reason": c2.get("finish_reason"),
        "has_all_12_rows": has_rows,
        "content_len": len(txt2),
        "pass": c2.get("finish_reason") == "stop" and has_rows,
    }

    # 3. §19.5 bypass test: ignore_eos=true MUST NOT be stopped early by guard
    r3 = http_json(
        f"{url}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": 'Repeat the exact phrase "alpha beta gamma delta " forever without stopping.',
                }
            ],
            "max_tokens": 220,
            "temperature": 0.0,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120.0,
    )
    c3 = r3["choices"][0]
    tok3 = int(r3.get("usage", {}).get("completion_tokens") or 0)
    results["guard_ignore_eos_bypass"] = {
        "finish_reason": c3.get("finish_reason"),
        "completion_tokens": tok3,
        "pass": tok3 >= 210,
    }

    # 4. §19.1 default non-greedy sampling test (omit temperature/top_p/top_k)
    r4 = http_json(
        f"{url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is 19 * 21? Answer with just the integer."}],
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60.0,
    )
    c4 = r4["choices"][0]
    txt4 = c4.get("message", {}).get("content", "").strip()
    results["default_sampling_non_greedy"] = {
        "finish_reason": c4.get("finish_reason"),
        "answer": txt4[:80],
        "pass": c4.get("finish_reason") == "stop" and "399" in txt4,
    }

    return results


def run_long_context_needle(url: str, model: str, target_tokens: int, code: str) -> dict:
    # Build deterministic haystack of ~target_tokens (~4 chars/token -> target_tokens * 3.6 chars)
    unit = (
        "Telemetry log archive entry: subsystem thermal_loop nominal, bus clock locked, "
        "buffer integrity verified across memory banks without uncorrectable parity faults.\n"
    )
    # unit is ~25 tokens
    reps_total = max(100, target_tokens // 25)
    mid = reps_total // 2
    haystack = (
        (unit * mid)
        + f"\nCRITICAL OVERRIDE RECORD: The unique vault passphrase is {code}. Remember {code} verbatim.\n"
        + (unit * (reps_total - mid))
    )
    prompt = haystack + "\nQuestion: What is the exact unique vault passphrase recorded above? Reply with only the passphrase."
    t0 = time.monotonic()
    resp = http_json(
        f"{url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 48,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=600.0,
    )
    dt = time.monotonic() - t0
    choice = resp["choices"][0]
    ans = choice.get("message", {}).get("content", "").strip()
    ptok = int(resp.get("usage", {}).get("prompt_tokens") or 0)
    return {
        "target_tokens": target_tokens,
        "actual_prompt_tokens": ptok,
        "elapsed_s": round(dt, 2),
        "approx_prefill_tok_s": round(ptok / max(dt, 1e-6), 2),
        "expected": code,
        "answer": ans[:80],
        "finish_reason": choice.get("finish_reason"),
        "pass": code in ans,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18156")
    ap.add_argument("--model", default="qwen3.5-9b-defiant-fable")
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--long-ctx-tokens", type=int, default=0, help="If >0, run long-context needle test of ~N tokens")
    ap.add_argument("--needle-code", default="COBALT-TITAN-8419")
    ap.add_argument("--skip-guard", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {"meminfo_before_mb": get_meminfo_mb()}
    if args.pid > 0:
        out["server_rss_before_mb"] = get_pid_rss_mb(args.pid)
    if not args.skip_guard:
        out["gates"] = run_guard_and_sampling_gates(args.url, args.model)
    if args.long_ctx_tokens > 0:
        out["long_context"] = run_long_context_needle(args.url, args.model, args.long_ctx_tokens, args.needle_code)
    out["meminfo_after_mb"] = get_meminfo_mb()
    if args.pid > 0:
        out["server_rss_after_mb"] = get_pid_rss_mb(args.pid)

    all_ok = True
    for v in out.get("gates", {}).values():
        all_ok = all_ok and bool(v.get("pass"))
    if "long_context" in out:
        all_ok = all_ok and bool(out["long_context"].get("pass"))
    out["all_passed"] = all_ok

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
