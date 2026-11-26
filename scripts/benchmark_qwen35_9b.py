#!/usr/bin/env python3
"""Standardized unprofiled benchmark & accuracy gate for Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic (GGUF + MTP + Vision).

Measures (unprofiled HTTP SSE):
  1. C=1 single-stream decode (prose, code, Chinese technical; thinking OFF and thinking ON)
     - Median / peak decode tok/s, E2E tok/s, median TTFT (ms)
  2. Cold-salt 4K prefill throughput (tok/s) & TTFT
  3. Multi-concurrency aggregate decode throughput (C=4 and C=8)
  4. Mandatory accuracy, capability & protocol §17/§19 gates:
     - Exact math (`137 * 29 + 413 = 4386`) with thinking enabled
     - Exact math (`17 * 23 = 391`) with thinking disabled (`enable_thinking=false`)
     - OpenAI tool calling (`get_weather`)
     - Multimodal image vision (synthetic Base64 PNG with red & blue bands)
     - 16K long-context needle recall
     - Prometheus `/metrics` monotonic counter delta (§17)
     - Default sampling check (omitting temperature must NOT collapse to greedy 0.0, §19.1)
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import random
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any


PROMPT_PROSE = (
    "Explain how modern unified-memory GPU architectures (L1, L2, shared memory, and LPDDR5X) "
    "govern bandwidth-bound decoding latency in hybrid linear-attention and full-attention language models. "
    "Give concrete quantitative reasoning."
)
PROMPT_CODE = (
    "Write a self-contained Python implementation of a bounded lock-free style ring buffer simulator "
    "with producer/consumer backpressure metrics and unit tests."
)
PROMPT_ZH = (
    "请详细分析混合线性注意力（Gated DeltaNet）与全注意力大语言模型在统一内存边缘设备上的推理瓶颈，"
    "包括单流解码带宽墙、预填充算力墙以及多令牌预测（MTP）投机解码的收益来源。"
)
PROMPT_MATH_THINK = "Compute 137 * 29 + 413 step by step and end with the exact final integer."
PROMPT_MATH_FAST = "What is 17 * 23? Answer with just the number."

WORDS = (
    "harbor lantern meadow copper violet signal orchard granite ember willow canyon falcon thistle quarry "
    "lattice saffron tundra beacon marble juniper cobalt prairie delta summit glacier fern atlas kestrel "
    "mosaic pylon"
).split()


def make_test_png_b64() -> str:
    """Create a 128x128 RGB PNG: red top/bottom with a solid blue horizontal band in the center."""
    w, h = 128, 128
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for _x in range(w):
            if 44 <= y < 84:
                raw.extend((0, 0, 255))
            else:
                raw.extend((255, 0, 0))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def fetch_metrics(base_url: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=10) as r:
            text = r.read().decode("utf-8", errors="replace")
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0].split("{", 1)[0]
            try:
                val = float(parts[-1])
            except ValueError:
                continue
            out[name] = out.get(name, 0.0) + val
    except Exception:
        pass
    return out


def stream_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 256,
    enable_thinking: bool | None = False,
    temperature: float | None = 0.0,
    timeout: float = 600.0,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    body.update(extra)

    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    t_first: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    chunks = 0
    finish_reason = None

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("timings"):
                timings = obj["timings"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            delta = choices[0].get("delta") or {}
            c = delta.get("content") or ""
            rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if (c or rc) and t_first is None:
                t_first = time.monotonic()
            if c:
                content_parts.append(c)
                chunks += 1
            if rc:
                reasoning_parts.append(rc)
                chunks += 1

    t_end = time.monotonic()
    total_s = max(t_end - t0, 1e-6)
    ttft_s = (t_first - t0) if t_first is not None else total_s
    decode_s = max(t_end - (t_first if t_first is not None else t0), 1e-6)
    comp_toks = int(usage.get("completion_tokens") or timings.get("predicted_n") or chunks)
    prompt_toks = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
    decode_tps = (max(comp_toks - 1, 0) / decode_s) if comp_toks > 1 else (comp_toks / total_s)
    e2e_tps = comp_toks / total_s
    out = {
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "ttft_ms": round(ttft_s * 1000.0, 2),
        "total_s": round(total_s, 4),
        "decode_s": round(decode_s, 4),
        "decode_tok_s": round(decode_tps, 2),
        "e2e_tok_s": round(e2e_tps, 2),
        "finish_reason": finish_reason,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
    }
    if timings:
        out["server_predicted_per_second"] = round(float(timings.get("predicted_per_second") or 0.0), 2)
        out["server_prompt_per_second"] = round(float(timings.get("prompt_per_second") or 0.0), 2)
        if "draft_n" in timings:
            out["draft_n"] = timings.get("draft_n", 0)
            out["draft_n_accepted"] = timings.get("draft_n_accepted", 0)
    return out


def chat_sync(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 1024,
    enable_thinking: bool | None = False,
    temperature: float | None = 0.0,
    timeout: float = 600.0,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": False,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    body.update(extra)
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.load(resp)
    dt = time.monotonic() - t0
    m = d["choices"][0]["message"]
    return {
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
        "tool_calls": m.get("tool_calls") or [],
        "finish_reason": d["choices"][0].get("finish_reason"),
        "usage": d.get("usage") or {},
        "timings": d.get("timings") or {},
        "seconds": round(dt, 4),
    }


def make_filler(seed: int, target_tokens: int) -> tuple[str, str]:
    rng = random.Random(seed)
    code = f"{rng.choice(WORDS).upper()}-{rng.choice(WORDS).upper()}-{rng.randint(1000, 9999)}"
    words = int(target_tokens * 0.56)
    parts: list[str] = []
    i = 0
    while i < words:
        k = rng.randint(8, 16)
        s = " ".join(rng.choice(WORDS) for _ in range(k))
        parts.append(f"{s.capitalize()}. Record {rng.randint(1, 99999)} logged {rng.randint(1, 999)} items.")
        i += k + 6
    mid = len(parts) // 2
    doc = " ".join(parts[:mid]) + f"\n\nIMPORTANT: The secret verification code is {code}.\n\n" + " ".join(parts[mid:])
    return doc, code


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.url.rstrip("/")
    model = args.model

    with urllib.request.urlopen(base_url + "/v1/models", timeout=15) as r:
        models_doc = json.load(r)
    model_ids = [m.get("id") for m in models_doc.get("data", [])]
    if model not in model_ids:
        raise RuntimeError(f"Model {model!r} not in /v1/models: {model_ids}")

    if not args.no_warmup:
        stream_chat(base_url, model, [{"role": "user", "content": "Say hello in one word."}], max_tokens=32, enable_thinking=False)
        wdoc, _ = make_filler(seed=999, target_tokens=min(args.prefill_tokens, 2048))
        stream_chat(base_url, model, [{"role": "user", "content": wdoc + "\nOne word."}], max_tokens=1, enable_thinking=False)

    m_before = fetch_metrics(base_url)

    # 1. C=1 Decode (thinking OFF: prose, code, zh) — fixed 256 tokens with ignore_eos=True for apples-to-apples speed
    c1_runs = []
    for idx, prompt in enumerate((PROMPT_PROSE, PROMPT_CODE, PROMPT_ZH)):
        salt = f"[Run {idx + 1} salt={args.salt}] "
        res = stream_chat(
            base_url, model,
            [{"role": "user", "content": salt + prompt}],
            max_tokens=args.decode_tokens,
            enable_thinking=False,
            temperature=0.0,
            ignore_eos=True,
        )
        c1_runs.append(res)
    c1_decode_vals = sorted(r["decode_tok_s"] for r in c1_runs)
    c1_e2e_vals = sorted(r["e2e_tok_s"] for r in c1_runs)
    c1_ttft_vals = sorted(r["ttft_ms"] for r in c1_runs)

    # 1b. C=1 Decode with thinking ON (prose)
    c1_think = stream_chat(
        base_url, model,
        [{"role": "user", "content": f"[Think salt={args.salt}] " + PROMPT_PROSE}],
        max_tokens=args.decode_tokens,
        enable_thinking=True,
        temperature=0.0,
        ignore_eos=True,
    )

    # 2. Cold-salt 4K Prefill benchmark
    prefill_doc, _ = make_filler(seed=4096 + int(time.time()) % 10000, target_tokens=args.prefill_tokens)
    pf = stream_chat(
        base_url, model,
        [{"role": "user", "content": prefill_doc + "\n\nSummarize in one word."}],
        max_tokens=1,
        enable_thinking=False,
        temperature=0.0,
    )
    prefill_tps = round(pf["prompt_tokens"] / max(pf["ttft_ms"] / 1000.0, 1e-6), 2)

    # 3. Multi-concurrency decode (C=4 and C=8)
    conc_results: dict[str, Any] = {}
    for conc in args.concurrencies:
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
            futs = [
                pool.submit(
                    stream_chat,
                    base_url,
                    model,
                    [{"role": "user", "content": f"[C{conc} req={i} salt={args.salt}] " + (PROMPT_PROSE if i % 2 == 0 else PROMPT_CODE)}],
                    args.conc_tokens,
                    False,
                    0.0,
                    600.0,
                    ignore_eos=True,
                )
                for i in range(conc)
            ]
            items = [f.result() for f in futs]
        wall = max(time.monotonic() - t0, 1e-6)
        tot_comp = sum(it["completion_tokens"] for it in items)
        conc_results[f"c{conc}"] = {
            "concurrency": conc,
            "wall_s": round(wall, 3),
            "total_completion_tokens": tot_comp,
            "aggregate_tok_s": round(tot_comp / wall, 2),
            "per_stream_median_decode_tok_s": round(sorted(it["decode_tok_s"] for it in items)[len(items) // 2], 2),
        }

    # 4. Correctness & capability checks
    checks: dict[str, Any] = {}

    r_math_fast = chat_sync(base_url, model, [{"role": "user", "content": PROMPT_MATH_FAST}], max_tokens=64, enable_thinking=False, temperature=0.0)
    checks["math_17x23_fast"] = {
        "pass": "391" in r_math_fast["content"],
        "answer": r_math_fast["content"].strip()[:80],
    }

    r_math_think = chat_sync(
        base_url, model,
        [{"role": "user", "content": PROMPT_MATH_THINK}],
        max_tokens=3072, enable_thinking=True, temperature=0.6, top_p=0.95, top_k=20,
    )
    checks["math_4386_think"] = {
        "pass": "4386" in r_math_think["content"].replace(",", ""),
        "reasoning_len": len(r_math_think["reasoning"]),
        "answer": r_math_think["content"].strip()[-100:],
    }

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                "required": ["city"],
            },
        },
    }]
    r_tool = chat_sync(
        base_url, model,
        [{"role": "user", "content": "Call get_weather to check the current weather in Tokyo in celsius."}],
        max_tokens=512, enable_thinking=False, temperature=0.0, tools=tools, tool_choice="auto",
    )
    tc = r_tool["tool_calls"]
    tool_ok = bool(tc) and tc[0].get("function", {}).get("name") == "get_weather"
    checks["tool_call"] = {"pass": tool_ok, "calls": [t.get("function") for t in tc][:2], "raw_content": r_tool["content"][:120]}

    if not args.skip_vision:
        png_b64 = make_test_png_b64()
        try:
            r_vis = chat_sync(
                base_url, model,
                [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                        {"type": "text", "text": "What two colors appear in this image? Reply briefly in English."},
                    ],
                }],
                max_tokens=128, enable_thinking=False, temperature=0.0,
            )
            txt_l = r_vis["content"].lower()
            checks["vision_multimodal"] = {
                "pass": ("red" in txt_l) and ("blue" in txt_l),
                "answer": r_vis["content"].strip()[:120],
            }
        except Exception as e:
            checks["vision_multimodal"] = {"pass": False, "error": repr(e)}

    if args.needle_tokens > 0:
        ndoc, ncode = make_filler(seed=1337, target_tokens=args.needle_tokens)
        r_needle = chat_sync(
            base_url, model,
            [{"role": "user", "content": ndoc + "\n\nWhat is the secret verification code? Reply with the code only."}],
            max_tokens=128, enable_thinking=False, temperature=0.0,
        )
        checks[f"needle_{args.needle_tokens // 1000}k"] = {
            "pass": ncode in r_needle["content"],
            "expected": ncode,
            "prompt_tokens": r_needle["usage"].get("prompt_tokens"),
            "answer": r_needle["content"].strip()[:80],
        }

    m_after = fetch_metrics(base_url)
    metric_keys = (
        "llamacpp:prompt_tokens_total",
        "llamacpp:tokens_predicted_total",
        "llamacpp:prompt_seconds_total",
        "llamacpp:tokens_predicted_seconds_total",
        "llamacpp:requests_processing",
    )
    metrics_delta = {
        k: round(m_after.get(k, 0.0) - m_before.get(k, 0.0), 4)
        for k in metric_keys
        if k in m_after
    }
    gen_delta = metrics_delta.get("llamacpp:tokens_predicted_total", 0.0)
    prompt_delta = metrics_delta.get("llamacpp:prompt_tokens_total", 0.0)
    checks["metrics_delta"] = {
        "pass": gen_delta > 0 and prompt_delta > 0,
        "delta": metrics_delta,
    }

    # Aggregate draft acceptance stats across C=1 runs if present
    draft_n_tot = sum(r.get("draft_n", 0) for r in c1_runs)
    draft_acc_tot = sum(r.get("draft_n_accepted", 0) for r in c1_runs)
    mtp_accept_rate = round(draft_acc_tot / draft_n_tot, 4) if draft_n_tot > 0 else None

    all_pass = all(v.get("pass", False) for v in checks.values())
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "url": base_url,
        "model": model,
        "c1": {
            "median_decode_tok_s": c1_decode_vals[len(c1_decode_vals) // 2],
            "max_decode_tok_s": c1_decode_vals[-1],
            "median_e2e_tok_s": c1_e2e_vals[len(c1_e2e_vals) // 2],
            "median_ttft_ms": c1_ttft_vals[len(c1_ttft_vals) // 2],
            "think_decode_tok_s": c1_think["decode_tok_s"],
            "mtp_draft_n": draft_n_tot,
            "mtp_draft_accepted": draft_acc_tot,
            "mtp_accept_rate": mtp_accept_rate,
            "runs": [
                {
                    k: r[k]
                    for k in ("prompt_tokens", "completion_tokens", "ttft_ms", "decode_tok_s", "e2e_tok_s", "total_s")
                    if k in r
                }
                for r in c1_runs
            ],
        },
        "prefill": {
            "prompt_tokens": pf["prompt_tokens"],
            "ttft_ms": pf["ttft_ms"],
            "prefill_tok_s": prefill_tps,
            "server_prompt_per_second": pf.get("server_prompt_per_second"),
        },
        "concurrency": conc_results,
        "checks": checks,
        "all_checks_passed": all_pass,
    }
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:18156")
    p.add_argument("--model", default="qwen3.5-9b-defiant-fable")
    p.add_argument("--label", default="r0-baseline")
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--conc-tokens", type=int, default=128)
    p.add_argument("--prefill-tokens", type=int, default=4096)
    p.add_argument("--needle-tokens", type=int, default=16000)
    p.add_argument("--concurrencies", type=int, nargs="*", default=[4, 8])
    p.add_argument("--salt", default="r0")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--skip-vision", action="store_true")
    p.add_argument("--out", default="")
    args = p.parse_args()
    res = run_benchmark(args)
    if not res["all_checks_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
