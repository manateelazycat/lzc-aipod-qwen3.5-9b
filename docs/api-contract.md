# API Contract — `cloud.lazycat.aipod.qwen35-9b-defiant-fable`

## Service Endpoints

| Field | Value |
| :--- | :--- |
| Package ID | `cloud.lazycat.aipod.qwen35-9b-defiant-fable` |
| Public Model ID (`--alias`) | `qwen3.5-9b-defiant-fable` |
| Inference Engine | `llama.cpp` (`llama-server`, Thor `sm_110a` optimized R1–R9) |
| Listen Address | `0.0.0.0:18156` (host network) |
| Health Check | `GET /health`, `GET /v1/models` (verified via `/runtime/healthcheck`) |
| Prometheus Metrics (§17) | `GET /metrics` (`llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`, `llamacpp:requests_processing`, etc.) |
| OpenAI-Compatible API | `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models` |

## Capabilities & Default Parameters

- **Multimodal Vision**: Enabled via mounted `mmproj-F16.gguf` (`image_url` data URI / base64 in OpenAI `messages[].content`).
- **Tool Calling**: Enabled (`--jinja`), supports OpenAI `tools` / `tool_choice` structured function calling.
- **Reasoning / Thinking Mode**: Enabled (`--reasoning-format deepseek`).
  - Pass `"chat_template_kwargs": {"enable_thinking": false}` for fast non-thinking responses.
  - Pass `"chat_template_kwargs": {"enable_thinking": true}` (or omit) for thinking mode with `reasoning_content` separation.
- **Context Window & Concurrency**:
  - Editable Context Length: `16,384` – `262,144` tokens (step `1,024`, default `131,072`).
  - Concurrency (`-np`): fixed `8` slots with `--kv-unified` and `-ctk q8_0 -ctv q8_0`.
- **Protocol §19.1 Default Non-Greedy Sampling**:
  - Server defaults: `--temp 0.7 --top-p 0.85 --top-k 20`. Callers may override per request.
- **Protocol §19.5 Zero-GPU-Overhead Runaway Repetition Guard**:
  - Automatically terminates pathological single-token streaks (`≥64`) and periodic token loops (period `2..64` tokens) with `finish_reason: "stop"`.
  - Exempts whitespace, Markdown tables, dividers, and decorative separator characters.
  - Automatically bypassed when the request sets `"ignore_eos": true` or when container env `LLAMA_DISABLE_REPETITION_GUARD=1`.
- **Lossless Native MTP Speculative Decoding**:
  - Enabled via `--spec-type draft-mtp --spec-draft-n-max 2` with `LLAMA_MTP_DRAFT_HEAD_TYPE=mxfp4`.
  - Target verification executes against the exact original 16-bit BF16 `output.weight` tensor; output distribution is mathematically lossless.

## Example Requests

### Fast Chat
```bash
curl -s http://127.0.0.1:18156/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b-defiant-fable",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### Prometheus Metrics Check (§17)
```bash
curl -s http://127.0.0.1:18156/metrics | grep -E "llamacpp:(prompt_tokens_total|tokens_predicted_total|requests_processing)"
```
