#!/usr/bin/env python3
"""R9 Patch: Protocol §19.5 Zero-GPU-Overhead C++ Runaway Repetition Guard in llama-server.

Detects pathological single-token streaks (>= 64 repeats) and multi-token periodic loops
(period 2..64 tokens) directly in server_context::process_token() on the CPU hot path:
  - Zero GPU kernel overhead, <500 ns CPU check per token
  - Exempts whitespace, Markdown tables, dividers, and decorative separator characters
  - Automatically bypassed when request sets ignore_eos=true or env LLAMA_DISABLE_REPETITION_GUARD=1
  - Terminates cleanly with STOP_TYPE_WORD (OpenAI finish_reason="stop") instead of burning
    tens of thousands of tokens until max_tokens or context exhaustion.
"""
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <llama.cpp-dir>")
        return 2
    root = pathlib.Path(sys.argv[1])
    path = root / "tools/server/server-context.cpp"
    text = path.read_text(encoding="utf-8")

    marker = "LLAMA_DISABLE_REPETITION_GUARD"
    if marker in text:
        print(f"[R9] already patched: {path}")
        return 0

    # 1. Add per-slot rolling token buffer in struct server_slot
    slot_anchor = "    llama_tokens generated_tokens;\n"
    if slot_anchor not in text:
        raise RuntimeError(f"slot generated_tokens anchor not found in {path}")
    slot_repl = (
        slot_anchor
        + "    // R9 (§19.5): Runaway repetition guard state (zero GPU overhead)\n"
        + "    llama_tokens rep_guard_tokens;\n"
        + "    llama_token  rep_guard_last_tok = LLAMA_TOKEN_NULL;\n"
        + "    size_t       rep_guard_same_run = 0;\n"
    )
    text = text.replace(slot_anchor, slot_repl, 1)

    # 2. Reset state in server_slot::reset()
    reset_anchor = "        n_sent_text    = 0;\n"
    if reset_anchor not in text:
        raise RuntimeError(f"slot reset anchor not found in {path}")
    reset_repl = (
        reset_anchor
        + "        rep_guard_tokens.clear();\n"
        + "        rep_guard_last_tok = LLAMA_TOKEN_NULL;\n"
        + "        rep_guard_same_run = 0;\n"
    )
    text = text.replace(reset_anchor, reset_repl, 1)

    # 3. Runaway repetition check in process_token() right before EOG check
    eog_anchor = """        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\\n");
        }"""
    if eog_anchor not in text:
        raise RuntimeError(f"EOG anchor not found in {path}")

    guard_code = """        // R9 (§19.5): Zero-GPU-overhead runaway repetition guard
        static const bool rep_guard_disabled = []() {
            const char * env = getenv("LLAMA_DISABLE_REPETITION_GUARD");
            return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0 || strcmp(env, "TRUE") == 0);
        }();
        if (!rep_guard_disabled && slot.has_next_token && !slot.task->params.sampling.ignore_eos && !llama_vocab_is_eog(vocab, result.tok)) {
            auto is_exempt_span = [](const std::string & s) -> bool {
                if (s.empty()) {
                    return true;
                }
                for (unsigned char c : s) {
                    const bool exempt = (c == ' ' || c == '\\t' || c == '\\n' || c == '\\r' ||
                                         c == '|' || c == '-' || c == '=' || c == '_' ||
                                         c == '*' || c == '#' || c == '.' || c == ':' ||
                                         c == '+' || c == '~' || c == '`' || c == ',');
                    if (!exempt) {
                        return false;
                    }
                }
                return true;
            };

            slot.rep_guard_tokens.push_back(result.tok);
            if (slot.rep_guard_tokens.size() > 1024) {
                slot.rep_guard_tokens.erase(slot.rep_guard_tokens.begin(), slot.rep_guard_tokens.begin() + 512);
            }

            if (result.tok == slot.rep_guard_last_tok) {
                slot.rep_guard_same_run++;
            } else {
                slot.rep_guard_last_tok = result.tok;
                slot.rep_guard_same_run = 1;
            }

            int detected_period = 0;
            int detected_reps   = 0;

            if (slot.rep_guard_same_run >= 64 && !is_exempt_span(token_str)) {
                detected_period = 1;
                detected_reps   = (int) slot.rep_guard_same_run;
            } else {
                const size_t n = slot.rep_guard_tokens.size();
                // Check periodic token loops at tail (only integer comparisons on hot path)
                if (n >= 96 && (n % 2 == 0)) {
                    const llama_token * data = slot.rep_guard_tokens.data();
                    for (int p = 2; p <= 64; ++p) {
                        const int min_reps = (p <= 12) ? std::max(8, (96 + p - 1) / p)
                                                       : std::max(4, (128 + p - 1) / p);
                        const size_t span = (size_t) p * (size_t) min_reps;
                        if (span > n) {
                            continue;
                        }
                        const size_t base = n - span;
                        bool match = true;
                        for (size_t k = (size_t) p; k < span; ++k) {
                            if (data[base + k] != data[base + (k % (size_t) p)]) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            // Verify the repeated period is not purely whitespace/table/divider syntax
                            std::string period_text;
                            period_text.reserve((size_t) p * 4);
                            for (int j = 0; j < p; ++j) {
                                period_text += common_token_to_piece(ctx_tgt, data[n - (size_t) p + (size_t) j], false);
                            }
                            if (!is_exempt_span(period_text)) {
                                detected_period = p;
                                detected_reps   = min_reps;
                                break;
                            }
                        }
                    }
                }
            }

            if (detected_period > 0) {
                slot.stop           = STOP_TYPE_WORD;
                slot.has_next_token = false;
                SLT_WRN(slot, "stopped by runaway repetition guard (period=%d, reps=%d, n_gen=%d)\\n",
                        detected_period, detected_reps, (int) slot.stats.n_gen);
            }
        }

""" + eog_anchor

    text = text.replace(eog_anchor, guard_code, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[R9] patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
