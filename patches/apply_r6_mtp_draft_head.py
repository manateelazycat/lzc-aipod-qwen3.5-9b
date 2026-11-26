#!/usr/bin/env python3
"""R6 Patch: Lossless Draft-Only Quantized LM Head for Qwen3.5 MTP Speculative Decoding.

Nsight Systems (R1-R5) revealed that on DavidAU's Qwen3.5-9B GGUF, output.weight (lm_head,
[4096, 248320]) is stored in 16-bit BF16 (2,034 MiB). During MTP speculative decoding:
  - Target verify runs mul_mat_f<BF16> once per step (~8.0 ms)
  - MTP draft runs mul_mat_vec_f<BF16> K times per step (~7.7 ms * K!)
Because src/models/qwen35.cpp already selects `layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output`
for the MTP draft head while keeping `model.output` for target verification, we can populate
`layers[n_layer].nextn.shared_head_head` at model load time with an in-VRAM quantized shadow copy
of `model.output` controlled by `LLAMA_MTP_DRAFT_HEAD_TYPE` (q4_0 | q4_k | q5_0 | q6_k | q8_0 |
mxfp4 | nvfp4 | f8_e4m3 | off).

Target verification STILL uses exact original BF16 `model.output`, so the output distribution
under speculative rejection sampling is 100% mathematically lossless!
"""
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <llama.cpp-dir>")
        return 2
    root = pathlib.Path(sys.argv[1])
    path = root / "src/llama-model.cpp"
    text = path.read_text(encoding="utf-8")

    marker = "LLAMA_MTP_DRAFT_HEAD_TYPE"
    if marker in text:
        print(f"[R6] already patched: {path}")
        return 0

    if "#include <thread>" not in text:
        inc_anchor = "#include <stdexcept>\n"
        if inc_anchor not in text:
            raise RuntimeError(f"Include anchor not found in {path}")
        text = text.replace(inc_anchor, inc_anchor + "#include <thread>\n", 1)

    target = "    load_tensors_hadamard();\n\n    return true;"
    if target not in text:
        raise RuntimeError(f"load_tensors_hadamard anchor not found in {path}")

    replacement = """    load_tensors_hadamard();

    // R6: Optional lossless draft-only quantized shadow copy of output.weight for Qwen3.5 MTP.
    // Target verification continues to use exact original model.output (BF16/F16/F32).
    if (const char * env_mtp_head = getenv("LLAMA_MTP_DRAFT_HEAD_TYPE")) {
        std::string mode(env_mtp_head);
        for (char & c : mode) {
            c = (char) tolower((unsigned char) c);
        }
        ggml_type draft_type = GGML_TYPE_COUNT;
        if      (mode == "q4_0")    draft_type = GGML_TYPE_Q4_0;
        else if (mode == "q4_k")    draft_type = GGML_TYPE_Q4_K;
        else if (mode == "q5_0")    draft_type = GGML_TYPE_Q5_0;
        else if (mode == "q6_k")    draft_type = GGML_TYPE_Q6_K;
        else if (mode == "q8_0")    draft_type = GGML_TYPE_Q8_0;
        else if (mode == "mxfp4")   draft_type = GGML_TYPE_MXFP4;
        else if (mode == "nvfp4")   draft_type = GGML_TYPE_NVFP4;
        else if (mode == "f8_e4m3") draft_type = GGML_TYPE_F8_E4M3;
        else if (mode == "off" || mode == "none" || mode.empty()) draft_type = GGML_TYPE_COUNT;
        else {
            LLAMA_LOG_WARN("%s: ignoring unsupported LLAMA_MTP_DRAFT_HEAD_TYPE='%s'\\n", __func__, env_mtp_head);
        }

        const size_t mtp_il = (size_t) hparams.n_layer();
        const bool eligible_arch = (arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE);
        const bool eligible_src = output != nullptr && output->buffer != nullptr &&
            (output->type == GGML_TYPE_BF16 || output->type == GGML_TYPE_F16 || output->type == GGML_TYPE_F32);
        if (draft_type != GGML_TYPE_COUNT && eligible_arch &&
            eligible_src && layers.size() > mtp_il &&
            layers[mtp_il].nextn.eh_proj != nullptr &&
            layers[mtp_il].nextn.shared_head_head == nullptr) {
            const int64_t t_start = ggml_time_us();
            const int64_t ne0 = output->ne[0];
            const int64_t ne1 = output->ne[1];
            const ggml_type_traits * src_traits = ggml_get_type_traits(output->type);
            if (src_traits && (output->type == GGML_TYPE_F32 || src_traits->to_float != nullptr)) {
                ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(output->buffer);
                ggml_init_params init_p = {
                    /*.mem_size   =*/ ggml_tensor_overhead(),
                    /*.mem_buffer =*/ NULL,
                    /*.no_alloc   =*/ true,
                };
                ggml_context_ptr draft_ctx { ggml_init(init_p) };
                ggml_tensor * draft_head = draft_ctx ? ggml_new_tensor_2d(draft_ctx.get(), draft_type, ne0, ne1) : nullptr;
                if (draft_head) {
                    ggml_set_name(draft_head, "mtp.draft_head.weight");
                    ggml_backend_buffer_ptr draft_buf { ggml_backend_alloc_ctx_tensors_from_buft(draft_ctx.get(), buft) };
                    if (draft_buf) {
                        ggml_backend_buffer_set_usage(draft_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                        ggml_quantize_init(draft_type);

                        const size_t src_row_bytes = output->nb[1];
                        const size_t dst_row_bytes = draft_head->nb[1];
                        const int64_t strip_rows_max = 4096;
                        const int n_threads = std::min<int>(12, std::max<int>(1, (int) std::thread::hardware_concurrency()));

                        std::vector<uint8_t> raw_strip((size_t) strip_rows_max * src_row_bytes);
                        std::vector<float>   f32_strip((size_t) strip_rows_max * (size_t) ne0);
                        std::vector<uint8_t> dst_strip((size_t) strip_rows_max * dst_row_bytes);

                        for (int64_t row0 = 0; row0 < ne1; row0 += strip_rows_max) {
                            const int64_t nrows = std::min<int64_t>(strip_rows_max, ne1 - row0);
                            ggml_backend_tensor_get(output, raw_strip.data(), (size_t) row0 * src_row_bytes, (size_t) nrows * src_row_bytes);

                            std::vector<std::thread> workers;
                            workers.reserve(n_threads);
                            const int64_t rows_per_thr = (nrows + n_threads - 1) / n_threads;
                            for (int t = 0; t < n_threads; ++t) {
                                const int64_t r_beg = t * rows_per_thr;
                                const int64_t r_end = std::min<int64_t>(nrows, r_beg + rows_per_thr);
                                if (r_beg >= r_end) {
                                    break;
                                }
                                workers.emplace_back([&, r_beg, r_end]() {
                                    const int64_t rn = r_end - r_beg;
                                    float * f32_ptr = f32_strip.data() + (size_t) r_beg * (size_t) ne0;
                                    if (output->type == GGML_TYPE_F32) {
                                        memcpy(f32_ptr, raw_strip.data() + (size_t) r_beg * src_row_bytes, (size_t) rn * (size_t) ne0 * sizeof(float));
                                    } else {
                                        src_traits->to_float(raw_strip.data() + (size_t) r_beg * src_row_bytes, f32_ptr, rn * ne0);
                                    }
                                    ggml_quantize_chunk(draft_type, f32_strip.data(), dst_strip.data(), r_beg * ne0, rn, ne0, nullptr);
                                });
                            }
                            for (auto & w : workers) {
                                w.join();
                            }

                            ggml_backend_tensor_set(draft_head, dst_strip.data(), (size_t) row0 * dst_row_bytes, (size_t) nrows * dst_row_bytes);
                        }

                        layers[mtp_il].nextn.shared_head_head   = draft_head;
                        layers[mtp_il].nextn.shared_head_head_s = nullptr;

                        std::vector<ggml_backend_buffer_ptr> bufs;
                        bufs.emplace_back(std::move(draft_buf));
                        pimpl->ctxs_bufs.emplace_back(std::move(draft_ctx), std::move(bufs));

                        const double dt_ms = (ggml_time_us() - t_start) / 1000.0;
                        LLAMA_LOG_INFO("%s: [R6] created lossless MTP draft head %s -> %s (%.1f MiB -> %.1f MiB) in %.1f ms (%d threads)\\n",
                            __func__, ggml_type_name(output->type), ggml_type_name(draft_type),
                            ggml_nbytes(output) / (1024.0 * 1024.0),
                            ggml_nbytes(draft_head) / (1024.0 * 1024.0),
                            dt_ms, n_threads);
                    } else {
                        LLAMA_LOG_WARN("%s: [R6] failed to allocate MTP draft head buffer on %s\\n", __func__, ggml_backend_buft_name(buft));
                    }
                }
            }
        }
    }

    return true;"""

    text = text.replace(target, replacement, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[R6] patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
