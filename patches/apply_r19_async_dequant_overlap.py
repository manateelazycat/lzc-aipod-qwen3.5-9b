#!/usr/bin/env python3
"""R19 Patch: Async Double-Buffered Lookahead Weight Dequantization Overlap in 4K Prefill.

Problem diagnosed from R15/R18 nsys (prefill4k.cuda_gpu_kern_sum.csv):
  - During 4K Prefill (`M > 256`), `ggml_cuda_mul_mat_cublas_impl` executes `dequantize_block_q4_K<__half>`
    (81.8 ms, 222 calls) + `Q6_K` dequantization SYNCHRONOUSLY on `main_stream` right before each
    compute-bound `nvjet_sm110_hss` cuBLAS GEMM (510.4 ms, avg 2.3-3.1 ms/call).
  - While `nvjet_sm110_hss` saturates Blackwell Tensor Cores, the LPDDR5X UMA memory bus is mostly idle;
    while `dequantize_block_q4_K` saturates LPDDR5X bandwidth (0.36 ms), Tensor Cores are 100% idle!

Solution (R19 — 100% bitwise exact, single-file `ggml-cuda.cu` patch):
  - Add a 2-slot double-buffered async weight dequantization prefetcher on a dedicated non-blocking
    `dequant_stream` per CUDA device:
    1. In `ggml_backend_cuda_graph_compute` (non-graph, single-stream prefill path), right before
       executing a large-batch `MUL_MAT` (`ne11 > 256`), look ahead up to 12 nodes in `cgraph` for the
       next constant quantized weight `MUL_MAT` (`ne11 > 256`, contiguous `src0`) and asynchronously
       launch `dequantize_block_*<half/bf16>` on `dequant_stream` into a free double-buffer slot.
    2. Because `src0` is an immutable model weight tensor (`GGML_BACKEND_BUFFER_USAGE_WEIGHTS`),
       intermediate activations/ops between `i` and `j` cannot alias or mutate it.
    3. Inside `ggml_cuda_mul_mat_cublas_impl`, if `src0` matches a prefetched slot, `main_stream`
       waits on `slot.dequant_done`, consumes `slot.buf` directly in `cublasGemmEx`, and records
       `slot.gemm_done` on `main_stream` immediately after GEMM dispatch.
  - Toggleable via `GGML_CUDA_ASYNC_DEQUANT=0` (default 1) for crisp A/B proof.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch_ggml_cuda_cu(src_root: Path) -> None:
    p = src_root / "ggml/src/ggml-cuda/ggml-cuda.cu"
    text = p.read_text(encoding="utf-8")

    if "R19_ASYNC_DEQUANT_OVERLAP" in text:
        print(f"[R19] already patched {p}")
        return

    # 1. Insert R19 double-buffered async dequant state right before ggml_cuda_mul_mat_cublas_impl
    anchor_impl = "template<ggml_type compute_type>\nstatic void ggml_cuda_mul_mat_cublas_impl("
    r19_helper = """// R19_ASYNC_DEQUANT_OVERLAP: double-buffered async weight dequantization stream for prefill
struct r19_dequant_slot_t {
    void *       buf          = nullptr;
    size_t       cap_bytes    = 0;
    cudaEvent_t  dequant_done = nullptr;
    cudaEvent_t  gemm_done    = nullptr;
    const void * src0_ptr     = nullptr;
    int64_t      nelements    = 0;
    int          src0_type    = -1;
    int          dst_type     = -1;
    bool         valid        = false;
    bool         gemm_started = false;
};

struct r19_dequant_state_t {
    cudaStream_t       stream = nullptr;
    r19_dequant_slot_t slots[2];
    int                next_slot = 0;
    bool               inited    = false;

    void init(int device) {
        if (inited) return;
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (int s = 0; s < 2; ++s) {
            CUDA_CHECK(cudaEventCreateWithFlags(&slots[s].dequant_done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&slots[s].gemm_done, cudaEventDisableTiming));
        }
        inited = true;
    }

    void drain(cudaStream_t main_stream) {
        if (!inited) return;
        for (int s = 0; s < 2; ++s) {
            if (slots[s].valid) {
                CUDA_CHECK(cudaStreamWaitEvent(main_stream, slots[s].dequant_done, 0));
                slots[s].valid = false;
            }
        }
    }
};

static r19_dequant_state_t & r19_get_state(int device) {
    static r19_dequant_state_t states[GGML_CUDA_MAX_DEVICES];
    return states[device];
}

static bool r19_async_dequant_enabled() {
    static const bool en = []() {
        const char * v = getenv("GGML_CUDA_ASYNC_DEQUANT");
        return v == nullptr || atoi(v) != 0;
    }();
    return en;
}

static ggml_type r19_predict_cublas_type(int device, const ggml_tensor * src0, const ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[device].cc;
    const ggml_prec prec = ggml_prec_get_acc(dst);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type) || compute_type == GGML_TYPE_F8_E4M3) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
        if (ggml_cuda_info().devices[device].prefer_fp32_acc) {
            compute_type = GGML_TYPE_BF16;
        }
    }
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }
    static const int r19_mode = []() {
        const char * e = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
        if (!e) return 0;
        std::string s = e;
        for (char & c : s) c = std::tolower(c);
        if (s == "f32" || s == "fp32") return 1;
        if (s == "f16" || s == "fp16") return 2;
        if (s == "bf16") return 3;
        if (s == "quant_f16" || s == "qf16") return 4;
        return 0;
    }();
    if (r19_mode == 1) compute_type = GGML_TYPE_F32;
    else if (r19_mode == 2) compute_type = GGML_TYPE_F16;
    else if (r19_mode == 3) compute_type = GGML_TYPE_BF16;
    else if (r19_mode == 4 && (ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F8_E4M3) &&
             prec != GGML_PREC_F32 && prec != GGML_PREC_BF16 && fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F16;
    }
    return compute_type;
}

static void r19_prefetch_next_weight(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int cur_i) {
    if (!r19_async_dequant_enabled()) return;
    const ggml_tensor * cur = cgraph->nodes[cur_i];
    if (cur == nullptr || cur->op != GGML_OP_MUL_MAT || cur->src[0] == nullptr || cur->src[1] == nullptr) return;
    if (cur->src[1]->ne[1] <= 256) return;

    const int dev = ctx.device;
    auto & st = r19_get_state(dev);

    const int max_j = std::min(cur_i + 12, cgraph->n_nodes - 1);
    for (int j = cur_i + 1; j <= max_j; ++j) {
        const ggml_tensor * nxt = cgraph->nodes[j];
        if (nxt == nullptr || nxt->op != GGML_OP_MUL_MAT || nxt->src[0] == nullptr || nxt->src[1] == nullptr) continue;
        if (nxt->src[1]->ne[1] <= 256) continue;
        const ggml_tensor * s0 = nxt->src[0];
        if (!ggml_is_quantized(s0->type) || !ggml_is_contiguously_allocated(s0)) continue;
        if (s0->data == cur->src[0]->data) continue;
        const ggml_tensor * root = s0->view_src ? s0->view_src : s0;
        if (root->buffer == nullptr || ggml_backend_buffer_get_usage(root->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) continue;

        const ggml_type ctype = r19_predict_cublas_type(dev, s0, nxt);
        if (ctype != GGML_TYPE_F16 && ctype != GGML_TYPE_BF16) continue;

        const int64_t ne = ggml_nelements(s0);
        const size_t need_bytes = (size_t) ne * 2;
        if (need_bytes == 0 || need_bytes > (160ull << 20)) continue;

        st.init(dev);
        // Check if already prefetched
        for (int s = 0; s < 2; ++s) {
            if (st.slots[s].valid && st.slots[s].src0_ptr == s0->data &&
                st.slots[s].nelements == ne && st.slots[s].src0_type == (int) s0->type &&
                st.slots[s].dst_type == (int) ctype) {
                return;
            }
        }
        // Pick a free slot
        int chosen = -1;
        for (int k = 0; k < 2; ++k) {
            int s = (st.next_slot + k) & 1;
            if (!st.slots[s].valid) {
                chosen = s;
                st.next_slot = (s + 1) & 1;
                break;
            }
        }
        if (chosen < 0) return;

        auto & sl = st.slots[chosen];
        if (sl.gemm_started) {
            CUDA_CHECK(cudaStreamWaitEvent(st.stream, sl.gemm_done, 0));
            sl.gemm_started = false;
        }
        if (sl.cap_bytes < need_bytes) {
            if (sl.buf != nullptr) {
                CUDA_CHECK(cudaStreamSynchronize(st.stream));
                CUDA_CHECK(cudaFree(sl.buf));
                sl.buf = nullptr;
                sl.cap_bytes = 0;
            }
            const size_t alloc_bytes = std::max(need_bytes, (size_t) (96ull << 20));
            CUDA_CHECK(cudaMalloc(&sl.buf, alloc_bytes));
            sl.cap_bytes = alloc_bytes;
        }
        if (ctype == GGML_TYPE_F16) {
            const auto fn = ggml_get_to_fp16_cuda(s0->type);
            if (!fn) return;
            fn(s0->data, (half *) sl.buf, ne, st.stream);
        } else {
            const auto fn = ggml_get_to_bf16_cuda(s0->type);
            if (!fn) return;
            fn(s0->data, (nv_bfloat16 *) sl.buf, ne, st.stream);
        }
        CUDA_CHECK(cudaEventRecord(sl.dequant_done, st.stream));
        sl.src0_ptr  = s0->data;
        sl.nelements = ne;
        sl.src0_type = (int) s0->type;
        sl.dst_type  = (int) ctype;
        sl.valid     = true;
        return;
    }
}

""" + anchor_impl

    if anchor_impl not in text:
        raise RuntimeError(f"Could not find ggml_cuda_mul_mat_cublas_impl anchor in {p}")
    text = text.replace(anchor_impl, r19_helper, 1)

    # 2. Consume prefetched src0 in ggml_cuda_mul_mat_cublas_impl
    target_src0 = """    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }"""

    repl_src0 = """    int r19_used_slot = -1;
    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else if (ggml_is_contiguously_allocated(src0)) {
        const int64_t ne_s0 = ggml_nelements(src0);
        if (r19_async_dequant_enabled() && (compute_type == GGML_TYPE_F16 || compute_type == GGML_TYPE_BF16)) {
            auto & st = r19_get_state(ctx.device);
            if (st.inited) {
                for (int s = 0; s < 2; ++s) {
                    auto & sl = st.slots[s];
                    if (sl.valid && sl.src0_ptr == src0->data && sl.nelements == ne_s0 &&
                        sl.src0_type == (int) src0->type && sl.dst_type == (int) compute_type) {
                        CUDA_CHECK(cudaStreamWaitEvent(main_stream, sl.dequant_done, 0));
                        src0_ptr = (const cuda_t *) sl.buf;
                        r19_used_slot = s;
                        break;
                    }
                }
            }
        }
        if (r19_used_slot < 0) {
            src0_alloc.alloc(ne_s0);
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne_s0, main_stream);
            src0_ptr = src0_alloc.get();
        }
        const size_t src0_bs = ggml_blck_size(src0->type);
        s01 *= src0_bs;
        s02 *= src0_bs;
        s03 *= src0_bs;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));
        const auto convert_func = traits::convert_nc(src0->type);
        GGML_ASSERT(convert_func != nullptr);
        convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
        s01 = ne00;
        s02 = ne01*s01;
        s03 = ne02*s02;
        is_src0_cont_2 = true;
        src0_ptr = src0_alloc.get();
    }"""

    if target_src0 not in text:
        raise RuntimeError(f"Could not find src0 convert anchor in {p}")
    text = text.replace(target_src0, repl_src0, 1)

    # 3. Record gemm_done right after cuBLAS GEMM dispatch
    target_after_gemm = """    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {"""
    repl_after_gemm = """    if (r19_used_slot >= 0) {
        auto & sl = r19_get_state(ctx.device).slots[r19_used_slot];
        CUDA_CHECK(cudaEventRecord(sl.gemm_done, main_stream));
        sl.gemm_started = true;
        sl.valid = false;
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {"""
    if target_after_gemm not in text:
        raise RuntimeError(f"Could not find Convert output back to F32 anchor in {p}")
    text = text.replace(target_after_gemm, repl_after_gemm, 1)

    # 4. Trigger lookahead prefetch in graph_compute loop right before ggml_cuda_compute_forward
    target_call = """                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {"""
    repl_call = """                if (!use_cuda_graph && !is_concurrent_event_active && cuda_ctx->curr_stream_no == 0) {
                    r19_prefetch_next_weight(*cuda_ctx, cgraph, i);
                }

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {"""
    if target_call not in text:
        raise RuntimeError(f"Could not find ggml_cuda_compute_forward call anchor in {p}")
    text = text.replace(target_call, repl_call, 1)

    # 5. Drain any leftover prefetches after graph loop
    target_end = """                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }"""
    repl_end = """                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
            if (!use_cuda_graph) {
                r19_get_state(cuda_ctx->device).drain(cuda_ctx->stream());
            }
        }"""
    if target_end not in text:
        # Try with 16 spaces indentation if slightly different
        target_end_alt = """                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
                }
            }
        }"""
        if target_end_alt not in text:
            raise RuntimeError(f"Could not find graph loop end anchor in {p}")
        text = text.replace(target_end_alt, repl_end, 1)
    else:
        text = text.replace(target_end, repl_end, 1)

    p.write_text(text, encoding="utf-8")
    print(f"[R19] patched {p}")


def main() -> int:
    src_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/nvidia/qwen35-9b-work/llama.cpp")
    patch_ggml_cuda_cu(src_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
