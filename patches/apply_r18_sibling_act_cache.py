#!/usr/bin/env python3
"""R18 Patch: Cross-Sibling Activation Conversion/Quantization Cache + float4 binbcast.

Problems diagnosed from R15/R16 nsys:
  1. 4K Prefill (`r15_prefill4k`, 1,391 ms):
     - `convert_unary_cont_vec4<float, __half>` consumes 127.66 ms (9.2%, 256 calls).
     - Sibling projections (ffn_gate/up, attn_q/k/v, GDN qkv/z/ba) share the exact same
       normalized FP32 input tensor `src1`, yet `ggml_cuda_mul_mat_cublas_impl` re-converts
       the identical 64 MiB tensor 2x-3x per layer!
  2. C=1 & C=8 Decode (`r15_c1` & `r16_c8`):
     - `quantize_mmq_few_q8_1` consumes 81.69 ms (28,416 calls) in C=1 and ~196 ms (4.5%) in C=8,
       repeatedly quantizing the exact same `src1` for sibling Q4_K/Q6_K projections!
  3. 4K Prefill non-GEMM (`r15_prefill4k`):
     - `k_bin_bcast<&op_add, float>` consumes 53.09 ms (3.8%, 270 calls) doing scalar 32-bit
       loads/stores + per-element `fastmodulo` even for contiguous same-shape residual adds.

Solution (R18 — 100% bitwise exact, zero approximation):
  1. Add persistent, overlap-invalidated 1-entry activation caches to `ggml_backend_cuda_context`:
     - `r18_cublas_src1` cache for contiguous `float -> half/bf16` conversions in `ggml_cuda_mul_mat_cublas_impl`.
     - `r18_mmq_few_src1` cache for `quantize_mmq_few_q8_1` in `ggml_cuda_mul_mat_q_few`.
     - Automatically invalidated on graph start/end, on stream change, or whenever ANY writing op's
       destination memory range `[dst->data, dst->data + nbytes)` overlaps the cached `src1` range.
     - CUDA-Graph-safe: never calls `cudaMalloc`/`cudaFree` during stream capture (bypasses cache if
       buffer needs growth during capture, and invalidates on capture boundary).
     - Toggleable via `GGML_CUDA_ACT_CACHE=0` (default 1) for crisp A/B proof.
  2. Add zero-modulo 128-bit `float4` contiguous fast path `k_bin_cont_f32_vec4` in `binbcast.cu`
     (toggleable via `GGML_CUDA_BINBCAST_VEC4=0`, default 1).
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch_common_cuh(src_root: Path) -> None:
    p = src_root / "ggml/src/ggml-cuda/common.cuh"
    text = p.read_text(encoding="utf-8")
    if "R18_SIBLING_ACT_CACHE" in text:
        print(f"[R18] already patched {p}")
        return

    target = "    int curr_stream_no = 0;"
    replacement = """    int curr_stream_no = 0;

    // R18_SIBLING_ACT_CACHE: persistent 1-entry caches for sibling MUL_MAT src1 conversions/quantizations
    struct r18_cublas_src1_cache_t {
        void *       buf        = nullptr;
        size_t       cap_bytes  = 0;
        const void * src_ptr    = nullptr;
        size_t       src_nbytes = 0;
        int64_t      nelements  = 0;
        int          src_type   = -1;
        int          dst_type   = -1;
        cudaStream_t stream     = nullptr;
        bool         valid      = false;
    } r18_cublas_cache;

    struct r18_mmq_few_src1_cache_t {
        void *       buf        = nullptr;
        size_t       cap_bytes  = 0;
        const void * src_ptr    = nullptr;
        size_t       src_nbytes = 0;
        int64_t      ne10 = 0, ne11 = 0, ne12 = 0, ne13 = 0;
        size_t       nb11 = 0, nb12 = 0, nb13 = 0;
        int          ds_layout  = -1;
        int          ncg        = 0;
        cudaStream_t stream     = nullptr;
        bool         valid      = false;
    } r18_mmq_few_cache;

    void r18_invalidate_act_caches() {
        r18_cublas_cache.valid  = false;
        r18_mmq_few_cache.valid = false;
    }

    void r18_invalidate_if_overlap(const void * dst_ptr, size_t dst_nbytes) {
        if (dst_ptr == nullptr || dst_nbytes == 0) {
            return;
        }
        const uintptr_t d0 = (uintptr_t) dst_ptr;
        const uintptr_t d1 = d0 + dst_nbytes;
        if (r18_cublas_cache.valid) {
            const uintptr_t s0 = (uintptr_t) r18_cublas_cache.src_ptr;
            const uintptr_t s1 = s0 + r18_cublas_cache.src_nbytes;
            if (d0 < s1 && s0 < d1) {
                r18_cublas_cache.valid = false;
            }
        }
        if (r18_mmq_few_cache.valid) {
            const uintptr_t s0 = (uintptr_t) r18_mmq_few_cache.src_ptr;
            const uintptr_t s1 = s0 + r18_mmq_few_cache.src_nbytes;
            if (d0 < s1 && s0 < d1) {
                r18_mmq_few_cache.valid = false;
            }
        }
    }"""

    if target not in text:
        raise RuntimeError(f"Could not find curr_stream_no anchor in {p}")
    text = text.replace(target, replacement, 1)
    p.write_text(text, encoding="utf-8")
    print(f"[R18] patched {p}")


def patch_ggml_cuda_cu(src_root: Path) -> None:
    p = src_root / "ggml/src/ggml-cuda/ggml-cuda.cu"
    text = p.read_text(encoding="utf-8")
    if "R18_SIBLING_ACT_CACHE" in text:
        print(f"[R18] already patched {p}")
        return

    # 1. Free persistent cache buffers in ~ggml_backend_cuda_context()
    target_dtor = "ggml_backend_cuda_context::~ggml_backend_cuda_context() {"
    repl_dtor = """ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    // R18_SIBLING_ACT_CACHE: release persistent cache buffers
    if (r18_cublas_cache.buf != nullptr) {
        ggml_cuda_set_device(device);
        cudaFree(r18_cublas_cache.buf);
        r18_cublas_cache.buf = nullptr;
    }
    if (r18_mmq_few_cache.buf != nullptr) {
        ggml_cuda_set_device(device);
        cudaFree(r18_mmq_few_cache.buf);
        r18_mmq_few_cache.buf = nullptr;
    }"""
    if target_dtor not in text:
        raise RuntimeError(f"Could not find destructor anchor in {p}")
    text = text.replace(target_dtor, repl_dtor, 1)

    # 2. Cache contiguous src1 conversion inside ggml_cuda_mul_mat_cublas_impl
    target_src1 = """    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }"""

    repl_src1 = """    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else if (ggml_is_contiguously_allocated(src1)) {
        static const bool r18_act_cache_en = []() {
            const char * v = getenv("GGML_CUDA_ACT_CACHE");
            return v == nullptr || atoi(v) != 0;
        }();
        const int64_t ne_src1 = ggml_nelements(src1);
        const size_t need_bytes = (size_t) ne_src1 * sizeof(cuda_t);
        const size_t src1_nbytes = ggml_nbytes(src1);
        auto & cache = ctx.r18_cublas_cache;

        bool used_cache = false;
        if (r18_act_cache_en && ne_src1 > 0 && need_bytes <= (256ull << 20)) {
            cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(main_stream, &cap_status);
            const bool is_capturing = (cap_status != cudaStreamCaptureStatusNone);

            if (cache.valid &&
                cache.src_ptr == src1->data &&
                cache.src_nbytes == src1_nbytes &&
                cache.nelements == ne_src1 &&
                cache.src_type == (int) src1->type &&
                cache.dst_type == (int) compute_type &&
                cache.stream == main_stream &&
                cache.buf != nullptr &&
                cache.cap_bytes >= need_bytes) {
                src1_ptr = (const cuda_t *) cache.buf;
                used_cache = true;
            } else if (!is_capturing || (cache.buf != nullptr && cache.cap_bytes >= need_bytes)) {
                if (cache.cap_bytes < need_bytes) {
                    if (cache.buf != nullptr) {
                        CUDA_CHECK(cudaStreamSynchronize(main_stream));
                        CUDA_CHECK(cudaFree(cache.buf));
                        cache.buf = nullptr;
                        cache.cap_bytes = 0;
                    }
                    const size_t alloc_bytes = std::max(need_bytes, (size_t) (64ull << 20));
                    CUDA_CHECK(cudaMalloc(&cache.buf, alloc_bytes));
                    cache.cap_bytes = alloc_bytes;
                }
                const auto convert_func = traits::convert(src1->type);
                GGML_ASSERT(convert_func != nullptr);
                convert_func(src1->data, (cuda_t *) cache.buf, ne_src1, main_stream);
                cache.src_ptr    = src1->data;
                cache.src_nbytes = src1_nbytes;
                cache.nelements  = ne_src1;
                cache.src_type   = (int) src1->type;
                cache.dst_type   = (int) compute_type;
                cache.stream     = main_stream;
                cache.valid      = true;
                src1_ptr = (const cuda_t *) cache.buf;
                used_cache = true;
            }
        }
        if (!used_cache) {
            src1_alloc.alloc(ne_src1);
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne_src1, main_stream);
            src1_ptr = src1_alloc.get();
        }
        const size_t src1_bs = ggml_blck_size(src1->type);
        s11 *= src1_bs;
        s12 *= src1_bs;
        s13 *= src1_bs;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));
        const auto convert_func = traits::convert_nc(src1->type);
        GGML_ASSERT(convert_func != nullptr);
        convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
        s11 = ne10;
        s12 = ne11*s11;
        s13 = ne12*s12;
        is_src1_cont_2 = true;
        src1_ptr = src1_alloc.get();
    }"""

    if target_src1 not in text:
        raise RuntimeError(f"Could not find src1 convert anchor in {p}")
    text = text.replace(target_src1, repl_src1, 1)

    # 3. Invalidate caches in ggml_cuda_compute_forward whenever a non-view op writes overlapping memory
    target_fwd = """static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {"""
    repl_fwd = """static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // R18_SIBLING_ACT_CACHE: invalidate cached src1 if any writing op overlaps or modifies memory
    if (dst->op != GGML_OP_NONE && dst->op != GGML_OP_VIEW && dst->op != GGML_OP_RESHAPE &&
        dst->op != GGML_OP_PERMUTE && dst->op != GGML_OP_TRANSPOSE) {
        if (dst->op == GGML_OP_CPY && dst->src[1] != nullptr) {
            ctx.r18_invalidate_if_overlap(dst->src[1]->data, ggml_nbytes(dst->src[1]));
        }
        if (dst->op == GGML_OP_SET_ROWS || dst->op == GGML_OP_SET || dst->op == GGML_OP_GATED_DELTA_NET) {
            ctx.r18_invalidate_act_caches();
        } else {
            ctx.r18_invalidate_if_overlap(dst->data, ggml_nbytes(dst));
        }
    }
    switch (dst->op) {"""
    if target_fwd not in text:
        raise RuntimeError(f"Could not find ggml_cuda_compute_forward anchor in {p}")
    text = text.replace(target_fwd, repl_fwd, 1)

    # 4. Invalidate at every graph compute boundary AND on fused nodes (nodes_to_skip != 0)
    target_gc = "static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {"
    repl_gc = """static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ((ggml_backend_cuda_context *) backend->context)->r18_invalidate_act_caches();"""
    if target_gc not in text:
        raise RuntimeError(f"Could not find graph_compute anchor in {p}")
    text = text.replace(target_gc, repl_gc, 1)

    target_fuse = """                if (nodes_to_skip != 0) {
#ifdef GGML_CUDA_DEBUG"""
    repl_fuse = """                if (nodes_to_skip != 0) {
                    cuda_ctx->r18_invalidate_act_caches();
#ifdef GGML_CUDA_DEBUG"""
    if target_fuse not in text:
        raise RuntimeError(f"Could not find nodes_to_skip anchor in {p}")
    text = text.replace(target_fuse, repl_fuse, 1)

    p.write_text(text, encoding="utf-8")
    print(f"[R18] patched {p}")


def patch_mmq_few_cu(src_root: Path) -> None:
    p = src_root / "ggml/src/ggml-cuda/mmq-few.cu"
    text = p.read_text(encoding="utf-8")
    if "R18_SIBLING_ACT_CACHE" in text:
        print(f"[R18] already patched {p}")
        return

    target = """    } else {
        GGML_ASSERT(ne11 <= MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS);
        GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);

        src1_q.alloc(int64_t(ne12)*ne13*nunits*ncg*MMQ_FEW_Q8_1_INT4);
        quantize_mmq_few_q8_1_cuda((const float *) src1->data, src1_q.get(), mmq_few_ds_layout(src0->type), ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
            ne11, ne12, ne13, stream);
    }

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q8_0>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q3_K>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q4_K>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q5_K>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q6_K>(src0, src1, dst, src1_q.get(), pairs.get(), npairs_max, ncg, stream);
            break;"""

    replacement = """    } else {
        GGML_ASSERT(ne11 <= MMQ_FEW_MAX_NCG*MMQ_FEW_NCOLS);
        GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);
    }

    // R18_SIBLING_ACT_CACHE: reuse quantized Q8_1 src1 across sibling projections (gate/up, q/k/v, qkv/z/ba)
    const int4 * src1_q_ptr = src1_q.get();
    if (!ids) {
        static const bool r18_act_cache_en = []() {
            const char * v = getenv("GGML_CUDA_ACT_CACHE");
            return v == nullptr || atoi(v) != 0;
        }();
        const mmq_q8_1_ds_layout ds_layout = mmq_few_ds_layout(src0->type);
        const int64_t q_int4 = int64_t(ne12)*ne13*nunits*ncg*MMQ_FEW_Q8_1_INT4;
        const size_t need_bytes = (size_t) q_int4 * sizeof(int4);
        const size_t src1_nbytes = ggml_nbytes(src1);
        auto & cache = ctx.r18_mmq_few_cache;

        bool used_cache = false;
        if (r18_act_cache_en && q_int4 > 0 && need_bytes <= (64ull << 20)) {
            cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(stream, &cap_status);
            const bool is_capturing = (cap_status != cudaStreamCaptureStatusNone);

            if (cache.valid &&
                cache.src_ptr == src1->data &&
                cache.src_nbytes == src1_nbytes &&
                cache.ne10 == ne10 && cache.ne11 == ne11 && cache.ne12 == ne12 && cache.ne13 == ne13 &&
                cache.nb11 == nb11 && cache.nb12 == nb12 && cache.nb13 == nb13 &&
                cache.ds_layout == (int) ds_layout &&
                cache.ncg == ncg &&
                cache.stream == stream &&
                cache.buf != nullptr &&
                cache.cap_bytes >= need_bytes) {
                src1_q_ptr = (const int4 *) cache.buf;
                used_cache = true;
            } else if (!is_capturing || (cache.buf != nullptr && cache.cap_bytes >= need_bytes)) {
                if (cache.cap_bytes < need_bytes) {
                    if (cache.buf != nullptr) {
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        CUDA_CHECK(cudaFree(cache.buf));
                        cache.buf = nullptr;
                        cache.cap_bytes = 0;
                    }
                    const size_t alloc_bytes = std::max(need_bytes, (size_t) (4ull << 20));
                    CUDA_CHECK(cudaMalloc(&cache.buf, alloc_bytes));
                    cache.cap_bytes = alloc_bytes;
                }
                quantize_mmq_few_q8_1_cuda((const float *) src1->data, (int4 *) cache.buf, ds_layout, ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
                    ne11, ne12, ne13, stream);
                cache.src_ptr    = src1->data;
                cache.src_nbytes = src1_nbytes;
                cache.ne10 = ne10; cache.ne11 = ne11; cache.ne12 = ne12; cache.ne13 = ne13;
                cache.nb11 = nb11; cache.nb12 = nb12; cache.nb13 = nb13;
                cache.ds_layout  = (int) ds_layout;
                cache.ncg        = ncg;
                cache.stream     = stream;
                cache.valid      = true;
                src1_q_ptr = (const int4 *) cache.buf;
                used_cache = true;
            }
        }
        if (!used_cache) {
            src1_q.alloc(q_int4);
            quantize_mmq_few_q8_1_cuda((const float *) src1->data, src1_q.get(), ds_layout, ne10, nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
                ne11, ne12, ne13, stream);
            src1_q_ptr = src1_q.get();
        }
    }

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q8_0>(src0, src1, dst, src1_q_ptr, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q3_K>(src0, src1, dst, src1_q_ptr, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q4_K>(src0, src1, dst, src1_q_ptr, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q5_K>(src0, src1, dst, src1_q_ptr, pairs.get(), npairs_max, ncg, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_q_few_switch_ncg<GGML_TYPE_Q6_K>(src0, src1, dst, src1_q_ptr, pairs.get(), npairs_max, ncg, stream);
            break;"""

    if target not in text:
        raise RuntimeError(f"Could not find mmq_few quantize anchor in {p}")
    text = text.replace(target, replacement, 1)
    # Also update IQ3_S, IQ4_NL, IQ4_XS switch branches to use src1_q_ptr
    for qtype in ("GGML_TYPE_IQ3_S", "GGML_TYPE_IQ4_NL", "GGML_TYPE_IQ4_XS"):
        text = text.replace(
            f"launch_mul_mat_q_few_switch_ncg<{qtype}>(src0, src1, dst, src1_q.get(),",
            f"launch_mul_mat_q_few_switch_ncg<{qtype}>(src0, src1, dst, src1_q_ptr,",
        )

    p.write_text(text, encoding="utf-8")
    print(f"[R18] patched {p}")


def patch_binbcast_cu(src_root: Path) -> None:
    p = src_root / "ggml/src/ggml-cuda/binbcast.cu"
    text = p.read_text(encoding="utf-8")
    if "R18_BINBCAST_VEC4" in text:
        print(f"[R18] already patched {p}")
        return

    target = """template <float (*bin_op)(const float, const float), int n_fuse = 1>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            cudaStream_t stream) {
        launch_bin_bcast_pack<bin_op, src0_t, src1_t, dst_t>(
            src0, src1, dst, src0_dd, src1_dd, dst_dd, stream, std::make_index_sequence<n_fuse>{});
    }
};"""

    replacement = """// R18_BINBCAST_VEC4: zero-modulo 128-bit float4 contiguous fast path for residual add/sub/mul
template <float (*bin_op)(const float, const float)>
static __global__ void k_bin_cont_f32_vec4(
        const float * __restrict__ src0, const float * __restrict__ src1, float * __restrict__ dst,
        const int64_t k4) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= k4) {
        return;
    }
    ggml_cuda_pdl_sync();
    const float4 a = reinterpret_cast<const float4 *>(src0)[i];
    const float4 b = reinterpret_cast<const float4 *>(src1)[i];
    float4 c;
    c.x = bin_op(a.x, b.x);
    c.y = bin_op(a.y, b.y);
    c.z = bin_op(a.z, b.z);
    c.w = bin_op(a.w, b.w);
    reinterpret_cast<float4 *>(dst)[i] = c;
}

template <float (*bin_op)(const float, const float), int n_fuse = 1>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            cudaStream_t stream) {
        if constexpr (n_fuse == 1 &&
                      std::is_same<src0_t, float>::value &&
                      std::is_same<src1_t, float>::value &&
                      std::is_same<dst_t, float>::value) {
            static const bool r18_vec4_en = []() {
                const char * v = getenv("GGML_CUDA_BINBCAST_VEC4");
                return v == nullptr || atoi(v) != 0;
            }();
            if (r18_vec4_en && src0_dd != nullptr && src1_dd != nullptr && dst_dd != nullptr &&
                ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst) &&
                ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
                const int64_t ne = ggml_nelements(dst);
                if ((ne & 3) == 0 &&
                    (((uintptr_t) src0_dd & 15) == 0) &&
                    (((uintptr_t) src1_dd & 15) == 0) &&
                    (((uintptr_t) dst_dd  & 15) == 0)) {
                    const int64_t k4 = ne >> 2;
                    const int64_t num_blocks = (k4 + 256 - 1) / 256;
                    if (num_blocks > 0 && num_blocks <= INT32_MAX) {
                        const ggml_cuda_kernel_launch_params lp((dim3)num_blocks, 256, 0, stream);
                        ggml_cuda_kernel_launch(k_bin_cont_f32_vec4<bin_op>, lp, src0_dd, src1_dd, dst_dd, k4);
                        return;
                    }
                }
            }
        }
        launch_bin_bcast_pack<bin_op, src0_t, src1_t, dst_t>(
            src0, src1, dst, src0_dd, src1_dd, dst_dd, stream, std::make_index_sequence<n_fuse>{});
    }
};"""

    if target not in text:
        raise RuntimeError(f"Could not find bin_bcast_cuda anchor in {p}")
    text = text.replace(target, replacement, 1)
    p.write_text(text, encoding="utf-8")
    print(f"[R18] patched {p}")


def main() -> int:
    src_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/nvidia/qwen35-9b-work/llama.cpp")
    patch_common_cuh(src_root)
    patch_ggml_cuda_cu(src_root)
    patch_mmq_few_cu(src_root)
    patch_binbcast_cu(src_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
