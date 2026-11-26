#!/usr/bin/env python3
"""R8 Patch: Vectorized float4 (128-byte coalesced LDG.E.128 / STG.E.128) Gated DeltaNet decode kernel.

In ggml/src/ggml-cuda/gated_delta_net.cu, for Qwen3.5 (S_v = 128, warp_size = 32, cols_per_warp = 4,
lanes_per_col = 8, rows_per_lane = 16, KDA = false), upstream issues 16 scalar 32-bit loads/stores
per thread (`r * 8 + lane_in_col`), generating 4x more memory instructions and 32B partial-line sectors.

This patch re-indexes the 16 elements per lane into 4 contiguous float4 vectors
(`(r4 * lanes_per_col + lane_in_col) * 4`), so 8 lanes load/store a full 128-byte L2 cache line in
a single coalesced LDG.E.128 / STG.E.128 instruction across curr_state, state, q, k, and replay buffers.
The inner dot-product reduction over all S_v = 128 rows is mathematically invariant.
"""
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <llama.cpp-dir>")
        return 2
    root = pathlib.Path(sys.argv[1])
    path = root / "ggml/src/ggml-cuda/gated_delta_net.cu"
    text = path.read_text(encoding="utf-8")

    marker = "R8_GDN_FLOAT4"
    if marker in text:
        print(f"[R8] already patched: {path}")
        return 0

    old_block = """    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    if constexpr (replay) {
        const int     n_r = rep.n_rep[sequence];
        const float * pk  = rep.pk  + sequence * rep.sk3 + iq1   * rep.sk1;
        const float * pv  = rep.pv  + sequence * rep.sv3 + h_idx * rep.sv1;
        const float * pgb = rep.pgb + sequence * rep.sg3 + h_idx * rep.sg1;

        for (int t = 0; t < n_r; t++) {
            const float * k_t      = pk + t * rep.sk2;
            const float   v_col    = pv[t * rep.sv2 + col];
            const float * g_t      = pgb + t * rep.sg2;
            const float   beta_val = pgb[(rep.n_pend + t) * rep.sg2];

            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_t[r * lanes_per_col + lane_in_col];
            }

            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                g_reg[r] = expf(KDA ? g_t[r * lanes_per_col + lane_in_col] : g_t[0]);
            }

            // same update as below, without the attention output
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += (KDA ? g_reg[r] : 1.0f) * s_shard[r] * k_reg[r];
            }
            const float kv_col    = warp_reduce_sum<lanes_per_col>(kv_shard);
            const float delta_col = (v_col - (KDA ? 1.0f : g_reg[0]) * kv_col) * beta_val;

#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] = g_reg[r] * s_shard[r] + k_reg[r] * delta_col;
            }
        }

        if (rep.keep) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[col * S_v + r * lanes_per_col + lane_in_col] = s_shard[r];
            }
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (replay && !KDA) {
            if (rep.st[0] != nullptr) {
                // keep the inputs of this token pending: one CTA per k head writes k, each column its v
                const bool first = blockIdx.z == 0 && threadIdx.y == 0 && col_in_warp == 0;
                if (first && h_idx < neqk1_magic.z) {
                    float * dk = rep.st[0] + sequence * rep.st3[0] + t * rep.st2[0] + h_idx * S_v;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        dk[r * lanes_per_col + lane_in_col] = k_reg[r];
                    }
                }
                if (lane_in_col == 0) {
                    rep.st[1][sequence * rep.st3[1] + t * rep.st2[1] + h_idx * S_v + col] = v_t[col];
                }
                if (first && lane_in_col == 0) {
                    rep.st[2][sequence * rep.st3[2] + t * rep.st2[2] + h_idx] = *g_t;
                    rep.st[3][sequence * rep.st3[3] + t * rep.st2[3] + h_idx] = beta_val;
                }
            }
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // same arithmetic as the replay loop, so that a replayed state matches bit for bit
            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                g_reg[r] = expf(g_t[r * lanes_per_col + lane_in_col]);
            }

            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += g_reg[r] * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_reg[r] * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        if (replay && rep.keep) {
            return;
        }
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }"""

    if old_block not in text:
        raise RuntimeError(f"Anchor block not found in {path}")

    new_block = """    // R8_GDN_FLOAT4: Use 128-bit coalesced float4 loads/stores when !KDA and rows_per_lane % 4 == 0 (S_v=128 -> 4x float4).
    constexpr bool use_vec4 = (!KDA && (rows_per_lane % 4 == 0) && (S_v >= 64));
    constexpr int vec4_per_lane = use_vec4 ? (rows_per_lane / 4) : 1;

    float s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    auto load_vec = [&](const float * __restrict__ src, float * dst_reg) {
        if constexpr (use_vec4) {
            const float4 * src4 = reinterpret_cast<const float4 *>(src);
#pragma unroll
            for (int r4 = 0; r4 < vec4_per_lane; r4++) {
                const float4 v4 = src4[r4 * lanes_per_col + lane_in_col];
                dst_reg[r4 * 4 + 0] = v4.x;
                dst_reg[r4 * 4 + 1] = v4.y;
                dst_reg[r4 * 4 + 2] = v4.z;
                dst_reg[r4 * 4 + 3] = v4.w;
            }
        } else {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                dst_reg[r] = src[r * lanes_per_col + lane_in_col];
            }
        }
    };

    auto store_vec = [&](float * __restrict__ dst_ptr, const float * src_reg) {
        if constexpr (use_vec4) {
            float4 * dst4 = reinterpret_cast<float4 *>(dst_ptr);
#pragma unroll
            for (int r4 = 0; r4 < vec4_per_lane; r4++) {
                dst4[r4 * lanes_per_col + lane_in_col] = make_float4(
                    src_reg[r4 * 4 + 0], src_reg[r4 * 4 + 1], src_reg[r4 * 4 + 2], src_reg[r4 * 4 + 3]);
            }
        } else {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                dst_ptr[r * lanes_per_col + lane_in_col] = src_reg[r];
            }
        }
    };

    ggml_cuda_pdl_sync();
    load_vec(curr_state, s_shard);

    if constexpr (replay) {
        const int     n_r = rep.n_rep[sequence];
        const float * pk  = rep.pk  + sequence * rep.sk3 + iq1   * rep.sk1;
        const float * pv  = rep.pv  + sequence * rep.sv3 + h_idx * rep.sv1;
        const float * pgb = rep.pgb + sequence * rep.sg3 + h_idx * rep.sg1;

        for (int t = 0; t < n_r; t++) {
            const float * k_t      = pk + t * rep.sk2;
            const float   v_col    = pv[t * rep.sv2 + col];
            const float * g_t      = pgb + t * rep.sg2;
            const float   beta_val = pgb[(rep.n_pend + t) * rep.sg2];

            float k_reg[rows_per_lane];
            load_vec(k_t, k_reg);

            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                g_reg[r] = expf(KDA ? g_t[r * lanes_per_col + lane_in_col] : g_t[0]);
            }

            // same update as below, without the attention output
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += (KDA ? g_reg[r] : 1.0f) * s_shard[r] * k_reg[r];
            }
            const float kv_col    = warp_reduce_sum<lanes_per_col>(kv_shard);
            const float delta_col = (v_col - (KDA ? 1.0f : g_reg[0]) * kv_col) * beta_val;

#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] = g_reg[r] * s_shard[r] + k_reg[r] * delta_col;
            }
        }

        if (rep.keep) {
            store_vec(state + col * S_v, s_shard);
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        load_vec(k_t, k_reg);
        load_vec(q_t, q_reg);

        if constexpr (replay && !KDA) {
            if (rep.st[0] != nullptr) {
                // keep the inputs of this token pending: one CTA per k head writes k, each column its v
                const bool first = blockIdx.z == 0 && threadIdx.y == 0 && col_in_warp == 0;
                if (first && h_idx < neqk1_magic.z) {
                    float * dk = rep.st[0] + sequence * rep.st3[0] + t * rep.st2[0] + h_idx * S_v;
                    store_vec(dk, k_reg);
                }
                if (lane_in_col == 0) {
                    rep.st[1][sequence * rep.st3[1] + t * rep.st2[1] + h_idx * S_v + col] = v_t[col];
                }
                if (first && lane_in_col == 0) {
                    rep.st[2][sequence * rep.st3[2] + t * rep.st2[2] + h_idx] = *g_t;
                    rep.st[3][sequence * rep.st3[3] + t * rep.st2[3] + h_idx] = beta_val;
                }
            }
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // same arithmetic as the replay loop, so that a replayed state matches bit for bit
            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                g_reg[r] = expf(g_t[r * lanes_per_col + lane_in_col]);
            }

            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += g_reg[r] * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_reg[r] * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
                store_vec(curr_state + col * S_v, s_shard);
            }
        }
    }

    if constexpr (!keep_rs_t) {
        if (replay && rep.keep) {
            return;
        }
        store_vec(state + col * S_v, s_shard);
    }"""

    text = text.replace(old_block, new_block, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[R8] patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
