# Source, Provenance, and License Notice

## Upstream Model Weights

| Field | Details |
| :--- | :--- |
| Upstream Repository (Hugging Face) | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` |
| Pinned Commit SHA | `4d47954f2ef8277567a20508dd3f88def5404455` |
| ModelScope Bit-Identical Mirror | `manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF` |
| Base Architecture | `Qwen/Qwen3.5-9B` (`Qwen3_5ForConditionalGeneration`, 24 Gated DeltaNet + 8 Full Attention + 1 MTP layer + 27-layer ViT) |
| License | **Apache License 2.0** |

### Locked Model Files (Bit-Identical Across HF & ModelScope)

| File | Size (Bytes) | SHA256 |
| :--- | ---: | :--- |
| `Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf` | `6,979,975,392` | `d7eb4fac9389d53fa576f64a6ff53e914a00bc7705dc354d1065887565147320` |
| `mmproj-F16.gguf` | `918,166,080` | `f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f` |

## Inference Engine & Runtime Container

| Field | Details |
| :--- | :--- |
| Upstream Engine | `ggml-org/llama.cpp` @ commit `07ba74b` |
| Engine License | **MIT License** |
| Target Hardware | NVIDIA Jetson AGX Thor (`sm_110a` / CC `11.0`, CUDA `13.2`, L4T `38.4.*` / `39.*`) |
| Container Base | `FROM scratch` (25 files, `573.5 MiB` unpacked, `572.3 MiB` `.tar.gz`) |
| Weight Bundling | **None** (`containsWeights: false`; weights mounted read-only at `/model:ro`) |
| Negative Audit | Zero Python, zero compilers/headers, zero package managers, zero `.a` static archives |

## Reproducible Optimization Patches (`patches/`)

All kernel and server optimizations applied to upstream `llama.cpp` are preserved as reproducible Python patch scripts in `patches/` and bundled in both Hugging Face and ModelScope runtime repositories:
1. `patches/apply_r2_gdn_mma.py` — Define `GGML_CUDA_CC_THOR 1100` and enable Tensor Core MMA Gated DeltaNet (`gdn_single`) on Thor.
2. `patches/apply_r3_thor_uma_l2.py` — Enable LPDDR5X UMA L2 prefetch, `MMVQ_PARAMETERS_GB10`, `mmq-few` `min_ncols=2`, FlashAttention `prefer_whole_tiles`, and `should_use_fp32_acc` for Thor `CC 1100`.
3. `patches/apply_r5_mmq_cublas_thresh.py` — Add runtime-tunable `GGML_CUDA_MMQ_MAX_BATCH` crossover to dispatch large-M prefill to cuBLAS Blackwell 2CTA Tensor Cores (`nvjet_sm110_tss_*`).
4. `patches/apply_r6_mtp_draft_head.py` — Add lossless `LLAMA_MTP_DRAFT_HEAD_TYPE` in-VRAM shadow quantized draft `lm_head` (`MXFP4`, `540 MiB`) while preserving exact BF16 `output.weight` for target verification.
5. `patches/apply_r7_thor_20sm_tuning.py` — Enable `rows_per_block=2` for `MXFP4` in `MMVQ_PARAMETERS_GB10` and cache-streaming `__ldcs` loads for BF16 `lm_head` in `mul_mat_f` (reverts R7a `MXFP4` 2×-nwarps regression).
6. `patches/apply_r8_gdn_float4.py` — Vectorize scalar Gated DeltaNet decode (`S_v=128`, `!KDA`) to 128-byte coalesced `float4` (`LDG.E.128` / `STG.E.128`).
7. `patches/apply_r9_repetition_guard.py` — Implement Protocol §19.5 zero-GPU-overhead C++ runaway repetition guard in `server_context::process_token()`.
