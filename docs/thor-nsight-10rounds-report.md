# Qwen3.5-9B-The-Defiant-Fable Jetson AGX Thor 移植与 10 轮 Nsight 性能优化测试报告

## 1. 测试平台与模型基线

| 维度 | 实测规格 |
| :--- | :--- |
| **测试硬件** | NVIDIA Jetson AGX Thor T5000 (`192.168.1.199`, Hostname `lzc-pod-XhkQoD`, Serial `1422326011571`) |
| **GPU / 计算能力** | Blackwell `sm_110a` (CC `11.0`), 20 SMs @ `1.569 GHz`, `NV Power Mode: MAXN` |
| **统一内存 / 带宽** | 122 GiB LPDDR5X UMA (`273 GB/s` 理论峰值), 14-core Arm Neoverse V3AE |
| **系统与工具链** | L4T `R39.2.0`, Driver `595.78`, CUDA `13.2`, `nsys 2026.3.1`, `ncu 2026.1.1` |
| **上游模型仓库** | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` (Apache-2.0) |
| **上游固定提交** | `4d47954f2ef8277567a20508dd3f88def5404455` |
| **锁定主权重** | `Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf` (`6,979,975,392 B`, SHA256 `d7eb4fac...147320`) |
| **锁定视觉投影器** | `mmproj-F16.gguf` (`918,166,080 B`, SHA256 `f70dc350...6ae029f`) |
| **模型架构** | `Qwen3_5ForConditionalGeneration`（32 层 = 24 层 Gated DeltaNet + 8 层 Full Attention + 1 层内置 MTP `blk.32.nextn.*` + 27 层 ViT，`vocab=248,320`，`output.weight` 为 16-bit BF16 `2,034 MiB`） |
| **推理引擎基线** | `ggml-org/llama.cpp` @ `07ba74b`（MIT） |
| **最终交付形态** | `FROM scratch` 精简单用途 Thor 容器（25 文件、`573.5 MiB` 解压 / `572.3 MiB` `.tar.gz`，零权重、零 Python、零编译器）+ 懒猫算力舱薄壳 LPK (`cloud.lazycat.aipod.qwen35-9b-defiant-fable`) |

### 评测方法论
- **剖析与复测严格隔离**：每轮先在稳态窗口内执行 `nsys`（`--cuda-graph-trace=node`，分别采集 C=1、4K Prefill、C=8 三个窗口）并在基线与终版执行 `ncu --set speedoflight` Roofline 硬件计数器剖析；所有性能数字均在**无任何剖析器挂载**的纯净 HTTP SSE 服务下复测取得（`scripts/benchmark_qwen35_9b.py`）。
- **基准工作负载**：
  - **C=1 解码**：3 条不同题材独立提示（英文长文、Python 算法代码、中文技术解释，各 256 tokens，`ignore_eos=true`），取中位数与峰值，另单测 Thinking 思考模式吞吐与 MTP 接受率。
  - **4K 冷 Prefill**：~4,100 tokens 唯一加盐无缓存提示词，实测冷首字延迟（TTFT）与 Prefill 吞吐。
  - **多并发聚合**：C=4（512 tokens 总计）与 C=8（1,024 tokens 总计）并发 SSE 请求聚合吞吐。
  - **能力与协议门禁**：快速数学（`17*23=391`）、思考数学（`73*59+413=4386`）、OpenAI 工具调用、多模态视觉识别、16K/128K/256K 大海捞针、§17 Prometheus `/metrics` 增量对账、§19.1 默认非贪婪采样与 §19.5 失控重复守卫正/反例。

---

## 2. 端到端优化总成绩（R0 基线 vs 最终 `FROM scratch` 精简容器）

| 核心指标 | R0 基线 (`sm_110`, 无 MTP) | R10 宿主机终版 (`K=2`) | **最终 `FROM scratch` 精简容器 (`K=2`)** | 峰值配置参考 (`K=3`) | 较 R0 基线提升（精简容器） |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **C=1 解码中位数 (tok/s)** | 32.95 | 58.48 | **59.20** | **62.18** | **`+79.7%` (`1.80×`)** |
| **C=1 解码峰值 (tok/s)** | 32.96 | 66.25 | **65.89** | **73.20** | **`+99.9%` (`2.00×`)** |
| **思考模式解码 (tok/s)** | 32.95 | 62.67 | **62.92** | **70.44** | **`+91.0%` (`1.91×`)** |
| **4K 冷 Prefill (tok/s)** | 1,326.99 | 2,816.02 | **2,806.73** | 2,848.02 | **`+111.5%` (`2.12×`)** |
| **4K 冷 Prefill TTFT (ms)** | 3,079.1 | 1,463.8 | **1,476.5** | 1,440.3 | **`-52.0%`（延迟腰斩）** |
| **C=4 并发聚合 (tok/s)** | 114.17 | 140.95 | **146.76** | 134.62 | **`+28.5%`** |
| **C=8 并发聚合 (tok/s)** | 192.92 | 201.09 | **219.43** | 179.24 | **`+13.7%`** |
| **C=1 GPU 核总耗时 (nsys)** | 8,234.4 ms | 5,264.3 ms | — | — | **`-36.1%`** |
| **4K Prefill GPU 总耗时 (nsys)** | 3,120.5 ms | 1,389.7 ms | — | — | **`-55.5%` (`2.25×`)** |
| **6/6 精度与协议 + 4/4 §19 门禁** | 基线 | 128K/256K 全过 | **10/10 全过（含 `--network none`）** | 6/6 全过 | 生产达标 |

---

## 3. 10 轮“Nsight 剖析 → 性能优化 → 非剖析复测”完整演进表

| 轮次 | 剖析发现的瓶颈 | 实施优化措施（对应补丁/参数） | C=1 中位 (tok/s) | C=1 峰值 (tok/s) | 4K Prefill (tok/s) | C=4 聚合 (tok/s) | C=8 聚合 (tok/s) | 决策与核心剖析证据 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **R0** | 基线：`sm_110` 编译、无 MTP、默认 F16 KV、`-ub 512` | 建立 `nsys`（3 窗口）+ `ncu`（45 核 Roofline）基线 | 32.95 | 32.96 | 1,326.99 | 114.17 | 192.92 | C=1 中 `MMVQ` 占 `66.9%`、BF16 `lm_head` 占 `24.1%`；Prefill `MMQ` 占 `80.3%`、标量 GDN 占 `8.6%` |
| **R1** | `sm_110` 未开架构专用指令；未启用模型自带原生 MTP 预测头 | 重编 `sm_110a` + `-DGGML_CUDA_FA_ALL_QUANTS=ON`；启用 `--spec-type draft-mtp` 并扫描 `K=1..4` | **42.44** (`K=1`) | 43.28 (`K=3` 峰 51.99) | 1,217.41 | **125.93** | 189.11 | **保留 `sm_110a` + MTP**；`K=1` 接受率 `71.1%`（`+28.8%`），但 `nsys` 暴露 BF16 `lm_head` 在 Target (`7.99 ms`) + Draft (`7.67 ms × K`) 合计占 **36.9%**，吃掉 `K=2/3` 收益 |
| **R2** | `r1_prefill4k` 显示上游 `gated-delta-net-mma.cu` 仅对 GB10 启用、漏掉 Thor `CC 1100`，导致 GDN 退化为标量核（`329 ms`, `9.7%`） | [apply_r2_gdn_mma.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r2_gdn_mma.py)：定义 `GGML_CUDA_CC_THOR 1100` 并开启 Tensor Core MMA `gdn_single` | **42.90** | 44.68 | **1,293.81** (`+6.3%`) | 123.08 | **193.68** | **保留**；`r2_prefill4k` 实证 GDN 从 `329 ms` 降至 **`127.0 ms`（`2.59×` 加速）**，6/6 精度全过 |
| **R3** | `r2_c1` 暴露 4 处上游 GB10 调优遗漏 Thor：无 LPDDR5X L2 预取、`MMVQ` 未用宽块、`M=2` 退回 `mul_mat_vec_q` 未进 `mmq-few` | [apply_r3_thor_uma_l2.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r3_thor_uma_l2.py)：为 Thor 启用 L2 预取、`MMVQ_PARAMETERS_GB10`、`mmq-few` `min_ncols=2`、`fattn` 整瓦片、`fp32_acc` | **45.73** (`+6.6%`) | 47.71 | **1,298.64** | 122.71 | **194.29** | **保留**；`r3_c1` 实证 `mul_mat_q_few` 接管 `M=2` 后 Q4_K 单次耗时 `75.2 → 65.7 µs`（`-12.6%`），C=8 FlashAttention 降 `22%` |
| **R4** | 128K/256K 下 F16 KV 占用 4.3–8.6 GiB 内存及带宽 | 扫描 `-fa on -ctk/-ctv`（`f16` vs `q8_0` vs `q4_0`）+ `--cache-reuse 256` | 44.72 (`q8_0`) | 47.58 | 1,288.49 | **124.56** | **194.14** | **保留 `q8_0` KV**；KV 显存减半，C=8 FlashAttention 再降 `12.8%`（`23.26 → 20.29 ms`），16K 大海捞针无损通过 |
| **R5** | `r4_prefill4k` 显示 `mul_mat_q` 独占 **`82.9%` (`2,656 ms`)**，仅跑到 `~28 TFLOPS`，远低于 Thor Blackwell Tensor Core 算力 | [apply_r5_mmq_cublas_thresh.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r5_mmq_cublas_thresh.py)：增加 `GGML_CUDA_MMQ_MAX_BATCH=256`，大 Batch 切换 cuBLAS 2CTA Tensor Core；扫描 `-b 4096 -ub 512/1024/2048/4096` | **45.66** | 47.24 | **2,814.26** (**`2.18×`**) | **126.96** | **210.33** | **保留（重大突破）**；`r5_prefill4k` 实证 `nvjet_sm110_tss_*` 全面接管，GPU 总耗时 `3,205.8 → 1,478.2 ms`（`-53.9%`），TTFT `3,245 → 1,468 ms` |
| **R6** | `r5_c1` 显示 BF16 `lm_head` 在 Draft 阶段每步耗 `7.68 ms`，成为 MTP `K≥2` 最大墙；且 `qwen35.cpp` 原生支持独立的 `nextn.shared_head_head` | [apply_r6_mtp_draft_head.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r6_mtp_draft_head.py)：启动时在显存内为 MTP Draft 生成量化影子 `lm_head`（扫描 `q4_0`/`q4_k`/`q8_0`/`mxfp4` × `K=1..4`），Target 验证 100% 保持原版 BF16（数学无损） | **58.77** (`mxfp4 K=2`) / **60.38** (`K=3`) | **63.03** / **66.84** | **2,823.06** | **133.24** / **139.27** | **220.43** (`K=2`) | **保留 `mxfp4 K=2`（最大解码突破）**；`r6_c1` 实证 Draft `lm_head` 从 `7.68 ms` 暴降至 **`2.72 ms`（`2.82×`）**；`K=2` 在 C=8 时 batch=`8×3=24` 精准卡在 `mmq-few` 最优上限，C=8 创 `220.43 tok/s` 新高 |
| **R7** | `r6_c1` 发现 `MXFP4` Draft Head 仅跑到 `198 GB/s` (`2.72 ms`)，且 2 GB BF16 `mul_mat_f` 线性扫描污染 32 MB L2；同时需评估 `PDL` 与 `plusIQ-TOOLS-Q6_K` | **R7a** 尝试 `MXFP4` 2×-nwarps 升宽 + `mmf __ldcs` + `PDL=0/1` + `Q6_K` 对比；**R7b**（[apply_r7_thor_20sm_tuning.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r7_thor_20sm_tuning.py)）回退负优化并改为 `MXFP4 rows_per_block=2` | **57.72** (`K=2`) / **62.18** (`K=3`) | **63.53** / **66.79**（思考 **70.44**） | **2,837.58** | **137.89** | **212.82** | **保留 R7b + `PDL=1` + `Q4_K_M`**；`r7a` 抓出 2×-nwarps 回退（`2,723 → 3,634 µs`）并果断回退；`r7b` `rows_per_block=2` 将 `MXFP4` 再压 **`-15.4%`** 至 **`2,304 µs`（`235 GB/s`）**，`mmf __ldcs` 降 `104 µs/step`；`Q6_K` 慢 `20%`（`46.95 tok/s`）故锁定 `Q4_K_M` |
| **R8** | `r7b_c8` 中非 GEMM 热点 `gated_delta_net_cuda<128>` 占 `4.7%` (`190 ms`, `105.5 µs/call`)，逐元素标量 `LDG.E.32` 访存指令多 | [apply_r8_gdn_float4.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r8_gdn_float4.py)：针对 `S_v=128, !KDA` 重排为 128B 合并 `float4` (`LDG.E.128`/`STG.E.128`) | **58.79** (`K=2`) | **64.15** (`K=3` 峰 **73.20**) | **2,848.02** | 136.71 | 193.30 | **保留**；`r8_c8` 实证 `gated_delta_net_cuda` 单次降 `5.0%`（`105.5 → 100.2 µs`），C=8 GPU 总时间降 `5.1%`（`4,070 → 3,864 ms`），Prefill GPU 达 `1,392.6 ms` |
| **R9** | 协议 §19.1/§19.5 要求默认非贪婪采样与零 GPU 开销失控重复守卫；14 核 ARM 主机线程与轮询调优 | [apply_r9_repetition_guard.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r9_repetition_guard.py) + `--temp 0.7 --top-p 0.85 --top-k 20`；对比默认 / `-t 8 -tb 12 --poll 100` / `-t 4 -tb 12 --poll 100` | **59.41** (`t8_poll100`) | **66.01** | **2,916.48** | **137.41** | **201.38** | **保留 `-t 8 -tb 12 --poll 100` + §19.5 守卫**；零性能回退、零 GPU 开销；诱导循环 98 tokens (`1.37 s`) `finish_reason="stop"` 拦截，12 行 Markdown 表格零误杀 |
| **R10** | 终版收口、128K/256K 极限上下文与 64GB 内存校验、终版 `nsys` + `ncu` 硬件计数对比及 `FROM scratch` 精简容器离线验收 | 固化生产配置并执行全套门禁、128K/256K 捞针、`ncu` Roofline、`FROM scratch` 打包与 `--network none` 离线复测 | **58.48**（精简容器 **59.20**） | **66.25**（精简容器 **65.89**） | **2,816.02**（精简容器 **2,806.73**） | **140.95**（精简容器 **146.76**） | **201.09**（精简容器 **219.43**） | **全量通过**：128K (`116K` tok) & 256K (`226K` tok) 捞针 100% 召回；256K 内存增量 `25.7 GiB`（适配 64GB Thor）；精简容器 `--network none` 6 秒就绪，10/10 门禁全过 |

---

## 4. Nsight Systems (`nsys`) 与 Nsight Compute (`ncu`) 深度剖析对比（R0 vs R10）

### 4.1 `nsys` 核函数分类耗时对比

#### C=1 解码窗口（256 tokens）
| 核函数类别 | R0 基线耗时 (ms / 占比) | R10 终版耗时 (ms / 占比) | 变化说明 |
| :--- | :---: | :---: | :--- |
| **总 GPU 核函数时间** | **`8,234.40 ms`**（227,238 launches） | **`5,264.26 ms`**（183,615 launches） | **`-36.1%` GPU 耗时**（叠加 MTP `K=2` 端到端 `+79.7%`） |
| `MMQ_FEW`（小 Batch 量化 GEMM） | 40.49 ms (`0.5%`) | 2,605.56 ms (`49.5%`) | R3 启用 `min_ncols=2` 后全面接管 Target Verify，Q4_K 仅 `66.0 µs/call` |
| `MMVQ`（含 MXFP4 Draft Head） | 5,506.65 ms (`66.9%`) | 918.98 ms (`17.5%`) | R6+R7b 将 MTP Draft `lm_head` 转为 `MXFP4 rows_per_block=2`（`2.30 ms/call`） |
| `FP16/BF16_GEMM`（Target `lm_head`） | 1,981.75 ms (`24.1%`) | 1,080.01 ms (`20.5%`) | MTP 减少 Target `lm_head` 调用次数近半（`256 → 133` 次），且 R7b `__ldcs` 防 L2 污染 |
| `QUANTIZE_ACT` / `NORM` / `GET_ROWS` | 367.78 ms (`4.4%`) | 265.10 ms (`5.0%`) | 随验证步数减少同步大幅下降 |

#### 4K 冷 Prefill 窗口（~4,100 tokens）
| 核函数类别 | R0 基线耗时 (ms / 占比) | R10 终版耗时 (ms / 占比) | 变化说明 |
| :--- | :---: | :---: | :--- |
| **总 GPU 核函数时间** | **`3,120.50 ms`**（10,376 launches） | **`1,389.74 ms`**（4,216 launches） | **`-55.5%` GPU 耗时（`2.25×` 加速），启动次数减 `59.4%`** |
| `MMQ`（INT8-MMA 量化 GEMM） | 2,506.71 ms (`80.3%`) | 53.54 ms (`3.9%`) | R5 切换 cuBLAS 后几乎清零 |
| `FP16/BF16_GEMM`（Blackwell 2CTA cuBLAS） | 7.79 ms (`0.2%`) | 505.06 ms (`36.3%`) | `nvjet_sm110_tss_256x256_64x4_2x1_2cta_v_bz_TNT` 释放 Tensor Core 算力 |
| `GDN_AR`（标量 GDN）→ `GDN_MMA`（Tensor Core GDN） | 268.63 ms (`8.6%`) | **136.61 ms** (`9.8%`) + 3.79 ms | R2 启用 Thor Tensor Core MMA `gdn_single`，提速 `2.0×–2.6×` |
| 反量化与激活转换（`dequantize` + `convert_bf16`） | 80.96 ms (`2.6%`) | 207.94 ms (`15.0%`) | cuBLAS BF16 路径的必要开销，但总计算时间仍比 MMQ 快 `3.5×` |

### 4.2 `ncu` Roofline 硬件计数器对比（R0 vs R10）
| 关键核函数 | R0 实测 Duration / SM% / Mem% / L2% | R10 实测 Duration / SM% / Mem% / L2% | 硬件级改善 |
| :--- | :--- | :--- | :--- |
| `mul_mat_q_few<Q4_K, 2, 2, 0>` | `116.74 µs` / `34.6%` / `22.9%` / `22.9%` | **`97.15 µs`** / `28.1%` / **`23.2%`** / **`23.2%`** | 单核硬件耗时 **`-16.8%`**，L2 预取命中与有效吞吐提升 |
| `mul_mat_q_few<Q6_K, 2, 1, 0>` | —（R0 走 MMVQ） | **`129.92 µs`** / `42.8%` / `36.5%` / `34.1%` | 高算力密度高效运行 |
| `gated_delta_net_cuda<128>` | `54.40 µs` / `24.6%` / `30.7%` / `10.3%` | **`53.02 µs`** / **`17.8%`** / `26.1%` / `8.5%` | R8 `float4` 4× 缩减访存指令数，SM 指令发射压力显著降低 |
| `flash_attn_ext_f16<256, 256>` | — | **`34.91 µs`** / `11.3%` / `10.1%` / `9.8%` | `q8_0` KV + 整瓦片调优轻量运行 |

---

## 5. 被拒绝 / 证伪的实验记录（严格工程留痕）

1. **R7a `MXFP4` 2×-nwarps 升宽（已回退）**：在 `mmvq.cu` 将 `GGML_TYPE_MXFP4` 加入 `MMVQ_PARAMETERS_GB10` 4-warp 升宽后，`nsys` 实测 `mul_mat_vec_q<MXFP4>` 从 `2,723 µs` 回退至 `3,634 µs`（`+33.4%`），C=1 从 `58.77` 降至 `55.72 tok/s`。根因是 `lm_head` 248,320 行下 4-warp 块增大共享内存归约开销。回退并改为 R7b `rows_per_block=2` 后降至 **`2,304 µs`**（`-15.4%`）。
2. **R7a 关闭 Blackwell PDL (`GGML_CUDA_PDL=0`)（已放弃）**：对比 `PDL=1`，C=1 下降 `3.0%`（`55.72 → 54.11`），C=8 下降 `2.6%`（`212.10 → 206.76`），因此保持 `GGML_CUDA_PDL=1`。
3. **R7a 权重选型：`plusIQ-TOOLS-Q6_K` (8.76 GB) vs `Q4_K_M` (6.98 GB)**：`Q6_K` 受 273 GB/s LPDDR5X 带宽约束，C=1（`46.95` vs `58.77`）与 C=8（`175.43` vs `220.43`）均慢约 `20%`，且多占 `1.78 GB` 显存；而 `Q4_K_M` 在全部数学/工具调用/视觉/16K–256K 捞针测试中 100% 通过，因此生产锁定 `Q4_K_M`。
4. **R1/R6 `K=3/K=4` 并发权衡**：虽然 `K=3` 单路 C=1 峰值可达 `73.20 tok/s`（中位 `62.18`），但在 `C=8` 时总验证 Batch `8 × 4 = 32` 超出 `mmq-few` `max_batch=24` 上限，导致 C=8 从 `220.43` 掉到 `176.96 tok/s`；生产默认锁定 **`K=2`**（单路 `59.20 tok/s`、C=8 `219.43 tok/s` 双优）。

---

## 6. 稳定性、长上下文（128K/256K）、64GB 内存与精简容器验收回执

| 验收项目 | 实测证据与结果 | 结论 |
| :--- | :--- | :---: |
| **128K 长上下文大海捞针** | 实测 `116,070` prompt tokens，耗时 `80.18 s`（`1,447.58 tok/s`），精确输出 `TITAN-OBSIDIAN-7194` | **PASS** |
| **256K 极限长上下文大海捞针** | 实测 `226,270` prompt tokens，耗时 `240.39 s`（`941.25 tok/s`），精确输出 `AURORA-QUARTZ-5283` | **PASS** |
| **64GB Thor T4000 内存预算** | 128K 服务端 RSS `7.0–8.0 GiB`（含统一内存总增量 ~12 GiB）；256K 全满总可用内存增量 **`25.68 GiB`**，留有 >38 GiB 系统裕量 | **PASS** |
| **§19.1 默认非贪婪采样** | `--temp 0.7 --top-p 0.85 --top-k 20`；未传采样参数请求 `19*21` 正常返回 `399` | **PASS** |
| **§19.5 诱导重复精准拦截** | 强制 `T=0` 无限重复短语在 **`98` tokens (`1.37 s`)** 触发 `period=4, reps=24`，以 `finish_reason="stop"` 优雅终止 | **PASS** |
| **§19.5 正常表格防误杀** | 12 行 Markdown 表格 + `ITEM-001..012` 完整输出，零误杀 | **PASS** |
| **§19.5 `ignore_eos=true` 旁路** | 基准压测 `ignore_eos=true` 跑满 `220` tokens，不误拦截 | **PASS** |
| **§17 Prometheus `/metrics`** | `llamacpp:prompt_tokens_total`（`+21,441`）、`tokens_predicted_total`（`+4,278`）、`requests_processing=0` 全部单调准确对账 | **PASS** |
| **`FROM scratch` 精简镜像审计** | `local/qwen35-9b-defiant-fable-thor:0.1.0`：仅 25 文件，解压 `573.5 MiB`，压缩包 `572.3 MiB`，零权重/零 Python/零编译器/零 `.a`，`violations: []` | **PASS** |
| **零外网离线执行 (`--network none`)** | `--network none` + `/model:ro` 挂载下 **`6 秒`** 通过 `/runtime/healthcheck`；随后完整复测 C=1 `59.20` / PF4K `2,806.73` / C=4 `146.76` / C=8 `219.43 tok/s`，10/10 全通过 | **PASS** |

---

## 7. 最终锁定交付清单与双源仓库

- **运行配置（`manifests/runtimes.json`）**：
  - 镜像：`local/qwen35-9b-defiant-fable-thor:0.1.0`
  - `imageId`：`sha256:f66efa272d39f5ed2506831611b5717b69f73084ba1fe83e3d47dc417c32809d`
  - `configImageId`：`sha256:49c043252e3fd54f0a3d78826796c4728af109055ec2daefed1e2c88c59a3b38`
  - `archiveSize`：`600,118,941 B`（`572.3 MiB`），`sha256`：`5bf92a5d8bec90781ab7bc7c966782e2ebcaa2fd4a3dda61de3d364729f0833e`
  - 核心环境变量：`GGML_CUDA_MMQ_MAX_BATCH=256`、`LLAMA_MTP_DRAFT_HEAD_TYPE=mxfp4`、`GGML_CUDA_PDL=1`
  - 核心启动参数：`-fa on --kv-unified --jinja --metrics --reasoning-format deepseek --spec-type draft-mtp --spec-draft-n-max 2 -ctk q8_0 -ctv q8_0 --cache-reuse 256 -b 4096 -ub 4096 --temp 0.7 --top-p 0.85 --top-k 20 -t 8 -tb 12 --poll 100`
- **双源锁定提交（发布完成后自动回填）**：
| 产物类别 | 提供方 | 仓库 | 40 位固定提交 | 文件大小 (Bytes) | SHA256 | 匿名校验 |
| :--- | :--- | :--- | :--- | ---: | :--- | :---: |
| **主模型 `Q4_K_M`** | Hugging Face | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` | `4d47954f2ef8277567a20508dd3f88def5404455` | `6,979,975,392` | `d7eb4fac9389d53fa576f64a6ff53e914a00bc7705dc354d1065887565147320` | HTTP 206 OK |
| **主模型 `Q4_K_M`** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF` | `d11804e0df7f71b43d8054e1f96622067a0ed03a` | `6,979,975,392` | `d7eb4fac9389d53fa576f64a6ff53e914a00bc7705dc354d1065887565147320` | HTTP 206 OK |
| **视觉投影 `mmproj-F16`** | Hugging Face | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` | `4d47954f2ef8277567a20508dd3f88def5404455` | `918,166,080` | `f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f` | HTTP 206 OK |
| **视觉投影 `mmproj-F16`** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF` | `d11804e0df7f71b43d8054e1f96622067a0ed03a` | `918,166,080` | `f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f` | HTTP 206 OK |
| **精简 Thor 运行时镜像** | Hugging Face | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `df9f5266644e93c344739389ff31ffdc45d43b45` | `600,118,941` | `5bf92a5d8bec90781ab7bc7c966782e2ebcaa2fd4a3dda61de3d364729f0833e` | HTTP 206 OK |
| **精简 Thor 运行时镜像** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `b268a4fbbd98b07c22b2a34fc4e526a92bf23cfc` | `600,118,941` | `5bf92a5d8bec90781ab7bc7c966782e2ebcaa2fd4a3dda61de3d364729f0833e` | HTTP 206 OK |
| **控制端归档包 (`amd64`)** | Hugging Face | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `980afa7f60b3658418e826b841adcc0e5d5d02d1` | `3,049,602` | `4a4f02632b9944a1986d7431cf55f86952ecfa4e0f46353daee24f319416191d` | HTTP 206 OK |
| **控制端归档包 (`amd64`)** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `a18735ac3e8f502b9973b651b13895ecaadd0004` | `3,049,602` | `4a4f02632b9944a1986d7431cf55f86952ecfa4e0f46353daee24f319416191d` | HTTP 206 OK |
| **最终薄壳 LPK** | 本地 & 双源发布 | `dist/cloud.lazycat.aipod.qwen35-9b-defiant-fable-v0.1.0.lpk` | `v0.1.0` | `52,736` | `a55e93429a4609f61eabc6580155f6ed4bce971744a42e650ca00197096e0c0f` | 已校验 |
