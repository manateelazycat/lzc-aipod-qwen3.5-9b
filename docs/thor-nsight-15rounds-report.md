# Qwen3.5-9B-The-Defiant-Fable Jetson AGX Thor 移植与 15 轮 Nsight 性能优化测试报告（v0.2.0）

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
| **模型架构** | `Qwen3_5ForConditionalGeneration`（32 层 = 24 层 Gated DeltaNet + 8 层 Full Attention + 1 层内置原生 MTP `blk.32.nextn.*` + 27 层 ViT，`vocab=248,320`，`output.weight` 为 16-bit BF16 `2,034 MiB`） |
| **推理引擎基线** | `ggml-org/llama.cpp` @ `07ba74b`（MIT）+ 自主研发 R1–R15 Thor `sm_110a` CUDA 优化补丁集 |
| **投机解码机制** | **原生内置 MTP**（Multi-Token Prediction，`--spec-type draft-mtp`，复用 GGUF 内嵌 `blk.32.nextn.*` 预测头，无外挂 Draft 模型；R6+R12 显存内生成无损 `Q2_K` 影子 Draft `lm_head`，Target 验证严格保持原生 BF16） |
| **最终交付形态（v0.2.0）** | `FROM scratch` 精简单用途 Thor 容器（25 文件、`573.8 MiB` 解压 / `572.6 MiB` `.tar.gz`，零权重、零 Python、零编译器）+ 懒猫算力舱薄壳 LPK (`cloud.lazycat.aipod.qwen35-9b-defiant-fable` `v0.2.0`) |

### 评测方法论
- **剖析与复测严格隔离**：每轮先在稳态窗口内执行 `nsys`（`--cuda-graph-trace=node`，分别采集 C=1、4K Prefill、C=8 三个窗口）并在 R0 / R10 / R15 执行 `ncu --set speedoflight` Roofline 硬件计数器剖析；所有性能数字均在**无任何剖析器挂载**的纯净 HTTP SSE 服务下复测取得（`scripts/benchmark_qwen35_9b.py`）。
- **基准工作负载**：
  - **C=1 解码**：3 条不同题材独立提示（英文长文、Python 算法代码、中文技术解释，各 256 tokens，`ignore_eos=true`），取中位数与峰值，另单测 Thinking 思考模式吞吐与 MTP 接受率。
  - **4K 冷 Prefill**：~4,100 tokens 唯一加盐无缓存提示词，实测冷首字延迟（TTFT）与 Prefill 吞吐。
  - **多并发聚合**：C=4（512 tokens 总计）与 C=8（1,024 tokens 总计）并发 SSE 请求聚合吞吐。
  - **能力与协议门禁**：快速数学（`17*23=391`）、思考数学（`73*59+413=4386`）、OpenAI 工具调用、多模态视觉识别、16K/128K/256K 大海捞针、§17 Prometheus `/metrics` 增量对账、§19.1 默认非贪婪采样与 §19.5 失控重复守卫正/反例。

---

## 2. 端到端优化总成绩（R0 基线 vs v0.1.0 R10 vs v0.2.0 R15 精简容器）

| 核心指标 | R0 基线 (`sm_110`, 无 MTP) | v0.1.0 R10 精简容器 (`mxfp4, K=2`) | **v0.2.0 R15 精简容器 (`q2_k, K=3`, 默认)** | v0.2.0 R15 精简容器 (`q2_k, K=2` 对照) | v0.2.0 较 R0 基线提升 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **C=1 解码中位数 (tok/s)** | 32.95 | 59.20 (`+79.7%`) | **62.16**（最高 **63.99**） | **60.46** | **`+88.6%` (`1.89×`)** |
| **C=1 解码峰值 (tok/s)** | 32.96 | 65.89 (`+99.9%`) | **72.07**（最高 **72.18**） | 65.98 | **`+118.7%` (`2.19×`)** |
| **思考模式解码 (tok/s)** | 32.95 | 62.92 (`+91.0%`) | **63.35**（最高 **69.20**） | 63.12 | **`+92.3% ~ +110.0%`** |
| **4K 冷 Prefill (tok/s)** | 1,326.99 | 2,806.73 (`2.12×`) | **2,804.14**（R15 宿主机 **2,898.23** / 服务端 **2,919.93**） | 2,794.08 | **`+111.3% ~ +118.4%` (`2.11×–2.18×`)** |
| **4K 冷 Prefill TTFT (ms)** | 3,079.1 | 1,476.5 (`-52.0%`) | **1,472.8**（R15 宿主机 **1,410.2**） | 1,482.1 | **`-52.2% ~ -54.2%`** |
| **C=4 并发聚合 (tok/s)** | 114.17 | 146.76 (`+28.5%`) | **142.16** | 127.78 | **`+24.5%`** |
| **C=8 并发聚合 (tok/s)** | 192.92 | 219.43 (`+13.7%`) | **200.52**（较 R12 `K=3` `172.11` 恢复 **`+16.5%`**） | **209.82** | **`+3.9%`（`K=2` `+8.8%`）** |
| **MTP Draft `lm_head` 耗时 (nsys)** | 7,678 µs (BF16) | 2,302 µs (`MXFP4`) | **1,529.1 µs (`Q2_K`)** | **1,529.8 µs (`Q2_K`)** | **`-80.1%` (`5.02×` 提速)** |
| **Draft `lm_head` 显存开销** | 0（共享 BF16） | +540 MiB (`MXFP4`) | **+334 MiB (`Q2_K`)** | **+334 MiB (`Q2_K`)** | **较 v0.1.0 再省 `206 MiB`** |
| **6/6 精度与协议 + 4/4 §19 门禁** | 基线 | 10/10 全过 | **10/10 全过（含 `--network none` 11s 就绪）** | 6/6 全过 | 生产达标 |

---

## 3. 15 轮“Nsight 剖析 → 性能优化 → 非剖析复测”完整演进表（R0–R15）

### 阶段一：基础架构移植与首期 10 轮优化（R0–R10 → `v0.1.0`）

| 轮次 | 剖析发现的瓶颈 | 实施优化措施（对应补丁/参数） | C=1 中位 (tok/s) | C=1 峰值 (tok/s) | 4K Prefill (tok/s) | C=4 聚合 (tok/s) | C=8 聚合 (tok/s) | 决策与核心剖析证据 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **R0** | 基线：`sm_110` 编译、无 MTP、默认 F16 KV、`-ub 512` | 建立 `nsys`（3 窗口）+ `ncu`（45 核 Roofline）基线 | 32.95 | 32.96 | 1,326.99 | 114.17 | 192.92 | C=1 中 `MMVQ` 占 `66.9%`、BF16 `lm_head` 占 `24.1%`；Prefill `MMQ` 占 `80.3%`、标量 GDN 占 `8.6%` |
| **R1** | `sm_110` 未开架构专用指令；未启用模型自带原生 MTP 预测头 | 重编 `sm_110a` + `-DGGML_CUDA_FA_ALL_QUANTS=ON`；启用 `--spec-type draft-mtp` 并扫描 `K=1..4` | **42.44** (`K=1`) | 43.28 (`K=3` 峰 51.99) | 1,217.41 | **125.93** | 189.11 | **保留 `sm_110a` + MTP**；`K=1` 接受率 `71.1%`（`+28.8%`），但 `nsys` 暴露 BF16 `lm_head` 在 Target (`7.99 ms`) + Draft (`7.67 ms × K`) 合计占 **36.9%**，吃掉 `K=2/3` 收益 |
| **R2** | `r1_prefill4k` 显示上游 `gated-delta-net-mma.cu` 仅对 GB10 启用、漏掉 Thor `CC 1100`，导致 GDN 退化为标量核（`329 ms`, `9.7%`） | [apply_r2_gdn_mma.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r2_gdn_mma.py)：定义 `GGML_CUDA_CC_THOR 1100` 并开启 Tensor Core MMA `gdn_single` | **42.90** | 44.68 | **1,293.81** (`+6.3%`) | 123.08 | **193.68** | **保留**；`r2_prefill4k` 实证 GDN 从 `329 ms` 降至 **`127.0 ms`（`2.59×` 加速）**，6/6 精度全过 |
| **R3** | `r2_c1` 暴露 4 处上游 GB10 调优遗漏 Thor：无 LPDDR5X L2 预取、`MMVQ` 未用宽块、`M=2` 退回 `mul_mat_vec_q` 未进 `mmq-few` | [apply_r3_thor_uma_l2.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r3_thor_uma_l2.py)：为 Thor 启用 L2 预取、`MMVQ_PARAMETERS_GB10`、`mmq-few` `min_ncols=2`、`fattn` 整瓦片、`fp32_acc` | **45.73** (`+6.6%`) | 47.71 | **1,298.64** | 122.71 | **194.29** | **保留**；`r3_c1` 实证 `mul_mat_q_few` 接管 `M=2` 后 Q4_K 单次耗时 `75.2 → 65.7 µs`（`-12.6%`），C=8 FlashAttention 降 `22%` |
| **R4** | 128K/256K 下 F16 KV 占用 4.3–8.6 GiB 内存及带宽 | 扫描 `-fa on -ctk/-ctv`（`f16` vs `q8_0` vs `q4_0`）+ `--cache-reuse 256` | 44.72 (`q8_0`) | 47.58 | 1,288.49 | **124.56** | **194.14** | **保留 `q8_0` KV**；KV 显存减半，C=8 FlashAttention 再降 `12.8%`（`23.26 → 20.29 ms`），16K 大海捞针无损通过 |
| **R5** | `r4_prefill4k` 显示 `mul_mat_q` 独占 **`82.9%` (`2,656 ms`)**，远低于 Thor Blackwell Tensor Core 算力 | [apply_r5_mmq_cublas_thresh.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r5_mmq_cublas_thresh.py)：增加 `GGML_CUDA_MMQ_MAX_BATCH=256`，大 Batch 切换 cuBLAS 2CTA Tensor Core；扫描 `-b 4096 -ub 512..4096` | **45.66** | 47.24 | **2,814.26** (**`2.18×`**) | **126.96** | **210.33** | **保留（重大突破）**；`r5_prefill4k` 实证 `nvjet_sm110_tss_*` 全面接管，GPU 总耗时 `3,205.8 → 1,478.2 ms`（`-53.9%`），TTFT `3,245 → 1,468 ms` |
| **R6** | `r5_c1` 显示 BF16 `lm_head` 在 Draft 阶段每步耗 `7.68 ms`，成为 MTP `K≥2` 最大墙 | [apply_r6_mtp_draft_head.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r6_mtp_draft_head.py)：启动时在显存内为 MTP Draft 生成量化影子 `lm_head`（扫描 `q4_0`/`q4_k`/`q8_0`/`mxfp4` × `K=1..4`），Target 保持原版 BF16 | **58.77** (`mxfp4 K=2`) / **60.38** (`K=3`) | **63.03** / **66.84** | **2,823.06** | **133.24** / **139.27** | **220.43** (`K=2`) | **保留 `mxfp4 K=2`**；Draft `lm_head` 从 `7.68 ms` 暴降至 **`2.72 ms`（`2.82×`）**；Target 100% 无损 |
| **R7** | `r6_c1` 发现 `MXFP4` Draft Head 仅跑到 `198 GB/s` (`2.72 ms`)，且 2 GB BF16 `mul_mat_f` 线性扫描污染 L2；需对比 `PDL` 与 `Q6_K` | **R7a** 尝试 `MXFP4` 2×-nwarps + `mmf __ldcs` + `PDL=0/1` + `Q6_K` 对比；**R7b**（[apply_r7_thor_20sm_tuning.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r7_thor_20sm_tuning.py)）回退负优化并改为 `MXFP4 rows_per_block=2` | **57.72** (`K=2`) / **62.18** (`K=3`) | **63.53** / **66.79**（思考 **70.44**） | **2,837.58** | **137.89** | **212.82** | **保留 R7b + `PDL=1` + `Q4_K_M`**；回退 2×-nwarps 回归（`2,723 → 3,634 µs`），`rows_per_block=2` 压至 **`2,304 µs`**（`-15.4%`），`mmf __ldcs` 降 `104 µs/step` |
| **R8** | `r7b_c8` 中非 GEMM 热点 `gated_delta_net_cuda<128>` 占 `4.7%` (`190 ms`, `105.5 µs/call`)，逐元素标量访存 | [apply_r8_gdn_float4.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r8_gdn_float4.py)：针对 `S_v=128, !KDA` 重排为 128B 合并 `float4` (`LDG.E.128`/`STG.E.128`) | **58.79** (`K=2`) | **64.15** (`K=3` 峰 **73.20**) | **2,848.02** | 136.71 | 193.30 | **保留**；`gated_delta_net_cuda` 单次降 `5.0%`（`105.5 → 100.2 µs`），C=8 GPU 总时间降 `5.1%` |
| **R9** | 协议 §19.1/§19.5 要求默认非贪婪采样与零 GPU 开销失控重复守卫；14 核 ARM 线程与轮询调优 | [apply_r9_repetition_guard.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r9_repetition_guard.py) + `--temp 0.7 --top-p 0.85 --top-k 20` + `-t 8 -tb 12 --poll 100` | **59.41** | **66.01** | **2,916.48** | **137.41** | **201.38** | **保留**；零 GPU 开销，诱导循环 `98 tokens` 精准拦截，12 行 Markdown 表格零误杀 |
| **R10** | 首期收口、128K/256K 极限上下文与 `v0.1.0` `FROM scratch` 精简容器离线验收 | 固化 `v0.1.0` 配置，完成 128K/256K 捞针、`ncu` Roofline、`FROM scratch` 打包与 `--network none` 离线复测 | **59.20**（精简容器） | **65.89** | **2,806.73** | **146.76** | **219.43** | **全量通过**：128K (`116K` tok) & 256K (`226K` tok) 捞针 100% 召回；256K 内存增量 `25.7 GiB`（适配 64GB Thor）；10/10 门禁全过 |

### 阶段二：新增 5 轮极限硬件级深潜优化（R11–R15 → `v0.2.0`）

| 轮次 | 剖析发现的瓶颈 | 实施优化措施（对应补丁/参数） | C=1 中位 (tok/s) | C=1 峰值 (tok/s) | 4K Prefill (tok/s) | C=4 聚合 (tok/s) | C=8 聚合 (tok/s) | 决策与核心剖析证据 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **R11** | `r10_c1` 中 Target BF16 `lm_head`（`mul_mat_f`, `M=3`, `2,034 MiB`）占 **`20.5%`**（`8,120 µs`），验证能否通过 128-bit `uint4 __ldcs` 向量化 `mmvf` 或 cuBLAS 进一步提速 | [apply_r11_bf16_lmhead_vec128.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r11_bf16_lmhead_vec128.py)：在 `mmvf.cu` 实现 `uint4 __ldcs` BF16 循环并新增 `GGML_CUDA_MMVF_BF16_MAX_BATCH` / `GGML_CUDA_MMF_MAX_BATCH` 扫描 | 54.64 (`mmvf4`) / 55.40 (`cublas`) | 61.87 / 61.91 | 2,807.43 | 133.40 | 213.04 | **证伪并拒绝**；`r11_c1` 实证 `mul_mat_vec_f<BF16, M=3>` 耗时 `10,769.4 µs`（反慢 `+32.6%`），而 R7b `mul_mat_f __ldcs` 的 `7.78–8.12 ms` 已逼近 Thor `273 GB/s` 物理下限（`2,034 MiB / 273 GB/s = 7.45 ms`，效率 **92%–96%**），保留默认 `-1/16` |
| **R12** | `r10_c1` 中 MTP Draft `lm_head`（`MXFP4`, `540 MiB`, `2,302 µs`）仍占解码核心延迟，且 `K=3` 时需连续调用 3 次（`6.9 ms/step`） | [apply_r12_draft_head_ultracompact.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r12_draft_head_ultracompact.py)：扩展无损 Draft `lm_head` 支持 `Q2_K`/`Q3_K`/`Q2_0`/`IQ4_NL`/`IQ4_XS` 并为 2/3-bit 量化启用 `mmvq.cu` `rows_per_block=2` 宽访存 | **62.80** (`q2_k K=3`) / 58.93 (`K=2`) | **72.20** (`K=3`, 思考 **64.61**) | **2,900.37** | 127.97 / 138.83 | 172.11 (`K=3`) / 201.36 (`K=2`) | **采纳 `Q2_K`（重大突破）**；`r12_c1` 实证 `mul_mat_vec_q<Q2_K, 1>` 仅 **`1,529.8 µs`（较 `MXFP4` 再降 `-33.5%`，较原始 BF16 快 `5.02×`！）**，仅占 `334 MiB`（再省 `206 MiB`），Target 保持 BF16 数学无损；`Q2_0` 因接受率跌至 `35.0%` 被淘汰；但 `r12_c8` 暴露 `K=3 @ C=8`（`M=32`）跌出 `mmq-few` 上限（`24`）回退大核 `mul_mat_q`（`2,638.8 ms`） |
| **R13** | `r12_c8` 精准定位：`K=3 @ C=8` 总验证 Batch = `8×4 = 32` 超过 `mmq-few.cu` 硬编码 `MMQ_FEW_MAX_NCG=3`（`24` 列），导致 `MMQ` 暴涨至 `50.6%`（`2,638.8 ms`），C=8 掉至 `172.11 tok/s` | [apply_r13_thor_mmq_few.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r13_thor_mmq_few.py)：将 `MMQ_FEW_MAX_NCG` 扩至 `4`（解锁 `M=25..32`），并在 Thor 20-SM 上对比 `NCG1_TALL=0` vs `1` | **61.20** (`t1`) / 59.51 (`t0`) | **70.41** / **72.19** | 2,786.15 / 2,805.94 | **143.43** (`+12.0%`) | **191.42 / 195.19** (`+13.4%`) | **采纳 `MMQ_FEW_MAX_NCG=4`（并发突破）**；`r13_c8` 实证 `mul_mat_q_few<Q4_K, ncg=4, nmt=1>`（`110.3 µs/call`）取代慢速 `mul_mat_q`（`553 µs/call`），C=8 总 GPU 耗时从 `5,217.6 ms` 暴降至 **`4,688.2 ms`（`-10.1%`, `-529.4 ms`！）**，成功解除 `K=3` 高并发悬崖 |
| **R14** | `r13_prefill4k` 显示 `nvjet_sm110_tss_*`（BF16 输入 + FP32 累加）占 `35.9%`（`498.3 ms`），且 `dequantize_block_q4_K<__nv_bfloat16>` 占 `80.5 ms`；而全局开 `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` 会把 2 GiB 原生 BF16 `lm_head` 强制转 FP16 | [apply_r14_thor_fp16_prefill.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r14_thor_fp16_prefill.py)：新增选择性 `quant_f16` 模式（仅量化层反量化至 FP16，原生 BF16 `lm_head` 零转换）+ `GGML_CUDA_F16_PREFER_F32_OUT=0/1` A/B 验证 | 59.14 (`acc32`) / 56.69 (`acc16`) | **72.79** (`acc32`) | **2,872.00** (`srv=2,893.29`, TTFT `1,424.8 ms`) | 129.94 | **201.34** | **采纳 `quant_f16` + `F32_OUT=1`**；`r14_prefill4k` 实证 `acc16` (`F32_OUT=0`) 虽令 GEMM 降 `43 ms`、反量化降 `14.3%`（`80.5 → 68.9 ms`），但引发 `256` 次 `convert_unary_cont_vec4<__half, float>`（`+134.9 ms`）导致净减速；改为 **`F32_OUT=1`** 后走 `nvjet_sm110_hss_*` 直接 FP32 输出，消除 `134.9 ms` 转换，PF4K 净增 `+73 tok/s`（`-63 ms` TTFT） |
| **R15** | `r13/r14_prefill4k` 暴露最大非 GEMM 热点：`unary_gated_op_kernel<&op_silu, float>` 占 `7.8%`（`108.9 ms`, `789.4 µs/call`），存在逐标量 32-bit 访存及每元素两次 64-bit 整除（`i/n`, `i%n`），仅跑到 65% 带宽 | [apply_r15_non_gemm_vec4.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r15_non_gemm_vec4.py)：在 `unary.cu` 实现无除法 128-bit `float4` 向量化核 `unary_gated_op_kernel_f32_vec4`（连续+步长双快路径）并完成最终 `nsys`/`ncu` 与 `v0.2.0` 精简容器验收 | **62.16**（容器 `K=3`，最高 **63.99**；`K=2` **60.46**） | **72.07**（最高 **72.18**，思考 **67.63–69.20**） | **2,898.23**（宿主机 `srv=2,919.93`, TTFT `1,410.2 ms`；容器 `2,804.14`） | **142.16** | **200.52**（`K=2` **209.82**） | **采纳并定版 `v0.2.0`**；`r15_prefill4k` 实证 `unary_gated_op_kernel_f32_vec4<&op_silu>` 降至 **`96.74 ms`（`701.0 µs/call`, `-11.2%`）**，C=8 `OTHER_ELEMWISE` 降 `-11.6%`，C=8 GPU 总耗时刷新最低纪录 **`4,661.3 ms`**；`r15_ncu` 验证 `silu`/`sigmoid` 双双命中 `vec4`；`v0.2.0` 纯精简容器 10/10 全过 |

---

## 4. Nsight Systems (`nsys`) 与 Nsight Compute (`ncu`) 深度剖析对比（R0 → R10 → R15）

### 4.1 `nsys` 核函数分类耗时演进

#### C=1 解码窗口（256 tokens）
| 核函数类别 | R0 基线 (无 MTP) | R10 (`mxfp4, K=2`) | **R15 (`q2_k, K=3`, 全栈)** | 变化与根因 |
| :--- | :---: | :---: | :---: | :--- |
| **总 GPU 核函数时间** | `8,234.40 ms` (227,238 launches) | `5,264.26 ms` (183,615 launches) | **`5,255.53 ms`** (178,652 launches) | 较 R0 GPU 耗时 **`-36.2%`**；叠加 `K=3` + `Q2_K` Draft Head 端到端解码提速 **`+88.6%`（峰值 `+118.7%`）** |
| `MMQ_FEW`（Target Verify GEMM） | 40.49 ms (`0.5%`) | 2,605.56 ms (`49.5%`) | 2,604.37 ms (`49.6%`) | 全面接管小 Batch Target Verify，Q4_K 仅 `66.4–88.6 µs/call` |
| `MMVQ`（含 Draft `lm_head`） | 5,506.65 ms (`66.9%`) | 918.98 ms (`17.5%`) (`MXFP4` `2,302 µs`) | **944.65 ms (`18.0%`)**（**`Q2_K` 仅 `1,529.1 µs`**, 378 次调用） | R12 `Q2_K` 2-row MMVQ 将单次 Draft `lm_head` 再压 **`-33.5%`**（较原始 BF16 提速 **`5.02×`**），支撑 `K=3` 爆发 |
| `FP16/BF16_GEMM`（Target `lm_head`） | 1,981.75 ms (`24.1%`) | 1,080.01 ms (`20.5%`) | **1,027.93 ms (`19.6%`)**（`124x`, `8,036 µs`） | Target 验证步数进一步减少（`256 → 133 → 124` 次），`__ldcs` 保持 92%–96% 内存带宽效率 |
| `OTHER_ELEMWISE` / `QUANT` / `NORM` | 451.55 ms | 356.10 ms | **308.30 ms** | R15 `float4` SwiGLU + 更少验证步数持续压缩非 GEMM 开销 |

#### 4K 冷 Prefill 窗口（~4,100 tokens）
| 核函数类别 | R0 基线 | R10 (`BF16 cuBLAS`) | R14 (`quant_f16, acc16`) | **R15 (`quant_f16, F32_OUT=1` + `GLU_VEC4`)** | 变化与根因 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **总 GPU 核函数时间** | `3,120.50 ms` | `1,389.74 ms` | `1,468.17 ms` | **`1,391.06 ms`**（非剖析 TTFT **`1,410.2 ms`**, `2,898.2 tok/s`） | 较 R0 提速 **`2.24×`**，内核启动数减 **`59.4%`**（`10,376 → 4,216`） |
| `MMQ` → `FP16/BF16_GEMM` | `2,506.71` / `7.79 ms` | `53.54` / `505.06 ms` | `53.62` / `455.23 ms` | **`53.62` / `510.40 ms`**（`nvjet_sm110_hss_*` 直接 FP32 输出） | 消除 R14 `acc16` 的 **`134.88 ms`** `half→float` 额外输出转换 |
| `GDN_AR` → `GDN_MMA` | `268.63 ms` (标量) | `136.61 ms` (MMA) | `136.77 ms` | **`137.07 ms`** | R2 Thor Tensor Core MMA 稳定提速 `2.0×–2.6×` |
| `unary_gated_op_kernel<&op_silu>` | `109.1 ms` | `108.94 ms` (`789.4 µs/call`) | `106.39 ms` (`770.9 µs/call`) | **`96.74 ms` (`701.0 µs/call`, `-11.2%`)** | R15 `unary_gated_op_kernel_f32_vec4` 彻底消除 64-bit 整除并启用 128-bit `float4` 读写 |

#### C=8 并发窗口（`K=3`, `M=32` 解锁对比：R12 vs R13 vs R15）
| 核函数类别 | R12 (`K=3, MAX_NCG=3`) | R13 (`K=3, MAX_NCG=4`) | **R15 (`K=3, MAX_NCG=4 + GLU_VEC4`)** | 变化与根因 |
| :--- | :---: | :---: | :---: | :--- |
| **总 GPU 核函数时间** | `5,217.61 ms` | `4,688.17 ms` (`-10.1%`) | **`4,661.26 ms` (`-10.7%`, `-556.4 ms`)** | R13 解锁 `M=32` 消除大核悬崖 + R15 向量化 SwiGLU |
| `MMQ`（大 Batch 回退） | `2,638.77 ms` (`50.6%`) | `615.79 ms` (`13.1%`) | **`604.21 ms` (`13.0%`)** | 仅剩 `Q2_K` Draft `lm_head` (`3,863 µs/call`)，全部线性层回流 `mmq-few` |
| `MMQ_FEW`（`ncg=4` 新接管） | `732.05 ms` (`14.0%`) | `2,133.54 ms` (`45.5%`) | **`2,100.10 ms` (`45.1%`)** | `mul_mat_q_few<Q4_K, 4, 1>` 仅 `110.2 µs/call`（取代 `553 µs` `mul_mat_q`） |
| `OTHER_ELEMWISE`（含 SwiGLU） | `163.22 ms` | `192.91 ms` | **`170.58 ms`（较 R13 `-11.6%`）** | R15 `float4` SwiGLU 在多并发激活流中同步生效 |

### 4.2 `ncu` Roofline 硬件计数器对比（R0 → R10 → R15）
| 关键核函数 | R0 实测 Duration / SM% / Mem% / L2% | R10 实测 Duration / SM% / Mem% / L2% | **R15 实测 Duration / SM% / Mem% / L2% / Occ%** | 硬件级演进结论 |
| :--- | :--- | :--- | :--- | :--- |
| `mul_mat_q_few<Q4_K, 2, 2, 0>` | `116.74 µs` / `34.6%` / `22.9%` / `22.9%` | `97.15 µs` / `28.1%` / `23.2%` / `23.2%` | `108.03 µs` / **`30.2%`** / **`24.5%`** / **`24.5%`** / `14.5%` | Thor 20-SM 稳定高吞吐运行，L2 与内存吞吐进一步提升 |
| `mul_mat_q_few<Q6_K, 2, 1, 0>` | —（R0 走 MMVQ） | `129.92 µs` / `42.8%` / `36.5%` / `34.1%` | `189.89 µs` / `35.4%` / `34.2%` / `33.2%` / `16.0%` | 大维度 FFN/输出投影持续高效占满算力与带宽通道 |
| `gated_delta_net_cuda<128>` | `54.40 µs` / `24.6%` / `30.7%` / `10.3%` | `53.02 µs` / `17.8%` / `26.1%` / `8.5%` | **`40.99 µs`** / `23.1%` / **`34.0%`** / **`11.0%`** / **`42.3%`** | R8 `float4` 128B 合并访存将单核耗时压至 `40.99 µs`（较 R0 **`-24.7%`**），带宽利用率提升至 `34.0%` |
| `unary_gated_op_kernel_f32_vec4<&op_silu>` | —（R0/R10 标量 + 64-bit 整除） | — | **`17.28 µs`**（短窗口）/ **`55.9%` Occ** | R15 向量化核成功替代标量 `silu` 与 `sigmoid` 两条路径 |

---

## 5. 被拒绝 / 证伪的实验记录（严格工程留痕，R0–R15）

1. **R7a `MXFP4` 2×-nwarps 升宽（已回退）**：`mul_mat_vec_q<MXFP4>` 从 `2,723 µs` 退化至 `3,634 µs`（`+33.4%`），改用 R7b `rows_per_block=2` 后降至 `2,304 µs`。
2. **R7a 关闭 Blackwell PDL (`GGML_CUDA_PDL=0`)（已放弃）**：C=1 与 C=8 分别下降 `3.0%` / `2.6%`，保持 `GGML_CUDA_PDL=1`。
3. **R7a 权重选型 `plusIQ-TOOLS-Q6_K` vs `Q4_K_M`**：`Q6_K` 慢约 `20%` 且多占 `1.78 GB` 显存，锁定 `Q4_K_M`。
4. **R11 Target BF16 `lm_head` 128-bit `mmvf` / 全 cuBLAS 路径（已拒绝）**：`nsys` 实证 `mul_mat_vec_f<BF16, M=3>` 耗 `10,769.4 µs`（慢 `+32.6%`），全 cuBLAS 路径 C=1 仅 `55.40 tok/s`；R7b `mul_mat_f __ldcs` 已跑出 `92%–96%` 物理内存带宽极限，保留默认路径。
5. **R12 `Q2_0` / `IQ4_NL` / `IQ4_XS` Draft Head（已淘汰）**：`Q2_0` 缺少超块缩放导致 MTP 接受率崩塌至 `35.0%`（C=1 跌至 `46.64 tok/s`）；`IQ4_NL`/`IQ4_XS` 反量化算术开销偏高（C=1 `55.5–55.9 tok/s`）；最终锁定 `Q2_K`（`1,529.8 µs`、`334 MiB`、接受率 `48%–56%`）。
6. **R14 `quant_f16` + `CUBLAS_COMPUTE_16F` (`F32_OUT=0`)（已放弃，改用 `F32_OUT=1`）**：虽降低 GEMM `43 ms` 及反量化 `11.5 ms`，但触发 `256` 次 `convert_unary_cont_vec4<__half, float>` 额外耗 `134.88 ms`，净慢 `~80 ms`（PF4K `2,642 tok/s`）；切换 `GGML_CUDA_F16_PREFER_F32_OUT=1` 直接 FP32 输出后 PF4K 升至 **`2,872–2,898 tok/s`**。

---

## 6. 稳定性、长上下文（128K/256K）、64GB 内存与 `v0.2.0` 精简容器验收回执

| 验收项目 | 实测证据与结果 | 结论 |
| :--- | :--- | :---: |
| **128K 长上下文大海捞针** | 实测 `116,070` prompt tokens，耗时 `80.18 s`（`1,447.58 tok/s`），精确输出 `TITAN-OBSIDIAN-7194` | **PASS** |
| **256K 极限长上下文大海捞针** | 实测 `226,270` prompt tokens，耗时 `240.39 s`（`941.25 tok/s`），精确输出 `AURORA-QUARTZ-5283` | **PASS** |
| **64GB Thor T4000 内存预算** | `Q2_K` Draft Head 较 `v0.1.0` 再省 `206 MiB`；256K 极限内存增量 `< 25.5 GiB`，为 64GB Thor 留有 >38 GiB 裕量 | **PASS** |
| **§19.1 默认非贪婪采样 (`v0.2.0` 容器)** | `--temp 0.7 --top-p 0.85 --top-k 20`；未传采样参数请求 `19*21` 精确返回 `399` | **PASS** |
| **§19.5 诱导重复精准拦截 (`v0.2.0` 容器)** | 强制 `T=0` 无限重复短语在 **`98` tokens (`2.82 s`)** 触发，以 `finish_reason="stop"` 优雅终止 | **PASS** |
| **§19.5 正常表格防误杀 (`v0.2.0` 容器)** | 12 行 Markdown 表格 + `ITEM-001..012` 完整输出（`619` 字符），零误杀 | **PASS** |
| **§19.5 `ignore_eos=true` 旁路 (`v0.2.0` 容器)** | 基准压测 `ignore_eos=true` 跑满 `220` tokens，不误拦截 | **PASS** |
| **§17 Prometheus `/metrics` (`v0.2.0` 容器)** | `llamacpp:prompt_tokens_total`（`+21,475`）、`tokens_predicted_total`（`+4,181`）、`requests_processing=0` 单调准确对账 | **PASS** |
| **`FROM scratch` `v0.2.0` 精简镜像审计** | `local/qwen35-9b-defiant-fable-thor:0.2.0`：25 文件，解压 `573.8 MiB`，压缩包 `572.6 MiB`，零权重/零 Python/零编译器/零 `.a`，`violations: []` | **PASS** |
| **零外网离线执行 (`--network none`)** | `--network none` + `/model:ro` 挂载下 **`11 秒`** 通过 `/runtime/healthcheck`；纯精简容器实测 `K=3` C=1 **`62.16`**（最高 `63.99`，峰值 **`72.07–72.18`**）/ PF4K **`2,804.14`** / C=4 **`142.16`** / C=8 **`200.52 tok/s`**（`K=2` 对照 C=1 `60.46` / C=8 `209.82`），10/10 全通过 | **PASS** |

---

## 7. 最终锁定交付清单与双源仓库（`v0.2.0`）

- **运行配置（`manifests/runtimes.json`）**：
  - 镜像：`local/qwen35-9b-defiant-fable-thor:0.2.0`
  - `imageId`：`sha256:3222133270f3d08a6d0f70f054ca5b1a2345e10bb264f2287291a0f239f1b459`
  - `configImageId`：`sha256:aca96d75368368b088a0b03ab88ef8f09af4a9f0482f2e9e7170b2ecdc0d3c07`
  - `archiveSize`：`600,387,251 B`（`572.6 MiB`），`sha256`：`dd9a9f10b712326c70cac340d7f09e260e3249159641b76553dc8c042e493d4e`
  - 核心环境变量：`GGML_CUDA_MMQ_MAX_BATCH=256`、`GGML_CUDA_MMQ_FEW_MAX_NCG=4`、`GGML_CUDA_CUBLAS_COMPUTE_TYPE=quant_f16`、`GGML_CUDA_F16_PREFER_F32_OUT=1`、`GGML_CUDA_GLU_VEC4=1`、`LLAMA_MTP_DRAFT_HEAD_TYPE=q2_k`、`GGML_CUDA_PDL=1`
  - 核心启动参数：`-fa on --kv-unified --jinja --metrics --reasoning-format deepseek --spec-type draft-mtp --spec-draft-n-max 3 -ctk q8_0 -ctv q8_0 --cache-reuse 256 -b 4096 -ub 4096 --temp 0.7 --top-p 0.85 --top-k 20 -t 8 -tb 12 --poll 100`
- **双源锁定提交（`v0.2.0`）**：
| 产物类别 | 提供方 | 仓库 | 40 位固定提交 | 文件大小 (Bytes) | SHA256 | 匿名校验 |
| :--- | :--- | :--- | :--- | ---: | :--- | :---: |
| **主模型 `Q4_K_M`** | Hugging Face | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` | `4d47954f2ef8277567a20508dd3f88def5404455` | `6,979,975,392` | `d7eb4fac9389d53fa576f64a6ff53e914a00bc7705dc354d1065887565147320` | HTTP 206 OK |
| **主模型 `Q4_K_M`** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF` | `d11804e0df7f71b43d8054e1f96622067a0ed03a` | `6,979,975,392` | `d7eb4fac9389d53fa576f64a6ff53e914a00bc7705dc354d1065887565147320` | HTTP 206 OK |
| **视觉投影 `mmproj-F16`** | Hugging Face | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` | `4d47954f2ef8277567a20508dd3f88def5404455` | `918,166,080` | `f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f` | HTTP 206 OK |
| **视觉投影 `mmproj-F16`** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF` | `d11804e0df7f71b43d8054e1f96622067a0ed03a` | `918,166,080` | `f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f` | HTTP 206 OK |
| **精简 Thor 运行时镜像 `0.2.0`** | Hugging Face | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `__HF_RUNTIME_OID__` | `600,387,251` | `dd9a9f10b712326c70cac340d7f09e260e3249159641b76553dc8c042e493d4e` | PENDING |
| **精简 Thor 运行时镜像 `0.2.0`** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `a3bfc6538c6182dc8cf8200273d1a619bb6140c9` | `600,387,251` | `dd9a9f10b712326c70cac340d7f09e260e3249159641b76553dc8c042e493d4e` | PENDING |
| **控制端归档包 (`amd64`, `0.2.0`)** | Hugging Face | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `__HF_CTRL_OID__` | `__CTRL_SIZE__` | `__CTRL_SHA256__` | PENDING |
| **控制端归档包 (`amd64`, `0.2.0`)** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `__MS_CTRL_OID__` | `__CTRL_SIZE__` | `__CTRL_SHA256__` | PENDING |
| **最终薄壳 LPK (`v0.2.0`)** | 本地 & 双源发布 | `dist/cloud.lazycat.aipod.qwen35-9b-defiant-fable-v0.2.0.lpk` | `v0.2.0` | `__LPK_SIZE__` | `__LPK_SHA256__` | PENDING |
