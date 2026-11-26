# Qwen3.5-9B-The-Defiant-Fable 双平台（Jetson AGX Thor + AGX Orin 64GB）移植与 Nsight 性能优化测试报告（v0.2.0）

## 1. 双平台实测硬件与模型基线

| 维度 | Jetson AGX Thor（`thor`, `sm_110a`） | Jetson AGX Orin 64GB（`orin`, `sm_87`） |
| :--- | :--- | :--- |
| **实测主机** | `192.168.1.199` (`lzc-pod-XhkQoD`, Serial `1422326011571`) | `192.168.1.86` (`lzc-aipod-ipslmk`, Serial `1425124334189`) |
| **GPU 架构 / SM** | Blackwell `sm_110a` (CC `11.0`), 20 SMs @ `1.569 GHz`, `MAXN` | Ampere `sm_87` (CC `8.7`), 16 SMs @ `1.29 GHz`, `MAXN` |
| **L2 缓存 / 共享内存** | 大容量 L2 + Blackwell 2CTA TCGen05 MMA + PDL | **4 MB L2** / 164 KB SMEM/SM / 无硬件 PDL / 无 FP8 KV |
| **统一内存 / 带宽** | 122 GiB LPDDR5X UMA（`273 GB/s` 理论峰值） | 61.4 GiB LPDDR5 UMA（`204.8 GB/s` 理论峰值） |
| **CPU** | 14-core Arm Neoverse V3AE（`-t 8 -tb 12`, `OMP=8`） | 12-core Cortex-A78AE（`-t 4 -tb 8`, `OMP=4`） |
| **系统与工具链** | L4T `R39.2.0`, Driver `595.78`, CUDA `13.2`, `nsys 2026.3.1`, `ncu 2026.1.1` | L4T `R39.2.1`, Driver `595.78`, CUDA `13.2.86`, `nsys 2026.3.1`, `ncu 2026.1.1` |
| **上游模型仓库** | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` (Apache-2.0) | 同左（双平台共享同一套无损权重，零重复下载） |
| **上游固定提交** | `4d47954f2ef8277567a20508dd3f88def5404455` | 同左 |
| **锁定主权重 & 视觉投影** | `Q4_K_M` (`6,979,975,392 B`, SHA256 `d7eb4fac...147320`) + `mmproj-F16.gguf` (`918,166,080 B`, SHA256 `f70dc350...6ae029f`) | 同左 |
| **模型架构** | `Qwen3_5ForConditionalGeneration`（24 层 Gated DeltaNet + 8 层 Full Attention + 1 层内置 MTP `blk.32.nextn.*` + 27 层 ViT，`vocab=248,320`，`output.weight` BF16 `2,034 MiB`） | 同左 |
| **投机解码机制** | **原生内置 MTP**（`--spec-type draft-mtp`，无外挂 Draft 模型；显存内无损 `Q2_K` 影子 Draft `lm_head`，Target 严格原生 BF16 + 并发自适应动态深度） | 同左（Thor `K=3/2/2`，Orin `K=3/2/1`） |
| **推理引擎基线** | `ggml-org/llama.cpp` @ `07ba74b`（MIT）+ 补丁 R2–R19 | 同左 + Orin `sm_87` 专属补丁 O2/O4 (`apply_orin_sm87_opt.py`, `apply_orin_o4_mmvq_tuning.py`) |
| **交付精简镜像 (`v0.2.0`)** | `FROM scratch`（25 文件，归档 `572.6 MiB`，解压 `573.8 MiB`，`violations: []`） | `FROM scratch`（29 文件，归档 `571.9 MiB`，`violations: []`） |

### 评测方法论
- **剖析与复测严格隔离**：每轮均先在稳态窗口内执行 `nsys`（`--cuda-graph-trace=node`，分别采集 C=1、4K Prefill、C=8 三个窗口）及 `ncu` Roofline（Orin UMA 严格加 `--cache-control none`）；所有对外报告的吞吐与延迟数字均在**无任何剖析器挂载**的纯净 HTTP SSE 服务下独立测得。
- **门禁标准**：6/6 功能精度（快速数学 `17*23=391`、思考数学 `137*29+413=4386`、OpenAI 工具调用、多模态视觉识别、16K 大海捞针、§17 Prometheus `/metrics` 单调增量）+ 4/4 §19 稳定性门禁（默认非贪婪采样、诱导循环拦截、12 行 Markdown 表格零误杀、`ignore_eos` 旁路）+ 128K/256K 极限长上下文与 64GB UMA 零 OOM 验证 + `--network none` 离线容器验收。

---

## 2. 双平台端到端优化总成绩（`v0.2.0` 纯精简容器实测）

| 核心指标 | Thor R0 基线 | **Thor `v0.2.0` 精简容器 (R20)** | Thor 提升 | Orin O0 基线 | **Orin `v0.2.0` 精简容器 (O5)** | Orin 提升 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C=1 单流解码中位数 (tok/s)** | `32.95` | **`60.92`**（宿主机 R19 `63.63`，R15 容器 `62.16–63.99`） | **`+84.9% ~ +93.1%` (`1.85×–1.93×`)** | `21.93` | **`43.23`**（Run 2 `42.80`） | **`+97.1%` (`1.97×`)** |
| **C=1 单流解码峰值 (tok/s)** | `32.96` | **`70.49`**（R16/R18 峰值 **`74.51`**） | **`+113.9% ~ +126.1%` (`2.14×–2.26×`)** | `21.94` | **`50.57`**（O4 峰值 **`53.94`**） | **`+130.5% ~ +145.9%` (`2.30×–2.46×`)** |
| **C=1 深度思考解码 (tok/s)** | `32.95` | **`65.70`**（R15/R18 峰值 **`72.73`**） | **`+99.4% ~ +120.7%`** | `21.92` | **`43.89`**（O4 峰值 **`48.02`**） | **`+100.2% ~ +119.1%`** |
| **4K 冷 Prefill 吞吐 (tok/s)** | `1,326.99` | **`2,874.28`**（`srv=2,894.53`, R18 峰 `2,922.92`） | **`+116.6%` (`2.17×`)** | `847.41` | **`1,020.48`**（`srv=1,025.74`, O4 峰 `1,027.26`） | **`+20.4% ~ +21.2%`** |
| **4K 冷 Prefill TTFT (ms)** | `3,079.1` | **`1,434.1`**（R18 最低 `1,406.8`） | **`-53.4%`** | `4,863.0` | **`4,025.5`** | **`-17.2%` (`-837 ms`)** |
| **C=4 聚合吞吐 (tok/s)** | `114.17` | **`133.46`**（R15 容器 `142.16`） | **`+16.9% ~ +24.5%`** | `74.34` | **`94.92`**（单路中位 `29.95`） | **`+27.7%`** |
| **C=8 聚合吞吐 (tok/s)** | `192.92` | **`198.98`**（R17 `adapt322` **`209.15`**） | **`+3.1% ~ +8.4%`** | `125.65` | **`123.85`**（O4 峰值 `133.71`） | 持平（高并发不掉速） |
| **C=1 GPU 总耗时 (`nsys`)** | `8,234.4 ms` | **`4,912.0 ms`** (R18) | **`-40.3%` (`1.68×`)** | `12,148.1 ms` | **`6,056.0 ms`** (O4) | **`-50.1%` (`2.01×`)** |
| **Draft `lm_head` (`nsys`)** | `7,678 µs` (BF16) | **`1,529.1 µs`** (`Q2_K`, `334 MiB`) | **`5.02×` 提速** | `2,572.6 µs` (O3) | **`1,981.7 µs`** (O4 1-row/warp) | **`-23.0%` (`82.4%` 带宽)** |
| **128K / 256K 大海捞针 & UMA** | — | `116K` (`1,448 tok/s`) & `226K` (`941 tok/s`), 256K RSS `<25.5 GiB` | **100% 召回** | — | `110K` (`705 tok/s`, RSS `19.1 GiB`) & `209K` (`528 tok/s`, RSS `25.95 GiB`, 余 `29.4 GiB`) | **100% 召回, 零 OOM** |
| **断网容器验收 (`--network none`)** | — | `12s` 通过 `/runtime/healthcheck`, 10/10 全过 | **PASS** | — | `14s` 通过 `/runtime/healthcheck`, 10/10 全过 | **PASS** |

---

## 3. AGX Thor（`sm_110a`）20 轮优化完整演进（R0–R20）

### 阶段一：基础移植与首期 10 轮优化（R0–R10 → `v0.1.0`）

| 轮次 | 剖析发现的瓶颈 | 实施优化与补丁 | C=1 中位 / 峰值 | 4K Prefill | C=4 / C=8 | 决策与核心证据 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **R0** | 基线：`sm_110`、无 MTP、`f16` KV、`-ub 512` | 建立 `nsys`（3 窗口）+ `ncu`（45 核）基线 | `32.95` / `32.96` | `1,326.99` | `114.17` / `192.92` | `MMVQ` 占 C=1 `66.9%`；Prefill `MMQ` 占 `80.3%`、标量 GDN 占 `8.6%` |
| **R1** | 未开 `sm_110a` 与原生 MTP 投机解码 | 重编 `sm_110a`；开启 `--spec-type draft-mtp` | `42.44` / `43.28` | `1,217.41` | `125.93` / `189.11` | **保留**；`K=1` 接受率 `71.1%`，但 BF16 `lm_head` 占 `36.9%` |
| **R2** | `gated-delta-net-mma.cu` 漏掉 Thor `CC 1100` | [apply_r2_gdn_mma.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r2_gdn_mma.py) | `42.90` / `44.68` | `1,293.81` | `123.08` / `193.68` | **保留**；GDN Prefill `329 → 127.0 ms`（**`2.59×`**） |
| **R3** | 上游 GB10 UMA 调优漏掉 Thor `CC 1100` | [apply_r3_thor_uma_l2.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r3_thor_uma_l2.py) | `45.73` / `47.71` | `1,298.64` | `122.71` / `194.29` | **保留**；`mmq-few` 接管 `M=2`，Q4_K `75.2 → 65.7 µs` |
| **R4** | 长上下文 `f16` KV 带宽与显存占压 | `-fa on -ctk q8_0 -ctv q8_0 --cache-reuse 256` | `44.72` / `47.58` | `1,288.49` | `124.56` / `194.14` | **保留 `q8_0` KV**；KV 减半，C=8 FA 降 `-12.8%` |
| **R5** | Prefill `mul_mat_q` 独占 `82.9%` (`2,656 ms`) | [apply_r5_mmq_cublas_thresh.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r5_mmq_cublas_thresh.py) (`MMQ_MAX_BATCH=256`, `-b 4096 -ub 4096`) | `45.66` / `47.24` | **`2,814.26`** | `126.96` / `210.33` | **保留（重大突破）**；2CTA cuBLAS 接管，PF4K 提速 **`2.18×`** |
| **R6** | Draft BF16 `lm_head` 每步 `7.68 ms` | [apply_r6_mtp_draft_head.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r6_mtp_draft_head.py)（无损显存影子 Draft Head） | `58.77` / `63.03` | `2,823.06` | `133.24` / `220.43` | **保留**；`MXFP4` Draft Head `7.68 → 2.72 ms`（`2.82×`） |
| **R7** | `MXFP4` 仅跑 `198 GB/s`；2 GB BF16 `lm_head` 污染 L2 | [apply_r7_thor_20sm_tuning.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r7_thor_20sm_tuning.py) (`rows_per_block=2` + `mmf __ldcs`) | `62.18` (`K=3`) | `2,837.58` | `137.89` / `212.82` | **保留 R7b**；`MXFP4` `2,723 → 2,304 µs`（`-15.4%`），`mmf __ldcs` 降 `104 µs/step` |
| **R8** | `gated_delta_net_cuda<128>` 逐标量访存 | [apply_r8_gdn_float4.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r8_gdn_float4.py)（128B `float4` 合并访存） | `58.79` / `64.15` | `2,848.02` | `136.71` / `193.30` | **保留**；GDN 单次 `-5.0%`，C=8 GPU 总耗降 `-5.1%` |
| **R9** | §19.1/§19.5 默认非贪婪采样与零 GPU 重复守卫 | [apply_r9_repetition_guard.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r9_repetition_guard.py) + `-t 8 -tb 12 --poll 100` | `59.41` / `66.01` | `2,916.48` | `137.41` / `201.38` | **保留**；零 GPU 开销，诱导循环 `98 tokens` 精准拦截 |
| **R10** | 首期收口与 `v0.1.0` `FROM scratch` 容器验收 | 128K/256K 捞针、`ncu`、断网容器验收 | `59.20` / `65.89` | `2,806.73` | `146.76` / `219.43` | **通过**；10/10 门禁全过，128K/256K 100% 召回 |

### 阶段二：第 11–20 轮极限硬件级深潜优化（R11–R20 → `v0.2.0`）

| 轮次 | 剖析发现的瓶颈 | 实施优化与补丁 | C=1 中位 / 峰值 | 4K Prefill | C=4 / C=8 | 决策与核心证据 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **R11** | 验证 BF16 Target `lm_head` (`M=2..4`) 能否用 `mmvf` 128-bit 加速 | [apply_r11_bf16_lmhead_vec128.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r11_bf16_lmhead_vec128.py) | `54.64` / `61.87` | `2,807.43` | `133.40` / `213.04` | **证伪拒绝**；`mmvf` 反慢 `+32.6%`（`10.77 ms`），R7b `mmf __ldcs` 已达 `273 GB/s` 物理极限的 **92–96%** |
| **R12** | `MXFP4` Draft Head 占 `540 MiB` / `2,302 µs`，限制 `K=3` | [apply_r12_draft_head_ultracompact.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r12_draft_head_ultracompact.py)（`Q2_K` + `rows_per_block=2`） | **`62.80`** / **`72.20`** | `2,900.37` | `127.97` / `172.11` | **采纳 `Q2_K`**；Draft Head 压至 **`1,529.8 µs`（`5.02×` vs BF16）**，省 `206 MiB`；淘汰 `Q2_0`（接受率 `35%`） |
| **R13** | `K=3 @ C=8` (`M=32`) 溢出 `mmq-few` (`MAX_NCG=3` 上限 24) | [apply_r13_thor_mmq_few.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r13_thor_mmq_few.py)（`MMQ_FEW_MAX_NCG=4`） | `61.20` / `72.19` | `2,805.94` | `143.43` / `195.19` | **采纳**；C=8 GPU 总耗时 `5,217.6 → 4,688.2 ms`（**`-10.1%`**），解除高并发悬崖 |
| **R14** | Prefill 量化层走 BF16 cuBLAS (`nvjet_sm110_tss`) 算力减半 | [apply_r14_thor_fp16_prefill.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r14_thor_fp16_prefill.py)（`quant_f16` + `F32_OUT=1`） | `59.14` / `72.79` | **`2,872.00`** | `129.94` / `201.34` | **采纳 `F32_OUT=1`**；淘汰 `F32_OUT=0`（引发 `134.9 ms` `half→float` 转换），直接 FP32 输出净提 `+73 tok/s` |
| **R15** | Prefill 最大非 GEMM 热点：标量 `unary_gated_op_kernel<&op_silu>` (`108.9 ms`) | [apply_r15_non_gemm_vec4.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r15_non_gemm_vec4.py)（无除法 128-bit `float4` SwiGLU） | **`62.16–63.99`** / `72.18` | **`2,898.23`** | **`142.16`** / `200.52` | **采纳**；SwiGLU `789.4 → 701.0 µs`（**`-11.2%`**），C=8 GPU 降至 `4,661.3 ms` |
| **R16** | 固定 `K=3` 在 C=8 并发仍存在多余 Draft 轮次 | [apply_r16_adaptive_mtp.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r16_adaptive_mtp.py)（动态 MTP 深度 `LLAMA_MTP_ADAPTIVE=1`） | **`62.65`** / **`74.51`** | `2,817.70` | `115.89` / `191.81` | **采纳（配合 R17 锁定 `K=3/2/2`）**；低并发保留 `K=3` 爆发力，高并发自动降深减压 |
| **R17** | 探索 `mmq-few` `min_blocks=3/4` 与 `q4k_stages=3` 寄存器/流水 | [apply_r17_mmq_few_thor_tuning.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r17_mmq_few_thor_tuning.py) | `61.30` / **`74.13`** | `2,802.88` | `127.85` / **`209.15`** | **调参核驳回，保留 `mb1_s2 + adapt322`**；`mb3/4` 与 `s3` 增加寄存器压力反慢，锁定 `STACK:0` + C=8 `209.15 tok/s` |
| **R18** | 同层兄弟投影（Q/K/V, Gate/Up）重复量化 `Q8_1` / 转 `FP16`；`binbcast` 逐标量 | [apply_r18_sibling_act_cache.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r18_sibling_act_cache.py)（跨兄弟激活缓存 + `k_bin_cont_f32_vec4` + 融合节点失效修复） | **`63.63`** / `71.99` | **`2,879.41`**（峰 `2,922.92`） | `120.81` / `201.66` | **采纳（含融合节点失效修复）**；修复初始 `nodes_to_skip!=0` 失效漏判后 `acc` 恢复 `46.2%`（6/6 PASS），C=1 GPU 创最低 `4,912 ms`，C=8 GPU `3,953 ms` |
| **R19** | 验证异步双缓冲流反量化与 2CTA cuBLAS 重叠 | [apply_r19_async_dequant_overlap.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_r19_async_dequant_overlap.py) | `59.91` / `70.58` | `2,811.63` | `118.02` / `195.65` | **默认关闭 (`ASYNC_DEQUANT=0`)**；Thor 仅 20 SM，异步反量化与 2CTA cuBLAS 争抢 L2/DRAM 带宽致 PF4K 反降 `-68 tok/s` |
| **R20** | Thor 最终 `v0.2.0` `FROM scratch` 精简容器构建、审计与离线双跑验收 | `build_slim_thor_runtime.py --target thor` + `--network none` + 2 轮无剖析纯容器基准 + `r20_ncu_sol.csv` | **`60.92`**（Run 1 `59.78`，峰 **`70.49`**，思考 **`65.70`**） | **`2,874.28`**（`srv=2,894.53`, `1,434 ms`） | **`133.46`** / **`198.98`** | **全量通过**：`572.6 MiB`、`25` 文件、`violations: []`、断网 `12s` 就绪、两轮均 `6/6` + `4/4` §19 全过 |

---

## 4. AGX Orin 64GB（`sm_87`, JetPack 7）5 轮硬件定向优化（O0–O5）

依据 `model-host-protocol.md` **§20 AGX Orin 64GB 硬件约束**（16 SM、**4 MB L2**、204.8 GB/s LPDDR5、61.4 GiB UMA、12 核 Cortex-A78AE、无硬件 PDL、无 FP8 KV），在实机 `192.168.1.86` 执行 5 轮闭环优化：

| 轮次 | 剖析发现的 Orin `sm_87` 专属瓶颈 | 实施优化与补丁 | C=1 中位 / 峰值 / 思考 | 4K Prefill (TTFT) | C=4 / C=8 | 决策与核心 `nsys`/`ncu` 证据 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **O0** | 基线：`sm_87` 编译、无 MTP、`f16` KV、`-ub 512`、默认线程 | 建立 Orin `o0_nsys`（3 窗口）+ `o0_ncu_sol.csv`（`--cache-control none`, 60 核） | `21.93` / `21.94` / `21.92` | `847.41` (`4,863 ms`) | `74.34` / `125.65` | `o0_prefill4k` 暴露：`dequantize_block_q4_K<__half>` 占 **`1,280.1 ms` (`25.1%`)**、标量 `gated_delta_net_cuda` 占 **`436.5 ms` (`8.6%`)**（上游漏掉 `GGML_CUDA_CC_ORIN`！） |
| **O1** | 未开 MTP；`f16` KV 占压 61 GiB UMA；`-ub 512` 触发 1,926 次反量化；默认 12 线程与 `PDL=1` | 启用 `Q2_K` MTP + `q8_0` KV + `OMP_NUM_THREADS=4 -t 4 -tb 8 --poll 100` + `GGML_CUDA_PDL=0` + `-b 4096 -ub 4096` | **`41.07`** / `45.71` / `44.26` | `947.23` (`4,374 ms`) | `89.42` / `124.24` | **保留**；单流暴涨 **`+87.3%`**，PF4K `+11.8%` |
| **O2** | 上游 `gated-delta-net-mma.cu` 仅允许 `GA10X (860)` 漏掉 Orin `870`；2 GB BF16 `lm_head` 冲刷 4 MB L2；`mmq-few` 未对 Orin 开 `min_ncols=2` | [apply_orin_sm87_opt.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_orin_sm87_opt.py)：解锁 `GGML_CUDA_CC_ORIN (870)` GDN TC MMA、`mmf __ldcs` 保护 4 MB L2、`mmq-few min_ncols=2`（**刻意关闭 Thor 全行 L2 预取**）+ `fattn` 整瓦片；A/B 测试 `F32_OUT=0/1` | **`42.01`** / **`50.20`** / `44.59` | **`1,016.45`** (`4,044 ms`, `F32_OUT=1`) | `91.76` / `124.99` | **采纳**；`F32_OUT=1` 直接 FP32 输出反超 `F32_OUT=0` (`975.73`) **`+40.7 tok/s`** |
| **O3** | 标定 Orin 16-SM 下 `MMF_MAX_BATCH=8`、`MMQ_MAX_BATCH=128/256` 及动态 MTP 深度；采集 `o3_nsys` | 参数网格扫描 + `o3_nsys` 三窗口剖析 | `41.84` / `48.57` / `44.30` | `967.77` (`4,282 ms`) | **`93.99`** / **`130.13`** (`K=222`) | **`o3_nsys` 实证**：GDN MMA **`436.5 → 164.1 ms`（`2.66×`）**、`dequantize_q4_K` **`1,280.1 → 155.2 ms`（`8.25×`）**、BF16 `lm_head` `__ldcs` 达 **`184.4 GB/s`（`90.0%` 峰值）**；并精准揪出 `Q2_K` Draft Head 仅 `130 GB/s` (`2,573 µs`) |
| **O4** | `MMVQ_PARAMETERS_ORIN` 对 `Q4_K`/`Q6_K` 启用 `return nwarps` (1 row/warp, 零 `__syncthreads`)，却漏掉 `Q2_K`/`Q3_K`/`MXFP4`；验证 4 MB L2 `-ub 1024/2048/4096` 与 Ampere 275 TOPS INT8 `MMQ` Prefill | [apply_orin_o4_mmvq_tuning.py](file:///home/andy/lzc-aipod-qwen3.5-9b/patches/apply_orin_o4_mmvq_tuning.py) + 8 组全矩阵对比 + `o4_nsys` | `38.51` / **`53.94`** / **`48.02`** | **`1,027.26`** (`4,025 ms`, `ub=4096`) | `90.20` / **`122.63`** (`K=321`) | **采纳 `Q2_K` 1-row/warp + `F32_OUT=1` + `ub=4096` + `K=3/2/1`**；`o4_nsys` 实证 `Q2_K` Draft Head **`2,572.6 → 1,981.7 µs`（`-23.0%`，带宽 `63.5% → 82.4%`）**；淘汰 INT8 `MMQ` Prefill（`820–830 tok/s`）与 `ub=1024`（`933 tok/s`） |
| **O5** | Orin 最终 `v0.2.0` `FROM scratch` 精简容器构建、断网验收、2 轮纯容器复测、128K/256K UMA 门禁与最终 `ncu` | `build_slim_thor_runtime.py --target orin` + `--network none` + `slim_run1/2` + `gates_128k` + `needle_256k` + `o5_ncu_sol.csv` | **`43.23`**（Run 2 `42.80`，峰 **`50.57`**，思考 **`43.89`**） | **`1,020.48`**（`srv=1,025.74`, `4,026 ms`） | **`94.92`** / **`123.85`** | **全量通过**：`571.9 MiB`、`29` 文件、`violations: []`、断网 `14s` 就绪、110K & 209K 捞针 100% 召回（256K 仍余 `29.4 GiB` 内存）、`ncu` `GDN` **`56.19 → 28.64 µs`（`-49.0%`）** 且 0 寄存器溢出 |

---

## 5. 被证伪 / 拒绝的实验留痕（双平台）

1. **Thor R7a `MXFP4` 2×-nwarps**：`2,723 → 3,634 µs`（`+33.4%`）→ 回退改用 `rows_per_block=2`（`2,304 µs`）。
2. **Thor R11 Target BF16 `lm_head` 128-bit `mmvf` / cuBLAS**：`8,120 → 10,769 µs`（`+32.6%`）→ 保留 R7b `mul_mat_f __ldcs`（92–96% 带宽极限）。
3. **Thor R12 `Q2_0` Draft Head**：无超块缩放致接受率跌至 `35.0%`（C=1 跌至 `46.6 tok/s`）→ 淘汰，锁定 `Q2_K`。
4. **Thor R14 `F32_OUT=0` (`acc16`)**：触发 `134.9 ms` `half→float` 转换致净减速 → 切换 `GGML_CUDA_F16_PREFER_F32_OUT=1`。
5. **Thor R17 `mmq-few` `min_blocks=3/4` & `q4k_stages=3`**：引发寄存器压力与吞吐退化 → 恢复 `mb1_s2`（`STACK:0`）。
6. **Thor R19 `GGML_CUDA_ASYNC_DEQUANT=1`**：20-SM 上与 2CTA cuBLAS 争抢 L2/DRAM 带宽致 PF4K 反降 `-68 tok/s` → 保留环境变量开关，默认 `0`。
7. **Orin O4 Ampere INT8 `MMQ` Prefill (`GGML_CUDA_MMQ_MAX_BATCH=4096`)**：4K Prefill 仅 `820–830 tok/s`，远低于 cuBLAS FP16 (`1,020–1,027 tok/s`) → 淘汰，保持 `GGML_CUDA_MMQ_MAX_BATCH=256`。
8. **Orin O4 `-ub 1024` 微批缩减**：增加反量化启动次数致 PF4K 降至 `933 tok/s` → 淘汰，锁定 `-ub 4096` + `F32_OUT=1`。

---

## 6. 最终双源发布清单与匿名 `HTTP 206` 验链回执（`v0.2.0`）

| 产物类别 | 提供方 | 仓库 | 40 位固定提交 | 精确大小 (Bytes) | SHA256 | 匿名 `HTTP 206` |
| :--- | :--- | :--- | :--- | ---: | :--- | :---: |
| **主权重 `Q4_K_M`** | Hugging Face | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` | `4d47954f2ef8277567a20508dd3f88def5404455` | `6,979,975,392` | `d7eb4fac9389d53fa576f64a6ff53e914a00bc7705dc354d1065887565147320` | **PASS** |
| **主权重 `Q4_K_M`** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF` | `d11804e0df7f71b43d8054e1f96622067a0ed03a` | `6,979,975,392` | `d7eb4fac9389d53fa576f64a6ff53e914a00bc7705dc354d1065887565147320` | **PASS** |
| **视觉投影 `mmproj-F16`** | Hugging Face | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` | `4d47954f2ef8277567a20508dd3f88def5404455` | `918,166,080` | `f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f` | **PASS** |
| **视觉投影 `mmproj-F16`** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-GGUF` | `d11804e0df7f71b43d8054e1f96622067a0ed03a` | `918,166,080` | `f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f` | **PASS** |
| **Thor 精简镜像 `0.2.0`** | Hugging Face | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `d39501b59132d4fc73f4c0b0f1a1d85ebf3c1f43` | `600,403,959` | `c38e56c8f1c599ee3ee696adc08d5f8ebef69ff88f3d88241a3c692d6dfdb248` | **PASS** |
| **Thor 精简镜像 `0.2.0`** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `d5824bd7a26f7a7aa38e257ec92cc65086014d40` | `600,403,959` | `c38e56c8f1c599ee3ee696adc08d5f8ebef69ff88f3d88241a3c692d6dfdb248` | **PASS** |
| **Orin 精简镜像 `0.2.0`** | Hugging Face | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `d39501b59132d4fc73f4c0b0f1a1d85ebf3c1f43` | `599,629,994` | `d282e0aa35feb8e60584ddc73a4f209d058667ed617a4dc85bd2e7b4b5a7cc3a` | **PASS** |
| **Orin 精简镜像 `0.2.0`** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `d5824bd7a26f7a7aa38e257ec92cc65086014d40` | `599,629,994` | `d282e0aa35feb8e60584ddc73a4f209d058667ed617a4dc85bd2e7b4b5a7cc3a` | **PASS** |
| **控制端归档包 (`amd64`)** | Hugging Face | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `4b34f41cba8dcb53315b38ec2bceceafb9f72d61` | `3,050,983` | `2146a1608e7c2cb664ff277f592e6b0f0ad56dc8ce594034a15355ee796c4117` | **PASS** |
| **控制端归档包 (`amd64`)** | ModelScope | `manateelazycat/Qwen3.5-9B-Defiant-Fable-Runtime` | `611db9468fc505d5b8aa97f70564d4466e8d8348` | `3,050,983` | `2146a1608e7c2cb664ff277f592e6b0f0ad56dc8ce594034a15355ee796c4117` | **PASS** |
| **最终薄壳 LPK (`v0.2.0`)** | 本地 & 双源发布 | `dist/cloud.lazycat.aipod.qwen35-9b-defiant-fable-v0.2.0.lpk` | `v0.2.0` | `52 KiB` | `e6f40488cd93f769fa10b1b5fe59f520928101c8109973b321fb4d248fddd4b5` | **PASS** |

- **Thor Docker IDs**：`imageId = sha256:d36d871a1ac946ed4e94a01fea8932b33502a4cddd7d8e3fa96374a84ebc0ddf`，`configImageId = sha256:46a4bd2f3dc2388896f6e43040fca37b6e1352049dc8e4278c077f9ef0b6f78f`
- **Orin Docker IDs**：`imageId = sha256:66ed94b476f9cdd217766f0a7324c1ef43941e096523129c61650c275a3dd790`，`configImageId = sha256:9187de5ee0f955f63bdf54cd7dd78cbf62723cac023279fc881a6b61aa74a84f`
