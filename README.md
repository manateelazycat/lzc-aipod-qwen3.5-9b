# Qwen3.5-9B-The-Defiant-Fable — LZC AI Pod Thin-LPK (`AGX Thor + AGX Orin`, `v0.2.0`)

[![License: Apache-2.0](https://img.shields.io/badge/Model_License-Apache--2.0-blue.svg)](docs/source-and-license.md)

将 `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF` 按懒猫算力舱 `model-host-protocol.md`（含 §18 Thor、§19 长上下文稳定性与 §20 AGX Orin 64GB）完整移植至 **NVIDIA Jetson AGX Thor (`sm_110a`)** 与 **NVIDIA Jetson AGX Orin 64GB (`sm_87`, JetPack 7)** 双平台，并分别完成 **Thor 20 轮（R0–R20）** 与 **Orin 5 轮（O0–O5）** Nsight Systems + Nsight Compute 硬件级性能剖析与深度优化。

## 核心亮点（`v0.2.0` 双平台）

- **薄壳 LPK（`52 KiB`）+ 双目标 `FROM scratch` 精简运行时**：
  - **AGX Thor (`sm_110a`)**：25 文件、`572.6 MiB` `.tar.gz`，断网（`--network none` + `/model:ro`）**`12 秒`** 通过健康检查。
  - **AGX Orin 64GB (`sm_87`, JetPack 7)**：29 文件、`571.9 MiB` `.tar.gz`，断网 **`14 秒`** 通过健康检查。
  - 双平台共享同一套锁定权重（`Q4_K_M` + `mmproj-F16.gguf`），全量 10/10 匿名 `HTTP 206` 双源验链通过。
- **AGX Thor (`sm_110a`) 20 轮优化成果（`v0.2.0` 纯精简容器）**：
  - **C=1 解码**：`32.95 → 60.92 tok/s`（宿主机 R19 `63.63`，峰值 **`70.49–74.51 tok/s`**，思考 **`65.70–72.73 tok/s`**）
  - **4K 冷 Prefill**：`1,326.99 → 2,874.28 tok/s`（**`2.17×`**，TTFT `3,079 → 1,434 ms`）
  - **C=4 / C=8 并发聚合**：`114.17 → 133.46–142.16 tok/s` / `192.92 → 198.98–209.15 tok/s`
- **AGX Orin 64GB (`sm_87`) 5 轮硬件定向优化成果（`v0.2.0` 纯精简容器）**：
  - **C=1 解码**：`21.93 → 43.23 tok/s`（**`+97.1%` / `1.97×`**，峰值 **`50.57–53.94 tok/s`**，思考 **`43.89–48.02 tok/s`**）
  - **4K 冷 Prefill**：`847.41 → 1,020.48 tok/s`（**`+20.4%`**，TTFT `4,863 → 4,026 ms`）
  - **C=4 / C=8 并发聚合**：`74.34 → 94.92 tok/s`（**`+27.7%`**）/ `123.85–133.71 tok/s`
  - **61 GiB UMA 极限验证**：110K 捞针（RSS `19.1 GiB`）与 209K/256K 捞针（RSS `25.95 GiB`，仍余 `29.4 GiB`）100% 召回，零 OOM。

## 文档索引

- [双平台（Thor R0–R20 + Orin O0–O5）Nsight 性能优化测试报告（v0.2.0 终版）](docs/thor-orin-nsight-report.md)
- [Thor 15 轮 Nsight 性能优化测试报告（R0–R15 存档）](docs/thor-nsight-15rounds-report.md)
- [Thor 首期 10 轮 Nsight 性能优化测试报告（v0.1.0 存档）](docs/thor-nsight-10rounds-report.md)
- [API 契约与服务规范](docs/api-contract.md)
- [模型来源、溯源与许可证声明](docs/source-and-license.md)
- [可复现优化补丁集合（R2–R19 + Orin O2/O4）](patches/)
- [原始评测与 Nsight 剖析数据](docs/results/)
