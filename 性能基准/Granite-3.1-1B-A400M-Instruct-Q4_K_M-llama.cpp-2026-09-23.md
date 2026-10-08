# Granite-3.1-1B-A400M-Instruct Q4_K_M：llama.cpp CPU 基线

## 结果

| 项目 | 值 |
|---|---|
| 测量日期 | 2026-09-23 |
| 指标 | `tg10` 解码吞吐 |
| 结果 | **16.63 ± 0.32 token/s**（5 次重复，工具输出的均值 ± 标准差） |
| 提示词 token | 0（`-p 0`） |
| 生成 token | 10（`-n 10`） |
| 线程 | 6 |
| GPU offload | 关闭（`-ngl 0`） |

这是一个短时解码基线。它没有测量真实提示词处理、模型加载、首 token 延迟、端到端对话或最大可用 token 数；后续结果必须明确区分这些指标。

## 测试对象与主机

- 模型：[`bartowski/granite-3.1-1b-a400m-instruct-GGUF`](https://huggingface.co/bartowski/granite-3.1-1b-a400m-instruct-GGUF) 中的 `granite-3.1-1b-a400m-instruct-Q4_K_M.gguf`，GGUF 量化为 `Q4_K_M`。
- 文件位置：仓库父目录的 `models/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf`。
- 文件大小：821,847,360 字节。
- SHA-256：`3A2EC1C2A78CB29D901E29BBF5162DCD03381E13803D2CBDCFF838D4D08142EB`。比较前应重新计算并确认一致。
- 主机：Windows 10 Pro，build 19045；Intel Core i5-10500，6 核 / 12 逻辑处理器，48 GiB 内存。测试使用 CPU。

## llama.cpp reference

- 上游：[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，本次源码提交 `42916d83f4a225e56709f873aa8050ac11f5b6a4`，测量时工作树干净。
- 对应的 [llama-bench 文档](https://github.com/ggml-org/llama.cpp/blob/42916d83f4a225e56709f873aa8050ac11f5b6a4/tools/llama-bench/README.md)。
- 构建：Windows x64 Release、MinGW-w64 GCC 16.1.0、`GGML_NATIVE=ON`、OpenMP 开启；没有启用 GPU backend。
- 工具版本：`llama-bench 0.4.1-dev (build 1, commit 42916d8)`。
- reference checkout 相对本仓库位于 `../llama.cpp`；二进制位于 `../llama.cpp/build-cpu-release/bin/llama-bench.exe`。

## 在当前 Windows 布局复跑

从本仓库根目录打开 PowerShell，先确认模型文件和 reference 二进制存在，再运行同一组参数：

```powershell
$root = (Resolve-Path .).Path
$bench = Join-Path $root '..\llama.cpp\build-cpu-release\bin\llama-bench.exe'
$model = Join-Path $root '..\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf'
if (!(Test-Path $bench) -or !(Test-Path $model)) { throw '找不到 reference 二进制或模型文件' }
$version = (& $bench --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $version -notmatch '0\.4\.1-dev.*commit 42916d8') { throw "reference 版本不匹配：$version" }
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $model).Hash
if ($hash -ne '3A2EC1C2A78CB29D901E29BBF5162DCD03381E13803D2CBDCFF838D4D08142EB') { throw "模型 SHA-256 不匹配：$hash" }
$hash
& $bench -m $model -p 0 -n 10 -r 5 -t 6 -ngl 0
```

若 reference 二进制缺失，不要用当前最新版悄悄替换此基线；需按记录的提交和构建条件重新构建，并把新构建的结果作为独立记录保存。不同机器、构建或模型文件的结果应注明差异，不能直接视为同一组数据。
