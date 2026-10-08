# Granite-3.1-1B-A400M-Instruct Q4_K_M：Q4_K 四行 AVX2 批处理对照

## 改动

托管 AVX2 后端新增可选的量化多行计算能力。`MappedMatrix` 对 Q4_K 权重按四行分块；内核对四行复用同一组 Q8 激活向量、量化尺度和输入和。每行仍按原次序执行相同整数点积与浮点累加。Q6_K、标量后端及不实现该能力的后端继续使用原逐行路径。

## 固定短测

| 项目 | 值 |
|---|---|
| 系统 / CPU | Windows 10 Pro build 19045；Intel Core i5-10500，6 核 / 12 逻辑处理器；48 GiB RAM |
| 构建 | .NET SDK 10.0.302，Release；C# 使用 `managed-cpu-avx2` |
| 模型 | `D:\github.com\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf`，Q4_K_M，821,847,360 字节 |
| SHA-256 | `3A2EC1C2A78CB29D901E29BBF5162DCD03381E13803D2CBDCFF838D4D08142EB` |
| C# 前版 / 候选 | 同一工作区改动前 Release 输出副本 / 四行内核 Release 输出；本地源码尚未提交 |
| llama.cpp reference | `42916d83f4a225e56709f873aa8050ac11f5b6a4`，`llama-bench 0.4.1-dev`，CPU-only |
| 固定解码参数 | 零提示词、10 个生成 token、6 线程、GPU 关闭；C# warm-up 为一次完整 10-token 生成 |

模型与硬件沿用 [llama.cpp reference](Granite-3.1-1B-A400M-Instruct-Q4_K_M-llama.cpp-2026-09-23.md)。开始前校验模型 SHA-256；模型加载不计入解码吞吐。

## 五次固定短测

| 测量对象 | 吞吐 | 平均 10-token 解码时间 | 重复 |
|---|---:|---:|---:|
| 改动前 C#（原记录） | 16.14 ± 0.35 token/s | 619.8 ms | 5 |
| Q4_K 四行候选（最终校验） | **18.14 ± 0.70 token/s** | **551.9 ms** | 5 |
| 同日 llama.cpp 重跑 | 17.31 ± 0.64 token/s | 未记录 | 5 |

相对改动前 C#，本轮均值吞吐高 12.4%，平均解码时间低 11.0%；相对同日 llama.cpp 重跑高约 4.8%。这是短时 `tg10` 样本，且两个进程并非交替配对测试，因此只作为本机当前构建的方向性结果，不能代表长生成或固定提升比例。

| 测量对象 | 重复 | 吞吐（每组均值 ± 标准差） | 平均 10-token 解码时间 |
|---|---:|---:|---:|
| 改动前 C#，四组 | 每组 20 次 | 13.06 ± 2.73；15.92 ± 1.04；14.94 ± 1.57；13.40 ± 1.81 token/s | 816.0；631.0；678.4；762.9 ms |
| 四行内核 C#，四组 | 每组 20 次 | 18.60 ± 0.77；19.31 ± 0.84；16.36 ± 2.88；17.20 ± 2.13 token/s | 538.7；518.8；631.6；591.1 ms |
| 同日 llama.cpp 重跑 | 5 次 | 17.31 ± 0.64 token/s | 未记录 |

四组 C# 吞吐均值的算术平均为改动前 14.33、候选 17.87 token/s；解码时间组均值的算术平均为 722.1 ms 和 570.1 ms。测量期间机器出现间歇性慢轮，各组波动不一致，因此这些汇总用于判断优化方向，不作为固定加速比例。原有 5 次 C# 与 llama.cpp 基线记录保持不变。

10 线程候选又运行两组、每组 10 次，结果分别为 20.78 ± 1.75 和 15.05 ± 4.50 token/s；同日改动前二进制一组 10 次为 12.76 ± 3.23 token/s。10 线程结果也受到明显波动影响，不与 6 线程 llama.cpp 计算比率。短测不代表真实提示词处理、首 token 延迟、长时间生成或上下文容量。

## 正确性与资源

- Q4_K 四行批处理与四次原始 AVX2 单行点积逐位相等；数值覆盖四个量化块。
- Granite GGUF、Tokenizer 和实际单 token 推理验证通过；总计 29 项断言。
- Release solution build：0 警告、0 错误。
- 6 线程工作集约 600–601 MiB，未观察到明显内存变化。

## 复跑

从仓库根目录运行候选，重复 20 组：

```powershell
$model = 'D:\github.com\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf'
dotnet run --no-build --no-launch-profile --project .\src\LlmStupid.Tool -c Release -- bench $model --repetitions 20 --tokens 10 --threads 6 --context 128
```

改动前 Release 二进制保存在仓库外 `D:\github.com\build\llm-stupid\baseline-2026-09-24\LlmStupid.Tool.dll`，可用同一组参数运行：

```powershell
dotnet 'D:\github.com\build\llm-stupid\baseline-2026-09-24\LlmStupid.Tool.dll' bench 'D:\github.com\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf' --repetitions 20 --tokens 10 --threads 6 --context 128
```

固定 reference 的原始结果和 5 次重跑命令见 [llama.cpp 基线](Granite-3.1-1B-A400M-Instruct-Q4_K_M-llama.cpp-2026-09-23.md)。

本次同日 llama.cpp 重跑命令：

```powershell
& 'D:\github.com\llama.cpp\build-cpu-release\bin\llama-bench.exe' -m 'D:\github.com\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf' -p 0 -n 10 -r 5 -t 6 -ngl 0
```
