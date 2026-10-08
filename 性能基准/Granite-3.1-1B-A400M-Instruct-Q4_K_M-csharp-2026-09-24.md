# Granite-3.1-1B-A400M-Instruct Q4_K_M：纯 C# 首次对照

## 等价短测：零提示词、10-token decode

| 项目 | 纯 C# 候选 | 同日 llama.cpp 重跑 | 保留的旧基线 |
|---|---:|---:|---:|
| 测量日期 | 2026-09-24 | 2026-09-24 | 2026-09-23 |
| 解码吞吐 | **16.14 ± 0.35 token/s** | **19.61 ± 0.76 token/s** | **16.63 ± 0.32 token/s** |
| 平均 10-token 解码耗时 | 619.8 ms | 未记录 | 未记录 |
| 重复次数 | 5 | 5 | 5 |
| 提示词 / 生成 token | 0 / 10 | 0 / 10 | 0 / 10 |
| CPU 线程 / GPU | 6 / 关闭 | 6 / 关闭 | 6 / 关闭 |

候选均值为同日 llama.cpp 重跑的 82.3%。同一固定 reference 在 9 月 23 日的原始均值是 16.63 ± 0.32 token/s，同日重跑为 19.61 ± 0.76；原记录保持不变，这种差异说明短测对运行条件较敏感。每轮只生成 10 个 token，不能据此判断更长生成时的稳定差异。该 `-p 0` 工作负载只比较短时 decode，不包含真实提示词处理、首 token、模型加载、服务传输或长上下文容量。

### 候选环境与模型

- 主机：Windows 10 Pro build 19045，Intel Core i5-10500（6 核 / 12 逻辑处理器），48 GiB RAM；CPU-only。
- 构建：.NET SDK 10.0.302，`Release`；托管 CPU AVX2 backend，6 个推理线程。源码当时尚未提交，候选为本地工作树。
- 模型：`D:\github.com\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf`，GGUF Q4_K_M，821,847,360 字节。
- SHA-256：`3A2EC1C2A78CB29D901E29BBF5162DCD03381E13803D2CBDCFF838D4D08142EB`，与 reference 相同。
- 模型加载耗时：约 0.09 秒，不计入 decode 吞吐；10-token warm-up 耗时 1,199.7 ms，也不计入五轮重复结果。
- 候选峰值记录工作集：600.3 MiB。此值来自短测进程采样，不代表最大上下文内存。

### 候选逐轮结果

| 轮次 | 吞吐 | 10-token 解码耗时 |
|---|---:|---:|
| 1 | 15.64 token/s | 639.4 ms |
| 2 | 15.99 token/s | 625.4 ms |
| 3 | 16.21 token/s | 617.0 ms |
| 4 | 16.27 token/s | 614.5 ms |
| 5 | 16.59 token/s | 602.7 ms |

### 复跑命令

在 `D:\github.com\llm-stupid\src` 的 PowerShell 中运行：

```powershell
dotnet run --no-build --no-launch-profile --project .\LlmStupid.Tool -c Release -- bench D:\github.com\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf --repetitions 5 --tokens 10 --threads 6 --context 128
```

程序会在开始测量前校验模型 SHA-256；它先执行一轮完整 10-token warm-up，再执行 5 次独立的零提示词、10-token decode。同日复跑固定 reference 使用：

```powershell
& 'D:\github.com\llama.cpp\build-cpu-release\bin\llama-bench.exe' -m 'D:\github.com\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf' -p 0 -n 10 -r 5 -t 6 -ngl 0
```

同日重跑的 reference 是 `llama-bench 0.4.1-dev`（build 1、commit `42916d8`），GNU 16.1.0 / Windows AMD64；原始构建开关和源码信息保留在 [llama.cpp 基线记录](Granite-3.1-1B-A400M-Instruct-Q4_K_M-llama.cpp-2026-09-23.md)，没有覆盖旧结果。运行前核对该版本并检查模型 SHA-256。

### C# 线程扩展测量

下表使用相同零提示词、10-token 工作负载，均为 5 次重复；不同线程数不与固定 6 线程的 llama.cpp reference 计算比率。

| C# 线程 | 解码吞吐 | 平均 10-token 解码耗时 |
|---:|---:|---:|
| 6 | 16.14 ± 0.35 token/s | 619.8 ms |
| 8 | 17.87 ± 0.47 token/s | 560.1 ms |
| 10 | 18.13 ± 0.42 token/s | 552.0 ms |
| 12 | 18.21 ± 0.53 token/s | 549.5 ms |

线程扩展命令与 6 线程结果使用同一个 `bench` 入口，仅改变线程数：

```powershell
$model = 'D:\github.com\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf'
foreach ($threads in @(8, 10, 12)) {
    dotnet run --no-build --no-launch-profile --project .\LlmStupid.Tool -c Release -- bench $model --repetitions 5 --tokens 10 --threads $threads --context 128
}
```

12 线程的零提示词结果仅比 10 线程高 0.4%，差异小于重复波动。相同真实请求也分别重复 3 次：

| C# 线程 | 45-token 提示处理时间（3 次） | 均值 | 10-token decode 吞吐（3 次） | 均值 |
|---:|---|---:|---|---:|
| 10 | 3,151.5 / 3,013.1 / 2,995.3 ms | 3,053.3 ms | 19.99 / 20.00 / 19.75 token/s | 19.91 token/s |
| 12 | 3,300.5 / 3,165.6 / 3,213.4 ms | 3,226.5 ms | 20.03 / 20.05 / 19.17 token/s | 19.75 token/s |

10 线程的真实请求平均提示处理更快，decode 均值也相近。引擎和服务因此默认最多 10 线程，服务可通过 `LLM_THREADS` 覆盖；该默认侧重真实请求响应时间，不代表所有模型或 CPU 的普遍最优值。

## 单独的真实提示词生成

以下为另一项单次真实对话测量，不与 `-p 0` decode 吞吐混合：

| 指标 | 结果 |
|---|---:|
| 用户提示 | `Say hello.` |
| 编码后提示 token 数 | 45 |
| 提示词处理 | 3,151.5 ms，14.28 token/s |
| 首 token 延迟 | 3,152.5 ms |
| 生成 | 10 token，500.2 ms，19.99 token/s |
| 工作集 | 837.1 MiB |
| 输出 | `Hello! How can I assist you today?` |

复跑命令（在 `src` 目录；`$model` 指向相同 GGUF）：

```powershell
dotnet run --no-build --project .\LlmStupid.Tool -c Release -- generate $model --prompt 'Say hello.' --tokens 10 --threads 10 --context 256 --temperature 0
```

该对话行是 10 线程三轮中的第一轮，模型加载 81 ms；12 线程三个提示处理时间为 3,300.5、3,165.6、3,213.4 ms，逐 token 吞吐分别为 20.03、20.05、19.17。所有对话测量都使用同一 prompt 和 `temperature=0`，提示词处理仍逐 token 执行，首 token 延迟主要受该路径影响。
