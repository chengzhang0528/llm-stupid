# Granite-3.1-1B-A400M-Instruct Q4_K_M：纯 C# 超过 30 token/s 优化记录

## 结果

在本机 CPU-only Release 构建中，固定零提示词、生成 10 token、上下文 128，显式设置 10 线程，8 次完整生成预热后测量 15 次，结果为 **32.07 ± 1.82 token/s**，平均每次 10-token 解码 312.9 ms，工作集约 596.9 MiB。均值达到本任务的 >30 token/s 目标；受机器间歇性负载影响，并非每一次短测都超过 30 token/s，因此不把这组结果解释为任意负载下的保证。

同一候选的线程数扫描为：6 线程 30.08 ± 3.44 token/s，10 线程 32.07 ± 1.82 token/s，12 线程 33.26 ± 1.89 token/s。12 线程另两组各 15 次为 35.46 ± 0.33 和 34.88 ± 1.39 token/s；10 线程交错复测的候选两组为 33.07 ± 1.75 和 33.48 ± 1.74 token/s。交错组中的旧路径分别为 23.85 ± 2.32、25.10 ± 2.38 token/s，但机器负载变化显著，不能据此报出可靠的加速百分比。

固定 llama.cpp reference 是 6 线程 16.63 ± 0.32 token/s，详见 [原始基线](Granite-3.1-1B-A400M-Instruct-Q4_K_M-llama.cpp-2026-09-23.md)。这组 reference 来自前一日且测试进程不同，本记录的 10 线程结果用于确认目标档位，不能当作严格同条件 A/B 比值。

## 从性能优化案例借鉴的方法

参照 SimdPaddleOCR 的优化路径，先用性能剖析和可切换实验找热点，再为真实的数据形状和 CPU 指令集做专门路径；每次只改一类工作，单独验证数值正确性与速度，只有重复测量支持时才保留。这里应用在自回归推理中：少做重复计算、减少调度层级、提高每次任务的有效工作量，并保留标量或原有路径作为准确性对照和兼容后备。

## 保留的优化

- **缓存 Q8 激活向量的分组和。** `QuantizedInputVector.Quantize` 生成 int8 激活时同步累计每 16 个值的和；Q4_K 与 Q6_K 点积直接复用它，避免每个权重行反复扫描同一输入。Q4_K 的 32 值分块组合两个 16 值和。这减少了跨大量输出行重复做的标量工作，是本轮达到目标的主要优化。
- **扩大 Q4_K 多行 AVX2 tile。** 从四行批处理扩大到 16 行，使同一激活向量和尺度能在更多输出行间复用。tile4/8/16 都与独立逐行点积逐位校验；tile32 在重复计时中没有显示稳定收益，故未保留为默认。
- **合并相邻的调度工作。** Q/K/V 投影共用一次行任务队列；激活专家的下投影把活动专家与输出行展平到有限的并行循环，减少嵌套并行和小任务调度开销。
- **缓存 GGUF 张量数据视图。** 矩阵实例复用经过验证的张量切片视图，避免逐行重新获取数据视图和重复计算张量字节范围。
- **让预热和复测显式化。** bench 工具支持多轮 warm-up；本记录使用 8 轮预热、15 次计时，减少首次 JIT 和短样本对结果的影响。

## 固定测量条件

| 项目 | 值 |
|---|---|
| 系统 / CPU | Windows 10 Pro build 19045；Intel Core i5-10500，6 核 / 12 逻辑处理器；48 GiB RAM |
| 构建 | .NET SDK 10.0.302，Release；托管 CPU AVX2 后端，CPU-only |
| 模型 | `D:\github.com\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf`，Q4_K_M，821,847,360 字节 |
| SHA-256 | `3A2EC1C2A78CB29D901E29BBF5162DCD03381E13803D2CBDCFF838D4D08142EB` |
| 输入与输出 | 空提示词；10 个生成 token；context 128 |
| 主要结果 | 10 线程；8 warm-up 重复；15 次计时重复；32.07 ± 1.82 token/s |
| 服务默认线程上限 | 10；环境配置可覆盖。本记录显式传入线程数，不改变默认值 |

模型加载时间不计入解码吞吐。空提示词短测不代表真实 prompt 的 Prefill 时间、首 token 延迟、完整对话速度、并发服务吞吐或最大上下文。10 个 token 的窗口较短，也无法代替长时间热态测量。期间其他本机任务造成了间歇性 CPU 负载；没有停止其他任务，故把不同测量组作为有噪声的本机样本，不宣称精确加速比例。

## 正确性检查

- Release solution build：0 警告、0 错误。
- Verification：43 项断言通过，AVX2 可用；涵盖标量/AVX2 Q4 与 Q6 内核、Q4 tile4/8/16 与逐行点积等价、预计算 Q8 分组和一致性及零值重置、真实 Granite GGUF/tokenizer 和单 token 推理。
- 使用 `Say hello.` 做独立真实提示词抽样时，新旧构建都生成 `Hello! How can I assist you today?`；旧构建 26.49、新构建 35.02 token/s（各一次 10-token 样本，只作为输出一致性和端到端路径抽查，不纳入空提示词基准）。

## 复跑

在仓库根目录用当前 Release 候选重复主要 10 线程测量：

```powershell
$model = 'D:\github.com\models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf'
dotnet run --no-build --no-launch-profile --project .\src\LlmStupid.Tool -c Release -- bench $model --repetitions 15 --warmup-repetitions 8 --tokens 10 --threads 10 --context 128
```

thread scan 或其它参考参数可在同一命令中将 `--threads 10` 改为 `6` 或 `12`。要复跑 llama.cpp 固定 reference，请按其[基线记录](Granite-3.1-1B-A400M-Instruct-Q4_K_M-llama.cpp-2026-09-23.md)使用原二进制、相同模型校验和及 `-p 0 -n 10 -r 5 -t 6 -ngl 0`；不要覆盖原始基线。
