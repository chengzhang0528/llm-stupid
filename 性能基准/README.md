# 模型性能基准

这里保存本地模型推理的可复查性能基线和后续比较结果，供纯 C# 推理实现评估时使用。当前 reference 是固定版本的 llama.cpp；模型与主机信息、测量范围及复跑命令见 [llama.cpp 基线](Granite-3.1-1B-A400M-Instruct-Q4_K_M-llama.cpp-2026-09-23.md)，首个纯 C# 引擎的同机结果见 [C# 对照记录](Granite-3.1-1B-A400M-Instruct-Q4_K_M-csharp-2026-09-24.md)。Q4_K 四行 AVX2 批处理实验见 [C# 优化对照记录](Granite-3.1-1B-A400M-Instruct-Q4_K_M-csharp-q4-row-tile-2026-09-24.md)，超过 30 token/s 的 C# 优化组合与复测见 [30 token/s 优化记录](Granite-3.1-1B-A400M-Instruct-Q4_K_M-csharp-over-30tok-2026-09-24.md)。

推理流程图：[SVG 版](Granite-token-generation.svg)适合独立查看和嵌入，包含输入编码、Prefill、24 层 Decode、KV Cache、采样、停止判断、流式输出和生成回环，并随系统浅色/深色偏好切换；[交互 HTML 版](Granite-token-generation.html)另有单层 RMSNorm、GQA、MoE Top-8/32、SwiGLU 与残差的说明卡。

当前数据只代表 CPU 上生成 10 个 token 的短测解码速度，不代表提示词处理、首 token 延迟、完整对话吞吐、峰值内存或模型最大上下文。比较时保持 GGUF 文件校验和、线程数、计算设备、构建模式和测试参数一致；不能匹配的维度要分开报告，不要合成一个“性能”数字。

每次新的实现或配置比较使用独立的日期记录，保留基线原文；同时记录主机、模型 SHA-256、引擎版本/提交、构建方式、运行参数、重复次数及各项指标。仓库只保存整理后的测量数据，不保存模型、二进制或原始运行日志。
