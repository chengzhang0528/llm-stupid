# llm-stupid 项目智能体入口

Status: Active
Kind: AgentEntry
Scope: llm-stupid / 纯 C# 本地推理服务
Owner: 项目维护者
Updated: 2026-09-24
Depends On:
- ../../WORKSPACE_STRUCTURE.md
- 当前设计.md

## 项目事实

- 产品目标是纯托管 C# 本地推理，优先利用当前 CPU，最大化可用生成能力。
- 首个模型范围是 Granite-3.1-1B-A400M-Instruct 的 Q4_K_M GGUF；llama.cpp 是独立性能和行为参考，不是运行时依赖。
- 唯一源码根为 `src/`；加载目标根的 `AGENTS.md`、现有实现和测试后再改代码。
- 长期引擎边界、部署宿主职责和首个模型支持范围归 [`当前设计.md`](当前设计.md)。
- 模型性能测量按仓库 skill `model-performance-benchmark` 执行，保留既有 llama.cpp 基线并分开报告生成、提示词处理、延迟和内存。

## 约束

- 推理引擎不依赖 llama.cpp、原生推理库或模型服务进程；C# 托管实现须有可验证的标量正确性回退。
- 模型文件、构建结果、原始日志和临时基准输出不得放入仓库。
- 每个改动只扩大到当前验收需要的模型格式、算子或 API；不能暗示完整通用 GGUF 支持。
