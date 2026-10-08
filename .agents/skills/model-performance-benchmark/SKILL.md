---
name: model-performance-benchmark
description: Use when running or comparing local LLM inference performance for this project, especially the Granite-3.1-1B-A400M-Instruct GGUF reference and future C# engine. Keep generation, prompt processing, latency, and memory results distinct.
---

# Model Performance Benchmark

Use this skill for repeatable, local model performance measurements and comparisons. Read the project [benchmark index](../../../性能基准/README.md) and [Granite llama.cpp baseline](../../../性能基准/Granite-3.1-1B-A400M-Instruct-Q4_K_M-llama.cpp-2026-09-23.md) before running a test.

## Reference workload

- Preserve the baseline as measured: Granite-3.1-1B-A400M-Instruct `Q4_K_M`, the recorded GGUF SHA-256, llama.cpp commit `42916d83f4a225e56709f873aa8050ac11f5b6a4`, CPU only, 6 threads, `-p 0 -n 10 -r 5 -t 6 -ngl 0`.
- On the recorded Windows layout, run `../llama.cpp/build-cpu-release/bin/llama-bench.exe` against `../models/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf`. Verify the model hash before comparing. The recorded result is 16.63 ± 0.32 token/s for this short `tg10` test.
- Do not silently update the model, reference commit, or build and overwrite the baseline. A refreshed reference is a separate dated measurement.
- This `-p 0` benchmark is a decode-only short test. Do not compare it as if it measured real prompt processing, first-token latency, model load time, end-to-end chat, memory use, or maximum context capacity.

## Compare an implementation

1. Confirm the candidate is a Release build and uses the same machine, GGUF bytes, CPU/GPU mode, thread count, and token-count/stop behavior as the selected reference workload. Record deviations instead of hiding them.
2. Run reference and candidate under equivalent conditions and repeat both at least five times. Keep startup/model-load and warm-up costs separate from steady decode results. If a C# runner cannot reproduce the zero-prompt synthetic workload, choose an identical real prompt for both engines and label that as a separate workload.
3. Report decode token/s, prompt-processing token/s, time to first token, total latency, and memory only when each was actually measured. Include the token counts and measurement boundaries for every metric; do not infer one metric from another.
4. Save one dated Markdown comparison under `性能基准/`, leaving the reference record intact. Include machine and build details, model path/quantization/hash, reference and candidate versions, exact commands/settings, repetitions, results with variation, and interpretation limits. Keep model files, binaries, and raw logs outside the repository.

If a required reference or candidate is missing or its identity cannot be verified, stop the comparison and report what is missing. Do not invent a result or treat a newer unpinned llama.cpp build as the recorded reference.
