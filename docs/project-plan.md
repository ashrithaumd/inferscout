# InferScout: Project Plan

## Goal

InferScout recommends the most cost-effective way to serve an open LLM under **latency, budget, and quality** limits. It then validates that recommendation against real benchmarks.

It is also a learning project. The aim is to learn GPUs, PyTorch, and LLM inference optimization from scratch, and to build a portfolio piece that shows that understanding with honest, measured results.

## Problem

Serving an LLM means choosing among many options: GPU type, model size, quantization (fp16, AWQ, GPTQ, …), batch size and concurrency, context length, and serving engine settings. Each choice trades off latency, throughput, cost, and output quality.

Existing tools already search part of this space:

- **BentoML llm-optimizer** searches serving configs against latency and throughput targets.
- **NVIDIA AIConfigurator** recommends configs for NVIDIA serving stacks.

These tools mostly optimize for speed. They don't tell you whether a faster, more heavily quantized config still gives good answers on *your* task, and they don't help when your GPU budget is tiny.

## What makes InferScout different

### 1. Quality-aware recommendations
Each quantization setting is evaluated on the user's own task, for example a small labeled eval set. Configs whose accuracy drops more than an allowed amount below the fp16 baseline are rejected, no matter how fast they are.

### 2. Budget-aware search
Benchmarking every config on a real GPU costs money. InferScout:
1. Uses a cheap predictor (Phase 3) to rule out configs that can't meet the constraints.
2. Runs **successive halving** on the rest: give every candidate a short, cheap benchmark, keep the best fraction, give the survivors a longer benchmark, and repeat.

Real GPU minutes are spent only on promising configs.

### 3. Hardware fingerprint in PyTorch
PyTorch microbenchmarks measure what a specific GPU *actually* delivers, rather than trusting spec-sheet numbers:
- Matmul throughput (achieved TFLOPS) for each dtype
- Memory bandwidth (achieved GB/s)
- Usable GPU memory

A **roofline model** uses these measured numbers to predict latency. Prediction error against measured results is reported openly, including where the model is wrong.

### 4. Bottleneck explanation
Every recommendation explains *why* it was chosen, for example:
- Decode is memory-bandwidth-bound, so quantizing the weights helps.
- KV-cache pressure limits concurrency at long context.
- Prefill is compute-bound for long prompts.

Evidence comes from the PyTorch profiler, vLLM metrics, and `nvidia-smi`.

### 5. Self-host vs. API break-even
InferScout compares the cost per request of the recommended self-hosted config with paid API pricing, and reports the request volume at which self-hosting becomes cheaper.

### Stretch goals
- **Realistic traffic replay:** replay request patterns with bursty arrivals and mixed prompt lengths instead of a fixed load.
- **CI regression check:** a GitHub Actions job that flags performance or quality regressions.

## Scope

- **GPUs:** NVIDIA T4 (free Colab/Kaggle) and L4 (rented for final validation).
- **Models:** small open models, 0.5B–3B parameters (such as Qwen and Llama 3.2), plus AWQ/GPTQ quantized versions.
- **Serving engine:** vLLM.
- **Precision note:** T4 has no bf16 support, so use **fp16** on T4.

## Phases

### Phase 0: Hardware fingerprint
- PyTorch microbenchmarks for matmul throughput at several sizes and dtypes.
- Memory bandwidth benchmark, using large tensor copies.
- Record total and usable GPU memory.
- Compare measured numbers with the spec sheet.
- **Output:** a fingerprint file per GPU, used by the Phase 3 predictor.

### Phase 1: Benchmark harness
- Run vLLM across model × quantization × concurrency.
- Workload profiles:
  - **Chat:** short prompts, medium outputs
  - **Long-prompt RAG:** long prompts, short outputs
  - **Batch:** many requests, throughput-oriented
- Metrics: time to first token (TTFT), inter-token latency, p50/p95 latency, throughput (tokens/s), peak GPU memory.
- **Output:** a reproducible benchmark script and results in a structured format such as CSV.

### Phase 2: Quality eval
- Pick a small, task-relevant eval set.
- Measure accuracy for each quantization setting against the fp16 baseline.
- **Output:** a quality score per config, used as a constraint by the recommender.

### Phase 3: Predictor
- Roofline model: predict prefill and decode latency from model size, sequence lengths, batch size, and the Phase 0 fingerprint.
- Compare predictions with Phase 1 measurements and report the error.
- **Output:** a predictor with a documented error.

### Phase 4: Recommender
- Inputs: model, workload, latency SLO, budget, minimum quality.
- Prune with the predictor, then run successive halving on real benchmarks.
- Explain the bottleneck behind each recommendation.
- Compute break-even against API pricing.
- **Output:** a recommendation with a justification.

### Phase 5: Validation + write-up
- Validate the recommendations on a rented L4.
- Compare T4 and L4.
- Publish a write-up with results, prediction error, and lessons learned.

## Budget

| Item | Amount |
|---|---|
| Total hard cap | **< $30** |
| Development (free Colab/Kaggle T4) | $0 |
| Final L4 validation | ~$20 reserved |
| Buffer | ~$10 |

Log every paid GPU session in the budget tracker in the [README](../README.md).
