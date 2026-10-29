# InferScout

> **Status: early development.** The repo is scaffolded and Phase 0 has not started yet. No results so far.

InferScout recommends the cheapest way to serve an open LLM that still meets your **latency, budget, and quality** limits. It then checks that recommendation against real benchmarks. It targets small GPUs (NVIDIA T4, L4) and small open models (0.5B–3B parameters, such as Qwen and Llama 3.2, including their AWQ/GPTQ versions).

## Why InferScout?

Tools such as BentoML's llm-optimizer and NVIDIA's AIConfigurator already find fast serving configs under a latency target. InferScout adds five things:

1. **Quality-aware recommendations.** Each quantization setting is scored on *your* task. A config that loses too much accuracy is rejected, however fast it is.
2. **Budget-aware search.** A cheap predictor rules out hopeless configs first. Successive halving then spends real GPU minutes only on the promising ones.
3. **Hardware fingerprint in PyTorch.** Microbenchmarks measure a GPU's real matmul throughput and memory bandwidth. A roofline model uses those numbers to predict latency, and the prediction error is reported openly.
4. **Bottleneck explanation.** Every recommendation says *why* it was chosen (for example, memory-bound decode or KV-cache pressure), using evidence from the PyTorch profiler, vLLM metrics, and `nvidia-smi`.
5. **Self-host vs. API break-even.** The request volume at which self-hosting becomes cheaper than a paid API.

**Stretch goals:** realistic traffic replay, and a CI regression check with GitHub Actions.

## Roadmap

| Phase | Name | Description | Status |
|---|---|---|---|
| 0 | Hardware fingerprint | PyTorch microbenchmarks: matmul throughput, memory bandwidth, GPU memory | Not started |
| 1 | Benchmark harness | vLLM × concurrency × quantization across workload profiles (chat, long-prompt RAG, batch). Measures TTFT, inter-token latency, p50/p95, throughput, peak GPU memory | Not started |
| 2 | Quality eval | Accuracy for each quantization setting | Not started |
| 3 | Predictor | Roofline model built from the fingerprint, with prediction error vs. measured results | Not started |
| 4 | Recommender | Predictor pruning, successive halving, constraints, bottleneck explanations, break-even vs. API | Not started |
| 5 | Validation + write-up | Rented L4 run, T4 vs. L4 comparison, published write-up | Not started |

See [docs/project-plan.md](docs/project-plan.md) for the detailed plan.

## Results

_No results yet._

## Budget tracker

The total budget for the whole project is **under $30**. Development runs on free Colab/Kaggle T4 GPUs, and about $20 is set aside for the final L4 validation.

| Date | Provider | GPU | Hours | Cost (USD) |
|---|---|---|---|---|
| | | | | |
| **Total** | | | | **$0.00** |

## Repository layout

```
inferscout/        Python package
  fingerprint/     Phase 0: hardware microbenchmarks
  harness/         Phase 1: benchmark harness
  predictor/       Phase 3: roofline latency predictor
  recommender/     Phase 4: search and recommendation
experiments/       Per-phase experiment scripts and results
tests/             Tests
docs/              Project plan and learning log
```

## License

[MIT](LICENSE) © 2026 Ashritha
