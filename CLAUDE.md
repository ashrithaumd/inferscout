# CLAUDE.md: InferScout

Context for Claude Code sessions in this repo.

## What this project is

InferScout recommends the most cost-effective way to serve an open LLM under **latency, budget, AND quality** limits, then validates that recommendation against real benchmarks. The owner is building it to learn GPUs, PyTorch, and LLM inference optimization **from scratch**. It is a public portfolio project.

Full plan: [docs/project-plan.md](docs/project-plan.md). Learning log: [docs/learning-log.md](docs/learning-log.md).

### Differentiators (vs. BentoML llm-optimizer, NVIDIA AIConfigurator)
1. **Quality-aware:** each quantization setting is evaluated on the user's own task, and configs that lose too much accuracy are rejected.
2. **Budget-aware search:** a cheap predictor rules out hopeless configs, then successive halving spends real GPU minutes only on promising ones.
3. **Hardware fingerprint in PyTorch:** microbenchmarks measure real matmul throughput and memory bandwidth. A roofline model uses them to predict latency, and prediction error is reported honestly.
4. **Bottleneck explanation:** every recommendation explains WHY (memory-bound decode, KV-cache pressure, …), using the PyTorch profiler, vLLM metrics, and nvidia-smi.
5. **Self-host vs. API break-even:** the request volume at which self-hosting becomes cheaper than a paid API.

Stretch: realistic traffic replay, CI regression check with GitHub Actions.

### Scope
- GPUs: T4 (free Colab/Kaggle) for development, L4 (rented) for final validation.
- Models: 0.5B–3B open models (Qwen, Llama 3.2) plus AWQ/GPTQ versions.
- **T4 has no bf16. Use fp16 on T4.**

### Phases
| Phase | Name | Content |
|---|---|---|
| 0 | Hardware fingerprint | PyTorch microbenchmarks: matmul throughput, memory bandwidth, GPU memory |
| 1 | Benchmark harness | vLLM × concurrency × quantization; workloads: chat, long-prompt RAG, batch; TTFT, inter-token latency, p50/p95, throughput, peak GPU memory |
| 2 | Quality eval | Accuracy for each quantization setting |
| 3 | Predictor | Roofline model from the fingerprint; prediction error vs. measured |
| 4 | Recommender | Predictor pruning + successive halving, constraints, bottleneck explanations, break-even vs. API |
| 5 | Validation + write-up | Rented L4, T4 vs. L4 comparison, published write-up |

Phase statuses are kept in the README roadmap table. Update them as phases progress.

## Budget rule (hard)

- **The total project spend must stay under $30.** About $20 is reserved for the final L4 validation.
- Development uses **free** Colab/Kaggle T4 only.
- **Never suggest a paid service, GPU rental, or API without first flagging its estimated cost** and how it fits the remaining budget. Get an explicit OK before anything that costs money.
- Log every paid session in the README budget tracker (date / provider / GPU / hours / cost).

## How to teach me

- I know nothing about GPUs yet. Teach from the basics, **one small piece at a time**. Don't overwhelm me.
- For each concept, give **a simple metaphor PLUS a little real technical explanation**.
- **Before each experiment, predict** what will happen and why. **After it**, help me read the numbers.
- Always connect back to **"why did I do this?"** for interview prep.
- **Go sequentially. Stop after each step and wait for me to say "next".**
- **Don't build ahead of the current phase.**
- After each experiment, help me add an entry to `docs/learning-log.md`.

## Repo layout

```
inferscout/{fingerprint,harness,predictor,recommender}/   package, one subpackage per phase area
experiments/phase0/results/                              experiment outputs (keep large dumps out of git)
tests/
docs/                                                    project-plan.md, learning-log.md
```
