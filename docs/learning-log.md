# Learning Log

One entry per experiment. Newest at the top.

<!-- Copy this template for each new entry. -->

## YYYY-MM-DD: <Experiment name>

- **Phase:**
- **Experiment:** What I ran, on which GPU and with which settings.
- **What I predicted:** What I expected to happen, and why.
- **What happened:** The actual numbers.
- **Why:** The explanation for the result, including where my prediction was wrong.
- **Interview-ready one-liner:** One sentence I could say in an interview.

## 2026-10-29: Memory bandwidth (device-to-device copy)

- **Phase:** 0 (Hardware fingerprint), step 3
- **Experiment:** Colab T4. Copied a 512 MiB fp16 tensor GPU→GPU, counting read + write (1.07 GB per copy). Measured four ways: a cold first copy, a deliberately wrong CPU timer without sync, and the right way (5 warmups, then 20 CUDA-event timings, take the median, report min/max). Saved as `inferscout/fingerprint/bandwidth.py` + `experiments/phase0/03_bandwidth.py`.
- **What I predicted:** Wrong CPU timer = absurd (thousands+ GB/s). Right median ≈ 220–250 GB/s, 70–78% of the 320 GB/s spec. Best and median within a few %.
- **What happened:**
  - Cold first copy: **111.4 GB/s**
  - WRONG CPU timer, no sync: **9553 GB/s** (30× faster than physically possible)
  - RIGHT median: **4.43 ms → 242.2 GB/s = 76% of spec**. Min/max 4.43 / 4.48 ms (~1% spread).
- **Why:**
  - All three predictions landed. 242 GB/s is near the top of the predicted range.
  - The CPU timer only measured how long it took to *queue* the copy. GPU work is asynchronous.
  - The cold run was under half speed: the GPU was idling at low clocks and had to ramp up, and the copy kernel was loaded lazily on first use (the same lazy loading seen in step 2). One-time costs, not the steady state.
  - 76%, not 100%: DRAM refresh, read/write turnaround, and ECC check bits sharing the bus.
  - The tight min/max spread means the number is stable, and a copy doesn't push the 70 W power limit.
  - This updates the decode roofline: a 3 GB model is capped at ~80 tok/s per user, not ~106.
- **Interview-ready one-liner:** "I measured the T4's real memory bandwidth at 242 GB/s, 76% of spec, using CUDA events with warmup and a median of 20 runs. A naive CPU timer reported a physically impossible 9,500 GB/s because GPU launches are asynchronous. The measured number lowered my decode ceiling for a 3 GB model from ~106 to ~80 tokens/sec."

## 2026-10-29: Device identity check (what PyTorch sees)

- **Phase:** 0 (Hardware fingerprint), step 2
- **Experiment:** One Colab cell on a free T4, printing GPU name, total/free memory, CUDA and torch versions, compute capability, and bf16 support. Saved as `inferscout/fingerprint/device_info.py` + `experiments/phase0/02_device_info.py`.
- **What I predicted:** Tesla T4, ~14.7 GiB total, ~14.5 GiB free, CUDA 12.x, torch 2.x, compute capability (7, 5), bf16 native False, bf16 with emulation True.
- **What happened:** Tesla T4 · 14.56 GiB total · 14.46 GiB free · CUDA 13.0 · torch 2.11.0+cu130 · (7, 5) · bf16 native False · bf16 with emulation True.
- **Why:**
  - CUDA 13.0, not 12.x: the prediction used stale knowledge. Colab now ships torch built against CUDA 13. CUDA 13 dropped GPUs older than Turing, so the T4 (sm_75) is now the oldest generation it still supports.
  - Total < 16: the T4 has 16 GiB physical, and on cloud GPUs ECC is on, which reserves 1/16 for error-check bits, leaving 15 GiB. The driver/firmware reserves roughly another 0.4–0.5 GiB, leaving ~14.56 GiB. (My "GB vs GiB" guess was the wrong main cause.)
  - Confirmed with `nvidia-smi`: ECC **Enabled**, 15360 MiB total (= 15 GiB), **448 MiB** driver reservation → 15360 − 448 = 14912 MiB = 14.56 GiB, exactly what PyTorch reports. Power limit **70 W**, max SM clock **1590 MHz**.
  - Where the 65 TFLOPS spec comes from: 320 Tensor Cores × 128 fp16 ops/clock × 1.59 GHz ≈ 65 TFLOPS. It assumes every Tensor Core is busy every cycle at max clock, which a 70 W card rarely holds.
  - Free is only ~0.1 GiB below total: CUDA lazy loading means the context is small until kernels actually run.
  - `torch.cuda.is_bf16_supported()` returns True on a T4 because it counts software emulation. Compute capability ≥ (8, 0) is the real test. **Trust the hardware number, not the convenience flag.**
- **Interview-ready one-liner:** "My fingerprint starts with device identity: a T4 is compute capability 7.5, so no native bf16. I check that from the hardware rather than PyTorch's default flag, which reports emulated bf16 as supported. And the real memory budget is ~14.5 GiB after ECC and driver reservations, not the 16 GB on the box."

## 2026-10-29: What a GPU is, and what Phase 0 measures (concepts, no experiment)

- **Phase:** 0 (Hardware fingerprint), step 1
- **Experiment:** None yet. Concept step: what a GPU is, and why matmul throughput, memory bandwidth and GPU memory matter for LLM latency. Target GPU: free Colab T4.
- **What I predicted:** My measured T4 numbers will come in under the spec sheet (~65 TFLOPS fp16, ~320 GB/s, 16 GB). Bandwidth around 60–80% of peak.
- **What happened:** Nothing measured yet. Key ideas:
  - GPU = thousands of simple cores working in parallel. LLMs are mostly matrix multiplications, which parallelize perfectly.
  - Prefill (processing the prompt) is compute-bound, so it drives TTFT. Decode (one token at a time) is memory-bound, so it drives inter-token latency.
  - Roofline sketch: a 3 GB fp16 model at 320 GB/s (spec) gives ~9.4 ms/token, so a ceiling of ~106 tokens/sec.
  - **Updated after step 3 with measured bandwidth:** 3 GB / 242 GB/s ≈ 12.4 ms/token ≈ **~80 tokens/sec** single-user ceiling. The spec overstated it by ~30%.
- **Why:**
  - "Decode is memory-bound" applies to a **single user**. With batching, the weights are read once and shared across every request in the batch, which is why concurrency raises total throughput. The ~100 tokens/sec roofline is **per user at batch size 1**, not the GPU's total limit.
  - The T4 is a **70W card** that often can't hold peak clocks under load, so measured TFLOPS may land well below the 65 TFLOPS spec. That's a real finding, not a bug.
- **Interview-ready one-liner:** "Before predicting LLM latency, I measured the hardware's real compute and bandwidth ceilings. Prefill is compute-bound, decode is memory-bound at batch size 1, and batching amortizes weight reads across requests."
