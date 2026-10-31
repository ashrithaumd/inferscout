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

## 2026-10-31: Matmul throughput (fp32 / fp16 sweep + 30 s sustained burst)

- **Phase:** 0 (Hardware fingerprint), step 4
- **Experiment:** Colab T4, commit fdb6083. Square matmuls N = 512–8192 in fp32 and fp16 (no bf16 on T4). Same timing as step 3: spin-up, 5 warmups, 20 CUDA-event timings, median plus min/max. FLOPs = 2·N³. Then fp16 N=8192 for 30 s, recording TFLOPS for each chunk of 10 matmuls and nvidia-smi clock/power/temp/reasons about once a second. Saved as `inferscout/fingerprint/matmul.py` + `experiments/phase0/04_matmul.py`.
- **What I predicted:** fp32 ≈ 2.5–3.5 TFLOPS at N=512, then ~4–4.5 (~55% of 8.14 spec). fp16 ≈ 5–8 at N=512, 15–18 at 1024, 22–27 for 2048+, best ~25–28 (~40% of 65 spec). Under the burst: power-capped (reason 0x4) at **1000–1300 MHz**, TFLOPS falling 5–10% as the card heats up.
- **What happened:**

  | N | fp32 TFLOPS | % of 8.14 | fp16 TFLOPS | % of 65.1 |
  |---|---|---|---|---|
  | 512 | 2.63 | 32.4% | 5.38 | 8.3% |
  | 1024 | 4.00 | 49.2% | 16.51 | 25.3% |
  | 2048 | 3.91 | 48.0% | **23.34** | 35.8% |
  | 4096 | **4.05** | 49.8% | 21.44 | 32.9% |
  | 8192 | 3.78 | 46.5% | 22.53 | 34.6% |

  - Burst: **22.17 → 21.11 TFLOPS** (first vs last 20%, −4.8%), min/max 20.83 / 22.82, 60 chunks.
  - Clocks: **SM 570–630 MHz (mostly 585–615), ~37% of the 1590 max.** Power 65–69.4 W, temp 62 → 73 °C, reason **0x4 (software power cap) the whole time**. Before: 1050 MHz / 69.4 W, already capped from the sweep. After: 720 MHz / 65.2 W.
- **Why:**
  - The size curve matched: small matmuls can't fill 40 SMs, launch overhead dominates a ~50 µs kernel, and fp16 at N=512 is below even the spec ridge point (intensity N/3 ≈ 170 < 269 FLOPs/byte). fp32 sizes all land within ~0.3 TFLOPS of the prediction.
  - **The clock prediction was far off (predicted 1000–1300, got ~600 MHz).** Running Tensor Cores at full rate is the most power-dense work the chip does. At 70 W, the only way to stay under the cap is to cut the clock by roughly 2.6×.
  - **Clock-adjusted peak:** 320 Tensor Cores × 128 fp16 FLOPs/clock × 0.6 GHz = **24.6 TFLOPS** (24.0–25.2 for 585–615 MHz). The burst delivered 21.1–22.2, which is **~86–90% of the peak at the actual clock**. cuBLAS is efficient; **the 70 W power budget is the bottleneck**, and it costs ~63% of spec.
  - fp32 hit 4.05 TFLOPS, which needs at least 4.05 / (2560 × 2) ≈ **0.79 GHz** even at 100% efficiency. So fp32 ran at a higher clock than fp16. CUDA-core math draws less power than Tensor Cores at full rate. (Inferred; clocks weren't sampled during the sweep.)
  - The −4.8% burst slide tracks the 62 → 73 °C rise: hotter silicon leaks more power, which leaves less of the 70 W for math.
  - fp16 N=4096 (21.4) below N=2048 (23.3) was an open question after this run. Clocks weren't sampled per size. Settled by the rerun below.
- **Rerun with per-size clock sampling (commit aabbe54, fresh Colab T4):** each (dtype, N) now runs ≥ 20 times *and* ≥ 3 s while nvidia-smi samples in the background. "first20" = median of the first 20 runs (the old method). "%clk" = TFLOPS ÷ (spec scaled to the median sampled clock).

  | dtype | N | TFLOPS (all runs) | first20 | SM MHz | power W | temp °C | %clk |
  |---|---|---|---|---|---|---|---|
  | fp32 | 512 | 3.74 | 2.29 | 1440 | 68.6 | 54 | 50.8% |
  | fp32 | 1024 | 3.70 | 5.52 | 990 | 69.2 | 56 | 73.1% |
  | fp32 | 2048 | 3.83 | 3.85 | 795 | 69.2 | 58 | 94.2% |
  | fp32 | 4096 | 3.89 | 3.89 | 780 | 69.0 | 59 | 97.4% |
  | fp32 | 8192 | 3.66 | 3.66 | 735 | 69.0 | 62 | 97.2% |
  | fp16 | 512 | 8.74 | 5.67 | 1590 | 53.4 | 63 | 13.4% |
  | fp16 | 1024 | 22.66 | 26.48 | 1148 | 67.8 | 65 | 48.2% |
  | fp16 | 2048 | 20.41 | **31.98** | 690 | 67.4 | 66 | 72.2% |
  | fp16 | 4096 | 21.01 | 20.93 | 600 | 68.2 | 67 | 85.5% |
  | fp16 | 8192 | 21.59 | 21.47 | 600 | 67.4 | 68 | 87.9% |

  - Burst fp16 N=8192 for 30 s: **21.38 → 20.52 TFLOPS** (first vs last 20%), min/max 20.49 / 22.04, 58 chunks. Clock 555–615 MHz, temp 68 → 77 °C, power 66–70 W, reason 0x4 throughout. Before: 1185 MHz. After: 555 MHz.
  - **1. N=2048 settled: short bursts beat the power cap.** first20 = 31.98 TFLOPS vs 20.41 sustained at 690 MHz. The first 20 runs (~15 ms) finish before the power controller clamps the clock. The earlier 23.3 was a real transient, not a sustained number, and it moves with whatever ran before it (23.3 in run 1 vs 31.98 here). Caveat: no clock sample lands inside those first ~15 ms, so the high clock is inferred from the speed, not measured. The lag works in both directions: fp32 and fp16 at N=512 have first20 *below* the full median (2.29 vs 3.74, 5.67 vs 8.74). They followed heavy work, so the clock was still clamped and then rose.
  - **2. N=4096 vs N=8192 at equal clock (600 MHz): 85.5% vs 87.9%.** The gap is small, and it is about as big as the clock reading's resolution: nvidia-smi steps of ~15 MHz at 600 MHz ≈ ±2.5%. cuBLAS kernel/tile choice is plausible, but this data can't separate it from measurement noise. The real takeaway: once clocks are equal, large-N fp16 is ~86–88% efficient.
  - **3. fp32 is power-capped too, at a higher clock: 735–795 MHz vs ~600 for fp16.** My 850–1100 MHz prediction was high. The earlier inference ("fp32 must run above fp16's clock") holds. At ~97% clock-adjusted efficiency, the power cap is the whole story for fp32. (97% is suspiciously close to perfect. The median clock sample probably reads a little under the true average, so read it as "≥ 90%".)
  - **4. fp16 N=512 is not power-limited.** It ran at the full 1590 MHz and only 53 W. Its 13% efficiency is purely too little work: 16 tiles for 40 SMs, plus launch/sync overhead on a ~30 µs kernel.
  - **5. Burst vs sustained: ~32 TFLOPS short-burst vs ~20.5–21.5 sustained.** This burst started at 68 °C and ended ~3% below run 1, which started at 60 °C. That fits hotter silicon leaking more power. But the two runs were likely on different physical T4s, so chip-to-chip variation is mixed in. Two runs can't separate the two causes.
  - **Max-time outliers** (e.g. 9.1 ms on fp16 N=512, median ~0.03 ms) are likely interference from the background nvidia-smi sampler. If the CPU stalls between recording the start event and launching a tiny kernel, the GPU sits idle inside the timed window. With ~45k runs, the medians are unaffected.
  - **Fingerprint now separates regimes:** `burst_fp16_tflops` 31.98, `sustained_fp16_tflops` 20.52 (burst last 20%), `sustained_fp32_tflops` 3.89 (best full median at N ≥ 2048), `burst_start_temp_c` 68. The old single "best" (22.66 at N=1024, a transient 1148 MHz) mixed the two regimes and was dropped.
  - **Roofline update:** sustained ridge point = 20.52 TFLOPS / 241.9 GB/s ≈ **85 FLOPs/byte** (burst upper bound ≈ 132; spec ≈ 269). Phase 3 uses sustained numbers for loaded servers. A spec-based TTFT would come out ~3× too optimistic. Batched fp16 decode (~B FLOPs per weight byte at batch B) turns compute-bound at **batch ≈ 85, not ~270**, so throughput stops scaling with concurrency much sooner than the spec suggests.
- **Interview-ready one-liner:** "A T4's fp16 matmul delivers ~32 TFLOPS in a short burst but only ~20.5 sustained, about a third of the 65 TFLOPS spec. Sampling nvidia-smi per size showed the 70 W power cap pulling the clock to ~600 MHz, where cuBLAS is still ~87% efficient. So the gap is power, not kernel quality, and my predictor uses the sustained number: a ridge point of ~85 FLOPs/byte, not 269."

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
- **Follow-up: repo scripts on a fresh clone (commit fff7a3b):**
  - **Reproducible:** the repo script measured **241.9 GB/s** vs **242.2 GB/s** from the earlier cell, 0.1% apart. Min/max 4.42 / 4.45 ms. Saved in `results/bandwidth_tesla_t4.json` and `results/fingerprint_tesla_t4.json`.
  - **Cold-run cause:** this time the cold copy was **236 GB/s**, not 111. The GPU was already busy from earlier runs (clocks up), but the script was a fresh process. So the earlier slow cold run was mostly **clock ramp-up from idle**, not loading. (Likely also why: a plain same-dtype GPU→GPU copy probably runs on the GPU's copy engine, with no kernel to lazy-load at all.)
  - **Free memory was 13.36 GiB, not 14.46:** another notebook process still held ~1 GiB of tensors. **Free memory is shared across all processes on the GPU**, so it's a point-in-time snapshot. The fingerprint now records `free_memory_note` and `measured_at` on future runs.
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
  - **Updated after step 4 with measured compute:** sustained ridge point = 20.52 TFLOPS / 241.9 GB/s ≈ **85 FLOPs/byte** (spec: ~269). Prefill on a loaded server must be predicted with ~20.5 TFLOPS sustained, not 65. The ~32 TFLOPS burst is only an upper bound for an idle GPU.
- **Why:**
  - "Decode is memory-bound" applies to a **single user**. With batching, the weights are read once and shared across every request in the batch, which is why concurrency raises total throughput. The ~100 tokens/sec roofline is **per user at batch size 1**, not the GPU's total limit.
  - The T4 is a **70W card** that often can't hold peak clocks under load, so measured TFLOPS may land well below the 65 TFLOPS spec. That's a real finding, not a bug.
- **Interview-ready one-liner:** "Before predicting LLM latency, I measured the hardware's real compute and bandwidth ceilings. Prefill is compute-bound, decode is memory-bound at batch size 1, and batching amortizes weight reads across requests."
