"""Phase 0, step 4: measure matmul throughput (TFLOPS) in fp32 and fp16, plus a 30 s sustained burst.

Saves results/matmul_<gpu>.json (every timing and nvidia-smi sample) and adds
matmul_fp16_tflops / matmul_fp32_tflops to results/fingerprint_<gpu>.json.
Run from the repo root: python experiments/phase0/04_matmul.py
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so the script runs without installing the package

import torch

from inferscout.fingerprint import update_fingerprint
from inferscout.fingerprint.matmul import measure_matmul, sustained_burst

# shortcut: T4 spec at 1590 MHz boost, hardcoded; look up per GPU when the L4 run happens.
# fp16: 320 Tensor Cores x 128 ops/clock. fp32: 2560 CUDA cores x 2 ops (FMA)/clock. Turing has no TF32.
SPEC_TFLOPS = {"float16": 320 * 128 * 1.59e9 / 1e12, "float32": 2560 * 2 * 1.59e9 / 1e12}

MAX_SM_CLOCK_MHZ = 1590  # spec numbers above assume this clock

rows = measure_matmul()
# TFLOPS = median over every run (>= 3 s per size); first20 = median of the first 20 runs (the old method).
# %clk = TFLOPS vs. the spec scaled down to the median SM clock actually seen: kernel efficiency at that clock.
print(f"{'dtype':<8} {'N':>5} {'median ms':>10} {'min ms':>8} {'max ms':>8} {'runs':>6} {'TFLOPS':>7} "
      f"{'first20':>7} {'% spec':>7} {'SM MHz':>7} {'power W':>8} {'temp C':>7} {'% clk':>6}")
for r in rows:
    spec = SPEC_TFLOPS[r["dtype"]]
    r["pct_of_spec"] = 100 * r["median_tflops"] / spec
    clk = r["median_sm_clock_mhz"]
    r["pct_of_clock_peak"] = 100 * r["median_tflops"] / (spec * clk / MAX_SM_CLOCK_MHZ) if clk else None
    pct_clk = f"{r['pct_of_clock_peak']:.1f}%" if clk else "None"
    print(f"{r['dtype']:<8} {r['n']:>5} {r['median_ms']:>10.3f} {r['min_ms']:>8.3f} {r['max_ms']:>8.3f} "
          f"{r['n_timed']:>6} {r['median_tflops']:>7.2f} {r['first_iters_median_tflops']:>7.2f} "
          f"{r['pct_of_spec']:>6.1f}% {clk!s:>7} {r['median_power_w']!s:>8} {r['median_temp_c']!s:>7} {pct_clk:>6}")
    # Beating the spec sheet means the timing is broken, not that the GPU is magic.
    assert max(r["median_tflops"], r["first_iters_median_tflops"]) < 1.05 * spec, \
        f"{r['dtype']} N={r['n']} above spec: timing bug?"

print("\nSustained burst: fp16 N=8192 for 30 s ...")
burst = sustained_burst()
tf = [c["tflops"] for c in burst["chunks"]]
k = max(1, len(tf) // 5)
print(f"TFLOPS first 20%: {statistics.median(tf[:k]):.2f}   last 20%: {statistics.median(tf[-k:]):.2f}   "
      f"min/max: {min(tf):.2f} / {max(tf):.2f}  ({len(tf)} chunks)")
print(f"{'when':<8} {'t_s':>5} {'SM MHz':>7} {'power W':>8} {'temp C':>7}  reasons")
for label, s in [("before", burst["before"]), *[("during", s) for s in burst["during"]], ("after", burst["after"])]:
    print(f"{label:<8} {s.get('t_s', ''):>5} {s['sm_clock_mhz']!s:>7} {s['power_w']!s:>8} {s['temp_c']!s:>7}  "
          f"{s['clock_event_reasons']}")

results = ROOT / "experiments/phase0/results"
gpu = torch.cuda.get_device_name().lower().replace(" ", "_")
(results / f"matmul_{gpu}.json").write_text(
    json.dumps({"spec_tflops": SPEC_TFLOPS, "sweep": rows, "burst": burst}, indent=2))
best = {d: max(r["median_tflops"] for r in rows if r["dtype"] == d) for d in SPEC_TFLOPS}
update_fingerprint(results / f"fingerprint_{gpu}.json",
                   matmul_fp16_tflops=round(best["float16"], 2), matmul_fp32_tflops=round(best["float32"], 2))
print(f"\nSaved results/matmul_{gpu}.json and updated results/fingerprint_{gpu}.json")
