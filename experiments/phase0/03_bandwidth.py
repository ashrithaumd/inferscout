"""Phase 0, step 3: measure GPU memory bandwidth with a device-to-device copy.

Saves results/bandwidth_<gpu>.json (every timing) and adds memory_bandwidth_gbps
to results/fingerprint_<gpu>.json.
Run from the repo root: python experiments/phase0/03_bandwidth.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so the script runs without installing the package

import torch

from inferscout.fingerprint import update_fingerprint
from inferscout.fingerprint.bandwidth import measure_bandwidth

r = measure_bandwidth()
print(f"Tensor: {r['tensor_mib']} MiB fp16; bytes moved per copy (read + write): {r['bytes_moved'] / 1e9:.2f} GB")
print(f"Cold first copy:            {r['cold_gbps']:.1f} GB/s")
print(f"Median of {r['iters']} (after {r['warmup']} warmups): {r['median_ms']:.2f} ms -> {r['median_gbps']:.1f} GB/s")
print(f"min / max:                  {r['min_ms']:.2f} / {r['max_ms']:.2f} ms")

results = ROOT / "experiments/phase0/results"
gpu = torch.cuda.get_device_name().lower().replace(" ", "_")
(results / f"bandwidth_{gpu}.json").write_text(json.dumps(r, indent=2))
update_fingerprint(results / f"fingerprint_{gpu}.json", memory_bandwidth_gbps=round(r["median_gbps"], 1))
print(f"\nSaved results/bandwidth_{gpu}.json and updated results/fingerprint_{gpu}.json")
