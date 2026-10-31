"""Phase 0, step 5: measure how much GPU memory PyTorch can actually allocate.

Run as a fresh process, before anything else touches the GPU:
    python experiments/phase0/05_memory.py
Saves results/memory_<gpu>.json and adds cuda_context_bytes / cublas_bytes / usable_memory_bytes
to results/fingerprint_<gpu>.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so the script runs without installing the package

import torch

from inferscout.fingerprint import update_fingerprint
from inferscout.fingerprint.memory import measure_usable_memory

GIB = 2**30
BOX_BYTES = 16 * GIB  # the "16 GB" on the T4 spec sheet (physical GDDR6)

r = measure_usable_memory()
if r["already_initialized"]:
    print("WARNING: CUDA was already initialized; nvidia-smi 'before' includes this process.")


def mib(x):
    return "None" if x is None else f"{x / 2**20:,.0f} MiB"


print(f"Physical on the box:               {BOX_BYTES / GIB:.2f} GiB")
print(f"Total visible to CUDA:             {r['total_bytes'] / GIB:.2f} GiB  (after ECC + driver reserve, step 2)")
print(f"nvidia-smi used: before / +context / +cuBLAS: "
      f"{mib(r['smi_used_before_bytes'])} / {mib(r['smi_used_after_context_bytes'])} / "
      f"{mib(r['smi_used_after_cublas_bytes'])}")
print(f"CUDA context (total - free):       {mib(r['context_bytes'])}")
print(f"cuBLAS (first matmul):             {mib(r['cublas_bytes'])}")
print(f"Usable by torch tensors:           {r['usable_bytes'] / GIB:.2f} GiB  "
      f"= {100 * r['usable_bytes'] / r['total_bytes']:.1f}% of total, {100 * r['usable_bytes'] / BOX_BYTES:.1f}% of the box")
print(f"Free left when full:               {mib(r['free_when_full_bytes'])}")
print(f"Allocator reserved when full:      {r['reserved_when_full_bytes'] / GIB:.2f} GiB")

# Sanity: tensors can't exceed what was free after setup, and the grab must have found most of it.
assert r["usable_bytes"] <= r["free_after_cublas_bytes"]
assert r["usable_bytes"] > 0.9 * r["free_after_cublas_bytes"], "grab stopped early: another process using the GPU?"

results = ROOT / "experiments/phase0/results"
gpu = torch.cuda.get_device_name().lower().replace(" ", "_")
(results / f"memory_{gpu}.json").write_text(json.dumps(r, indent=2))
update_fingerprint(results / f"fingerprint_{gpu}.json", cuda_context_bytes=r["context_bytes"],
                   cublas_bytes=r["cublas_bytes"], usable_memory_bytes=r["usable_bytes"])
print(f"\nSaved results/memory_{gpu}.json and updated results/fingerprint_{gpu}.json")
