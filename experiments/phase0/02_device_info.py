"""Phase 0, step 2: print what PyTorch sees and save it to results/fingerprint_<gpu>.json.

Run from the repo root: python experiments/phase0/02_device_info.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so the script runs without installing the package

from inferscout.fingerprint import update_fingerprint
from inferscout.fingerprint.device_info import get_device_info

info = get_device_info()
for key, value in info.items():
    if key.endswith("_bytes"):
        value = f"{value / 2**30:.2f} GiB"
    print(f"{key:<22} {value}")

gpu = info["gpu_name"].lower().replace(" ", "_")
out = ROOT / f"experiments/phase0/results/fingerprint_{gpu}.json"
update_fingerprint(out, **info)
print(f"\nSaved {out.relative_to(ROOT)}")
