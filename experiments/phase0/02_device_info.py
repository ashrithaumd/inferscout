"""Phase 0, step 2: print what PyTorch sees and save it to results/device_info.json.

Run from the repo root: python experiments/phase0/02_device_info.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so the script runs without installing the package

from inferscout.fingerprint.device_info import get_device_info

info = get_device_info()
for key, value in info.items():
    if key.endswith("_bytes"):
        value = f"{value / 2**30:.2f} GiB"
    print(f"{key:<22} {value}")

out = ROOT / "experiments/phase0/results/device_info.json"
out.write_text(json.dumps(info, indent=2))
print(f"\nSaved {out.relative_to(ROOT)}")
