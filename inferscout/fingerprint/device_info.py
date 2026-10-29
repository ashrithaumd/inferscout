"""Identify the GPU PyTorch sees: the first piece of the hardware fingerprint."""

import subprocess
from datetime import datetime, timezone

import torch


def _nvidia_smi_info(device: int) -> dict:
    # shortcut: assumes nvidia-smi index == torch index, breaks if CUDA_VISIBLE_DEVICES reorders GPUs; fine on single-GPU Colab
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(device), "--format=csv,noheader,nounits",
             "--query-gpu=ecc.mode.current,power.limit,clocks.max.sm"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        ecc, power, clock = (v.strip() for v in out.split(","))
        return {"ecc_mode": ecc, "power_limit_w": float(power), "max_sm_clock_mhz": int(clock)}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"ecc_mode": None, "power_limit_w": None, "max_sm_clock_mhz": None}


def get_device_info(device: int = 0) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible. In Colab: Runtime -> Change runtime type -> T4 GPU.")

    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    capability = torch.cuda.get_device_capability(device)
    return {
        "gpu_name": props.name,
        "total_memory_bytes": total,
        "free_memory_bytes": free,
        "free_memory_note": "point-in-time snapshot; shared with every process on the GPU",
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "compute_capability": capability,
        # bf16 needs Ampere (sm_80)+; torch's default check also counts slow software emulation.
        "bf16_native": capability >= (8, 0),
        **_nvidia_smi_info(device),
    }
