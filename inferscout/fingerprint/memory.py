"""Usable GPU memory: how much of the card PyTorch can actually fill (room for weights + KV cache)."""

import subprocess

import torch

MIB = 2**20


def _smi_used_bytes():
    # shortcut: nvidia-smi GPU 0 only, same single-GPU assumption as device_info.py
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", "0", "--format=csv,noheader,nounits", "--query-gpu=memory.used"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        return int(out.strip()) * MIB
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _grab(chunk_bytes: int, held: list) -> None:
    try:
        while True:
            held.append(torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda"))
    except torch.cuda.OutOfMemoryError:
        pass


def measure_usable_memory() -> dict:
    """Run in a fresh process: the 'before' reading must come before this process touches CUDA."""
    already_initialized = torch.cuda.is_initialized()
    used_before = _smi_used_bytes()

    free_after_context, total = torch.cuda.mem_get_info()  # first CUDA call creates the context
    used_after_context = _smi_used_bytes()

    a = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    (a @ a).sum().item()  # first matmul loads cuBLAS and its workspace
    del a
    torch.cuda.empty_cache()
    free_after_cublas = torch.cuda.mem_get_info()[0]
    used_after_cublas = _smi_used_bytes()

    held = []
    for chunk_mib in (256, 2):  # big chunks to go fast, then 2 MiB (the allocator's granularity) for the tail
        _grab(chunk_mib * MIB, held)
    usable = sum(t.numel() for t in held)
    free_when_full = torch.cuda.mem_get_info()[0]
    reserved_when_full = torch.cuda.memory_reserved()
    del held
    torch.cuda.empty_cache()

    return {
        "already_initialized": already_initialized,  # True = 'before' already includes this context
        "total_bytes": total,
        "smi_used_before_bytes": used_before,
        "smi_used_after_context_bytes": used_after_context,
        "smi_used_after_cublas_bytes": used_after_cublas,
        "free_after_context_bytes": free_after_context,
        "free_after_cublas_bytes": free_after_cublas,
        "context_bytes": total - free_after_context,
        "cublas_bytes": free_after_context - free_after_cublas,
        "usable_bytes": usable,
        "free_when_full_bytes": free_when_full,
        "reserved_when_full_bytes": reserved_when_full,
    }
