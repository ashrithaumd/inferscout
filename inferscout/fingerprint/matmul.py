"""Matmul throughput: how fast the GPU's cores do math (prefill's speed limit)."""

import statistics
import subprocess
import threading
import time

import torch


def _time(fn) -> float:
    # Same method as bandwidth.py: CUDA events time the GPU work, not the async launch.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1e3  # ms -> s


def _matrices(n: int, dtype: torch.dtype):
    a = torch.randn(n, n, dtype=dtype, device="cuda")
    b = torch.randn(n, n, dtype=dtype, device="cuda")
    return a, b, torch.empty_like(a)


def measure_matmul(sizes=(512, 1024, 2048, 4096, 8192), dtypes=(torch.float32, torch.float16),
                   warmup: int = 5, iters: int = 20) -> list[dict]:
    # Step 3 showed an idle GPU starts at low clocks; spin it up so N=512 isn't timed mid-ramp.
    a, b, c = _matrices(4096, torch.float16)
    for _ in range(50):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()

    rows = []
    for dtype in dtypes:
        for n in sizes:
            a, b, c = _matrices(n, dtype)
            for _ in range(warmup):
                torch.matmul(a, b, out=c)
            times = [_time(lambda: torch.matmul(a, b, out=c)) for _ in range(iters)]
            median = statistics.median(times)
            flops = 2 * n**3  # n*n outputs, each n multiplies + n adds
            rows.append({
                "dtype": str(dtype).removeprefix("torch."),
                "n": n,
                "flops": flops,
                "median_tflops": flops / median / 1e12,
                "median_ms": median * 1e3,
                "min_ms": min(times) * 1e3,
                "max_ms": max(times) * 1e3,
                "times_ms": [t * 1e3 for t in times],
            })
    del a, b, c
    torch.cuda.empty_cache()
    return rows


def _smi_sample() -> dict:
    # shortcut: nvidia-smi GPU 0 only, same single-GPU assumption as device_info.py
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", "0", "--format=csv,noheader,nounits",
             "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        clock, power, temp, reasons = (v.strip() for v in out.split(","))
        return {"sm_clock_mhz": int(clock), "power_w": float(power), "temp_c": int(temp),
                "clock_event_reasons": reasons}  # bitmask; 0x4 = software power cap
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"sm_clock_mhz": None, "power_w": None, "temp_c": None, "clock_event_reasons": None}


def sustained_burst(n: int = 8192, dtype: torch.dtype = torch.float16, seconds: float = 30.0,
                    matmuls_per_chunk: int = 10, sample_every_s: float = 1.0) -> dict:
    """Keep the GPU busy for `seconds`; record TFLOPS per chunk and nvidia-smi readings over time."""
    a, b, c = _matrices(n, dtype)
    flops = 2 * n**3 * matmuls_per_chunk

    def chunk():
        for _ in range(matmuls_per_chunk):
            torch.matmul(a, b, out=c)

    before = _smi_sample()
    samples, stop, t0 = [], threading.Event(), time.perf_counter()

    def sampler():  # separate thread so sampling never pauses the GPU
        while not stop.wait(sample_every_s):
            samples.append({"t_s": round(time.perf_counter() - t0, 1), **_smi_sample()})

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    chunks = []
    while (t := time.perf_counter() - t0) < seconds:
        chunks.append({"t_s": round(t, 1), "tflops": flops / _time(chunk) / 1e12})
    stop.set()
    thread.join()
    after = _smi_sample()

    del a, b, c
    torch.cuda.empty_cache()
    return {"dtype": str(dtype).removeprefix("torch."), "n": n, "seconds": seconds,
            "matmuls_per_chunk": matmuls_per_chunk,
            "before": before, "during": samples, "after": after, "chunks": chunks}
