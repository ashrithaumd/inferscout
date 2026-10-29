"""Memory bandwidth: how fast data moves between GPU memory and the cores (decode's speed limit)."""

import statistics

import torch


def _time_copy(dst: torch.Tensor, src: torch.Tensor) -> float:
    # CUDA events timestamp on the GPU itself; a CPU timer would only see the async kernel launch.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    dst.copy_(src)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1e3  # ms -> s


def measure_bandwidth(size_mib: int = 512, warmup: int = 5, iters: int = 20) -> dict:
    # Far bigger than the T4's 4 MB L2 cache, so this measures DRAM, not cache.
    src = torch.empty(size_mib * 2**20 // 2, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    bytes_moved = 2 * src.nbytes  # a copy reads src and writes dst

    cold = _time_copy(dst, src)  # only truly cold in a fresh process
    for _ in range(warmup):
        dst.copy_(src)
    times = [_time_copy(dst, src) for _ in range(iters)]
    median = statistics.median(times)
    return {
        "tensor_mib": size_mib,
        "bytes_moved": bytes_moved,
        "warmup": warmup,
        "iters": iters,
        "cold_gbps": bytes_moved / cold / 1e9,
        "median_gbps": bytes_moved / median / 1e9,
        "median_ms": median * 1e3,
        "min_ms": min(times) * 1e3,
        "max_ms": max(times) * 1e3,
        "times_ms": [t * 1e3 for t in times],  # run order kept, so drift is visible
    }
