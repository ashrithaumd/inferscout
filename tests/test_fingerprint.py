import json

from inferscout.fingerprint import update_fingerprint


def test_update_fingerprint_merges_steps(tmp_path):
    path = tmp_path / "fingerprint.json"
    update_fingerprint(path, gpu_name="Tesla T4", total_memory_bytes=1)
    update_fingerprint(path, memory_bandwidth_gbps=242.2)
    update_fingerprint(path, total_memory_bytes=2)  # rerunning a step overwrites only its own keys

    assert json.loads(path.read_text()) == {
        "gpu_name": "Tesla T4",
        "total_memory_bytes": 2,
        "memory_bandwidth_gbps": 242.2,
    }
