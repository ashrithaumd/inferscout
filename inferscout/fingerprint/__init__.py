import json
from pathlib import Path


def update_fingerprint(path: Path, **fields) -> None:
    """Merge fields into the fingerprint JSON, so each Phase 0 step adds its piece."""
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(fields)
    path.write_text(json.dumps(data, indent=2))
