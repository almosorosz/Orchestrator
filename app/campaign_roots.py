"""Remember recording roots so custom folders are discoverable after restart."""
from hashlib import sha256
import json
from pathlib import Path

from app.paths import DATA_DIR

REGISTRY = DATA_DIR / "runtime" / "campaign_roots"


def register_root(root):
    root = str(Path(root).resolve())
    REGISTRY.mkdir(parents=True, exist_ok=True)
    target = REGISTRY / (sha256(root.casefold().encode()).hexdigest() + ".json")
    if not target.exists():
        from experiment.recording import atomic_json
        atomic_json(target, {"root": root})


def registered_roots():
    roots = []
    for path in REGISTRY.glob("*.json"):
        try:
            root = json.loads(path.read_text(encoding="utf-8"))["root"]
            if isinstance(root, str) and Path(root).is_dir():
                roots.append(root)
        except (OSError, ValueError, KeyError):
            continue
    return roots
