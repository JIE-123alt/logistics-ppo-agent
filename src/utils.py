import copy
import json
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path):
    """Read a YAML file (always UTF-8, works on Windows)."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def with_changes(cfg, section, **changes):
    """Return a COPY of the env config with some values changed (used by stress tests)."""
    new_cfg = copy.deepcopy(cfg)
    new_cfg[section].update(changes)
    return new_cfg


def get_splits(cfg):
    """Return the scenario seeds of each split, e.g. {'train': [1000..1049], ...}."""
    out = {}
    for name, s in cfg["splits"].items():
        out[name] = list(range(s["start"], s["start"] + s["count"]))
    return out


def get_dirs(quick):
    """Quick mode writes to separate folders so it never mixes with the full run."""
    suffix = "_quick" if quick else ""
    results, runs = ROOT / f"results{suffix}", ROOT / f"runs{suffix}"
    results.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    return results, runs


def get_docs_dir(quick):
    """Full run writes into docs/. Quick run writes a preview inside results_quick/ instead."""
    docs = (ROOT / "results_quick" / "docs_preview") if quick else (ROOT / "docs")
    docs.mkdir(parents=True, exist_ok=True)
    return docs


def library_versions():
    """Versions of everything that matters for reproducing a run."""
    import gymnasium
    import numpy
    import stable_baselines3
    import torch
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
    }
