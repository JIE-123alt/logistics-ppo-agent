import copy
import json
import os
import platform
import shutil
import stat
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path):
    """Read a YAML file (always UTF-8, works on Windows)."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


RUN_NAMES = ["run_a_underfit", "run_b_overfit", "run_c_final"]


def load_run(name):
    """Read config/<name>.yaml on top of config/base_ppo.yaml (a run file lists only what differs)."""
    cfg = load_yaml("config/base_ppo.yaml")
    for key, value in load_yaml(f"config/{name}.yaml").items():
        if isinstance(value, dict):
            cfg.setdefault(key, {}).update(value)
        else:
            cfg[key] = value
    return cfg


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


def _clear_readonly(func, path, _error):
    """Called by rmtree when a delete fails: clear the read-only flag, then try again."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def delete_folder(folder):
    """Delete a folder on Windows, even if some files inside are marked read-only.

    Use it for ONE run folder only (for example runs/run_c_final), never for the
    whole runs/ or results/ folder.
    """
    folder = Path(folder)
    if not folder.exists():
        return
    for _attempt in range(3):
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(folder, onexc=_clear_readonly)    # Python 3.12 and newer
            else:
                shutil.rmtree(folder, onerror=_clear_readonly)  # older Python
            return
        except PermissionError:
            time.sleep(1)  # another program may still hold a file, so wait and retry
    raise PermissionError(
        f"Cannot delete {folder}. Close TensorBoard and any File Explorer window "
        "that shows this folder, then run again."
    )