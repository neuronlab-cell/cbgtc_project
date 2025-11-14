# checkpoint.py
# Optional: lightweight checkpointing for long STN–GPi simulations.

from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Dict, Any
import numpy as np

def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def save_checkpoint(out_dir: Path,
                    t_ms: float,
                    state: Dict[str, Any],
                    freq_s: int = 60) -> None:
    """
    Save a periodic checkpoint snapshot.
    Parameters
    ----------
    out_dir : Path
        Run directory (contains 'checkpoints/' subfolder)
    t_ms : float
        Current simulation time
    state : dict
        Must contain arrays or serializable data (e.g., V, w, synapse.g)
    freq_s : int
        Minimum wall-clock seconds between checkpoints
    """
    chk_dir = Path(out_dir) / "checkpoints"
    ensure_dir(chk_dir)
    tag = f"{int(t_ms):08d}ms"
    arr_dir = chk_dir / tag
    ensure_dir(arr_dir)

    # Save arrays
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            np.save(arr_dir / f"{k}.npy", v)
        else:
            with open(arr_dir / f"{k}.json", "w") as f:
                json.dump(v, f, indent=2)

    # Metadata log
    meta = dict(t_ms=float(t_ms), wall_time=time.time())
    with open(chk_dir / "latest.json", "w") as f:
        json.dump(meta, f, indent=2)

def load_latest_checkpoint(chk_root: Path) -> Dict[str, Any] | None:
    """
    Load latest checkpoint metadata and arrays.
    Returns None if no checkpoint exists.
    """
    chk_root = Path(chk_root)
    meta_path = chk_root / "latest.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r") as f:
        meta = json.load(f)
    tag = f"{int(meta['t_ms']):08d}ms"
    arr_dir = chk_root / tag
    if not arr_dir.exists():
        return None

    state = {}
    for npy in arr_dir.glob("*.npy"):
        key = npy.stem
        state[key] = np.load(npy)
    for js in arr_dir.glob("*.json"):
        key = js.stem
        with open(js, "r") as f:
            state[key] = json.load(f)
    state["meta"] = meta
    return state

