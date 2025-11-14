# manifest.py
# Build a standardized run manifest capturing config, hashes, env info, and versions.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json, hashlib, subprocess, sys, platform, socket, os, datetime

def _json_dumps_canonical(obj: Any) -> str:
    """Stable JSON string (sorted keys, no whitespace) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def hash_config(cfg: Dict[str, Any]) -> str:
    """SHA1 hash of the config dict (canonical JSON)."""
    s = _json_dumps_canonical(cfg)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def git_commit_short() -> Optional[str]:
    """Return the current Git short hash if available; otherwise None."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None

def python_env_info() -> Dict[str, Any]:
    """Minimal Python/env info for reproducibility."""
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import scipy
    except Exception:
        scipy = None
    try:
        import jax
    except Exception:
        jax = None

    return dict(
        python=dict(
            version=sys.version.split()[0],
            executable=sys.executable,
        ),
        numpy=getattr(np, "__version__", None),
        scipy=getattr(scipy, "__version__", None),
        jax=getattr(jax, "__version__", None),
        platform=dict(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            processor=platform.processor(),
        ),
        hostname=socket.gethostname(),
        pid=os.getpid(),
    )

@dataclass
class ManifestCore:
    module: str
    run_id: str
    start_time: str
    dt_ms: float
    fs_hz: float
    T_steps: int
    burn_steps: int
    seed: int
    cfg_hash: str
    git_commit: Optional[str]

def build_manifest(
    module: str,
    run_id: str,
    cfg_used: Dict[str, Any],
    dt_ms: float,
    fs_hz: float,
    T_steps: int,
    burn_steps: int,
    start_time: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assemble a complete manifest dict with:
      - core run fields
      - the full config used
      - environment (python, numpy/scipy/jax, platform)
    """
    start_iso = start_time or datetime.datetime.now().isoformat()
    mcore = ManifestCore(
        module=module,
        run_id=run_id,
        start_time=start_iso,
        dt_ms=float(dt_ms),
        fs_hz=float(fs_hz),
        T_steps=int(T_steps),
        burn_steps=int(burn_steps),
        seed=int(cfg_used.get("seed", -1)),
        cfg_hash=hash_config(cfg_used),
        git_commit=git_commit_short(),
    )
    manifest = dict(
        core=asdict(mcore),
        config=cfg_used,
        env=python_env_info(),
    )
    return manifest

