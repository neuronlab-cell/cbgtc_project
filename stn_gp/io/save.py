# STUB: file saving to local + upload to GCS (use upload_run.py)
# - write manifest.json
# - write figures (raster, psd)

# save.py
# Standardized saving for STN–GPi runs + optional upload to GCS.

from __future__ import annotations
import os, json, subprocess
from pathlib import Path
from typing import Dict, Optional
import numpy as np

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_manifest(out_dir: Path, manifest: Dict) -> Path:
    """Write manifest.json to the run directory."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    mpath = out_dir / "manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    return mpath

def save_arrays(out_dir: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Save multiple numpy arrays under arrays/ subfolder."""
    arrdir = Path(out_dir) / "arrays"
    ensure_dir(arrdir)
    for name, arr in arrays.items():
        np.save(arrdir / f"{name}.npy", arr)

def save_figures(out_dir: Path, figures: Dict[str, "matplotlib.figure.Figure"]) -> None:
    """
    Save matplotlib figures under figs/ subfolder.
    The values are expected to be matplotlib Figure objects.
    """
    figdir = Path(out_dir) / "figs"
    ensure_dir(figdir)
    for name, fig in figures.items():
        fig.savefig(figdir / f"{name}.png", dpi=160)
        # Do not close here; caller decides (some may want to inspect interactively)

def save_text(out_dir: Path, relpath: str, text: str) -> None:
    """Save an arbitrary text blob inside the run folder."""
    p = Path(out_dir) / relpath
    ensure_dir(p.parent)
    with open(p, "w") as f:
        f.write(text)

def upload_with_helper(local_run_dir: Path, subfolder: str = "stn_gpi") -> bool:
    """
    Call the project-level helper to upload a run directory to GCS.
    Returns True if the command exits with status 0.
    """
    local_run_dir = Path(local_run_dir).expanduser().resolve()
    helper = Path("~/cbgtc_project/upload_run.py").expanduser()
    if not helper.exists():
        print("⚠️  upload_run.py not found; skipping upload.")
        return False
    cmd = ["python3", str(helper), str(local_run_dir), subfolder]
    print("⏫ Uploading via:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(res.stdout.strip())
        if res.stderr.strip():
            print(res.stderr.strip())
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Upload failed:", e)
        if e.stdout: print(e.stdout)
        if e.stderr: print(e.stderr)
        return False
