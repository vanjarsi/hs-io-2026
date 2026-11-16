"""Loading run results.

Each training run from the Workflow Studio gets its own folder in data/runs/:

    data/runs/<run-name>/metrics.json   (required: download from the run's archive)
    data/runs/<run-name>/results.csv    (optional: per-epoch curves, for loss charts)
    data/runs/<run-name>/notes.txt      (optional: what you changed in this run)
"""
import json
from pathlib import Path

import pandas as pd

RUNS_DIR = Path(__file__).parent / "data" / "runs"


def list_runs():
    """Names of run folders that contain a metrics.json, oldest first."""
    if not RUNS_DIR.exists():
        return []
    runs = [p for p in RUNS_DIR.iterdir() if (p / "metrics.json").exists()]
    return [p.name for p in sorted(runs, key=lambda p: p.stat().st_mtime)]


def _flatten(d, prefix=""):
    """Turn nested dicts into one level: {"a": {"b": 1}} -> {"a/b": 1}."""
    flat = {}
    for key, value in d.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def load_metrics(run):
    """All numbers from a run's metrics.json, flattened to one level."""
    with open(RUNS_DIR / run / "metrics.json") as f:
        raw = json.load(f)
    return {k: v for k, v in _flatten(raw).items() if isinstance(v, (int, float))}


def load_curves(run):
    """Per-epoch training curves (Ultralytics results.csv), or None."""
    path = RUNS_DIR / run / "results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # Ultralytics pads names with spaces
    return df


def load_notes(run):
    path = RUNS_DIR / run / "notes.txt"
    return path.read_text().strip() if path.exists() else ""


def save_run(run, metrics_bytes, csv_bytes=None, notes=""):
    """Save uploaded files as a new run folder."""
    json.loads(metrics_bytes)  # fail early if it isn't valid JSON
    folder = RUNS_DIR / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_bytes(metrics_bytes)
    if csv_bytes:
        (folder / "results.csv").write_bytes(csv_bytes)
    if notes:
        (folder / "notes.txt").write_text(notes)
