"""MLflow and local experiment tracking & model provenance.

Provides seamless logging to MLflow if installed, with automatic fallback
to an immutable local JSON/CSV experiment ledger.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Try importing MLflow (free & open-source industry standard)
try:
    import mlflow

    HAVE_MLFLOW = True
except ImportError:
    HAVE_MLFLOW = False


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 content hash of a data or model file for version tracking."""
    if not filepath.exists():
        return "missing"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def get_git_commit() -> str:
    """Retrieve current Git commit hash for reproducible lineage."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unversioned"


class ExperimentTracker:
    """Production-grade experiment tracker supporting MLflow and local ledger."""

    def __init__(
        self,
        experiment_name: str = "vic_rent_forecast",
        artifact_dir: Optional[Path] = None,
    ):
        self.experiment_name = experiment_name
        self.artifact_dir = artifact_dir or Path("artifacts/runs")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.active_run: Optional[Dict[str, Any]] = None

        if HAVE_MLFLOW:
            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> "ExperimentTracker":
        run_id = f"{run_name}_{int(time.time())}"
        self.active_run = {
            "run_id": run_id,
            "run_name": run_name,
            "git_commit": get_git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "params": {},
            "metrics": {},
            "artifacts": {},
        }
        if HAVE_MLFLOW:
            mlflow.start_run(run_name=run_name)
            mlflow.set_tag("git_commit", self.active_run["git_commit"])
        return self

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        self.active_run["params"].update(params)
        if HAVE_MLFLOW:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        self.active_run["metrics"].update(metrics)
        if HAVE_MLFLOW:
            mlflow.log_metrics(metrics)

    def log_artifact(self, file_path: Path, artifact_name: Optional[str] = None) -> None:
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        name = artifact_name or file_path.name
        self.active_run["artifacts"][name] = {
            "path": str(file_path),
            "sha256": compute_file_hash(file_path),
        }
        if HAVE_MLFLOW and file_path.exists():
            mlflow.log_artifact(str(file_path))

    def end_run(self) -> Dict[str, Any]:
        if not self.active_run:
            return {}
        # Write to local immutable ledger
        run_file = self.artifact_dir / f"{self.active_run['run_id']}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(self.active_run, f, indent=2)

        if HAVE_MLFLOW:
            mlflow.end_run()

        finished_run = self.active_run
        self.active_run = None
        return finished_run
