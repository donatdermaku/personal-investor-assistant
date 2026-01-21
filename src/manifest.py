from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.streamlit_data import CoverageMeta
from src.utils_io import DATA, ROOT


@dataclass
class RunManifest:
    run_id: str
    timestamp: str  # ISO 8601 UTC
    input_hash: str
    data_hash: str
    code_version: str
    coverage_summary: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> RunManifest:
        data = json.loads(json_str)
        return cls(**data)

    def save(self) -> Path:
        manifest_dir = DATA / "cache" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        path = manifest_dir / f"{self.run_id}.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def compute_file_hash(path: Path) -> str:
    """Compute distinct SHA256 hash of a file. Returns empty hash for missing files."""
    if not path.exists():
        return hashlib.sha256(b"").hexdigest()
    
    sha256 = hashlib.sha256()
    # Read in 64k chunks to handle large files efficiently
    with open(path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def compute_input_hash() -> str:
    """
    Compute hash of all user inputs:
    - watchlist.yml
    - ui_state.json
    - user_uploads/transactions.csv
    - user_uploads/holdings.csv
    """
    hasher = hashlib.sha256()
    
    # Order matters for stable hashing
    files = [
        ROOT / "watchlist.yml",
        DATA / "user_uploads" / "ui_state.json",
        DATA / "user_uploads" / "transactions.csv",
        DATA / "user_uploads" / "holdings.csv",
    ]
    
    for path in files:
        file_hash = compute_file_hash(path)
        hasher.update(file_hash.encode("utf-8"))
        
    return hasher.hexdigest()


def compute_data_hash() -> str:
    """
    Compute hash of the underlying data layer.
    Since we rely on latest parquet files, we hash the filenames and their mtimes
    or content of the latest files found.
    """
    hasher = hashlib.sha256()
    parq_dir = DATA / "parquet"
    
    # We care about specific prefixes used in the app
    prefixes = ["scores_daily", "fundamentals_quarterly", "prices_daily"]
    
    for prefix in sorted(prefixes):
        files = sorted(parq_dir.glob(f"{prefix}_*.parquet"))
        if files:
            latest = files[-1]
            # Hashing the content is safest for reproducibility
            hasher.update(compute_file_hash(latest).encode("utf-8"))
        else:
            hasher.update(b"missing")
            
    return hasher.hexdigest()


def get_git_revision() -> str:
    # Simple placeholder - in a real deployment we might fetch this from env or .git
    return "dev"


def create_manifest(coverage_map: dict[str, CoverageMeta] | None = None) -> RunManifest:
    """
    Create a new RunManifest for the current execution state.
    """
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    
    input_h = compute_input_hash()
    data_h = compute_data_hash()
    version = get_git_revision()
    
    # Summarize coverage for the manifest (simplify complexity)
    cov_summary = {}
    if coverage_map:
        for key, meta in coverage_map.items():
            cov_summary[key] = {
                "total": meta.total,
                "covered": meta.covered,
                "last_date": meta.last_date,
                "missing_count": len(meta.missing_tickers)
            }
            
    manifest = RunManifest(
        run_id=run_id,
        timestamp=timestamp,
        input_hash=input_h,
        data_hash=data_h,
        code_version=version,
        coverage_summary=cov_summary
    )
    manifest.save()
    return manifest
