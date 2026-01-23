from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json
from pathlib import Path

from storage.repo import Repo
from src.utils_io import ROOT

app = FastAPI(title="Nexus Analytics API")

# Allow CORS for Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo = Repo()

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Nexus Analytics API"}

@app.get("/api/v1/run/latest")
def get_latest_run():
    """
    Get metadata for the last completed run.
    """
    last_run = repo.get_latest_run()
    if not last_run:
        raise HTTPException(status_code=404, detail="No runs found")
    
    return {
        "run_id": last_run.run_id,
        "status": last_run.status,
        "timestamp": last_run.completed_at.isoformat() if last_run.completed_at else None,
        "input_hash": last_run.input_hash,
        "data_hash": last_run.data_hash,
    }

@app.get("/api/v1/run/{run_id}/summary")
def get_run_summary(run_id: str):
    """
    Serve the precomputed JSON summary artifact.
    """
    # Security: Validate run_id format to prevent traversal (basic)
    if not run_id.isalnum() and "-" not in run_id:
         raise HTTPException(status_code=400, detail="Invalid Run ID")
         
    summary_path = ROOT / "data" / "exports" / run_id / f"summary_{run_id}.json"
    
    if not summary_path.exists():
        # Fallback to "latest" link if implemented, or just error
        raise HTTPException(status_code=404, detail="Summary artifact not found")
        
    try:
        data = json.loads(summary_path.read_text())
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load artifact: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
