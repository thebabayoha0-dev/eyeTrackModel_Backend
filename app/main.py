from __future__ import annotations

import io
import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import sklearn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .pipeline_utils import (
    TASK_FILE_ORDER,
    build_summary_row,
    extract_task_feature_tables,
    load_csv_upload,
    run_classification_from_summary,
    run_end_to_end_from_raw,
)

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(APP_DIR, "data", "runs"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(APP_DIR, "models", "model.joblib"))

os.makedirs(DATA_DIR, exist_ok=True)


def _parse_origins() -> List[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    )
    return [x.strip() for x in raw.split(",") if x.strip()]


app = FastAPI(title="ADHD Eye Tracking Inference Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionAnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_meta: Dict[str, Any] = Field(default_factory=dict, alias="sessionMeta")
    events: List[Dict[str, Any]] = Field(default_factory=list)
    gaze: List[Dict[str, Any]] = Field(default_factory=list)
    calibration: List[Dict[str, Any]] = Field(default_factory=list)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_path": MODEL_PATH,
        "data_dir": DATA_DIR,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "ADHD Eye Tracking Inference Backend",
        "recommended_flow": "POST /analyze-session after the last game ends",
        "endpoints": [
            "/health",
            "/analyze-session",
            "/predict-from-features",
            "/predict-from-session-files",
            "/runs/{run_id}/download",
        ],
    }


def _run_dir(run_id: str, create: bool = True) -> str:
    path = os.path.join(DATA_DIR, run_id)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_dataframe(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _persist_run_artifacts(
    run_id: str,
    session_meta: Dict[str, Any],
    events_df: pd.DataFrame,
    gaze_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    task_tables: Dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    result: Dict[str, Any],
) -> str:
    run_dir = _run_dir(run_id, create=True)
    _save_json(os.path.join(run_dir, "session_meta.json"), session_meta)
    _save_dataframe(os.path.join(run_dir, "events.csv"), events_df)
    _save_dataframe(os.path.join(run_dir, "gaze.csv"), gaze_df)
    _save_dataframe(os.path.join(run_dir, "calibration.csv"), calibration_df)

    for task_name, filename in TASK_FILE_ORDER:
        _save_dataframe(os.path.join(run_dir, filename), task_tables.get(task_name, pd.DataFrame()))

    _save_dataframe(os.path.join(run_dir, "summary_features.csv"), summary_df)
    _save_json(os.path.join(run_dir, "result.json"), result)
    return run_dir


@app.post("/analyze-session")
def analyze_session(payload: SessionAnalyzeRequest) -> Dict[str, Any]:
    if not payload.events:
        raise HTTPException(status_code=400, detail="events array is required")
    if not payload.gaze:
        raise HTTPException(status_code=400, detail="gaze array is required")

    events_df = pd.DataFrame(payload.events)
    gaze_df = pd.DataFrame(payload.gaze)
    calibration_df = pd.DataFrame(payload.calibration)

    try:
        task_tables, summary_df, result = run_end_to_end_from_raw(events_df, gaze_df, MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Session analysis failed: {exc}") from exc

    run_id = str(uuid.uuid4())
    _persist_run_artifacts(
        run_id=run_id,
        session_meta=payload.session_meta,
        events_df=events_df,
        gaze_df=gaze_df,
        calibration_df=calibration_df,
        task_tables=task_tables,
        summary_df=summary_df,
        result=result,
    )

    return {
        "run_id": run_id,
        "status": "completed",
        "result": result,
        "generated_feature_files": {task_name: filename for task_name, filename in TASK_FILE_ORDER},
        "summary_feature_columns": list(summary_df.columns),
    }


@app.post("/predict-from-features")
async def predict_from_features(feature_files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    expected_names = {filename for _, filename in TASK_FILE_ORDER}
    uploaded_names = {f.filename.lower() for f in feature_files}
    missing = expected_names - uploaded_names
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required feature files: {', '.join(sorted(missing))}")

    task_tables: Dict[str, pd.DataFrame] = {}
    for task_name, filename in TASK_FILE_ORDER:
        upload = next((f for f in feature_files if f.filename.lower() == filename), None)
        if upload is None:
            continue
        task_tables[task_name] = load_csv_upload(await upload.read())

    try:
        summary_df = build_summary_row(task_tables)
        result = run_classification_from_summary(summary_df, MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature prediction failed: {exc}") from exc

    run_id = str(uuid.uuid4())
    _persist_run_artifacts(
        run_id=run_id,
        session_meta={},
        events_df=pd.DataFrame(),
        gaze_df=pd.DataFrame(),
        calibration_df=pd.DataFrame(),
        task_tables=task_tables,
        summary_df=summary_df,
        result=result,
    )

    return {
        "run_id": run_id,
        "status": "completed",
        "result": result,
        "summary_feature_columns": list(summary_df.columns),
    }


@app.post("/predict-from-session-files")
async def predict_from_session_files(
    events_file: UploadFile = File(...),
    gaze_file: UploadFile = File(...),
    session_meta_file: Optional[UploadFile] = File(None),
    calibration_file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    try:
        events_df = load_csv_upload(await events_file.read())
        gaze_df = load_csv_upload(await gaze_file.read())
        calibration_df = load_csv_upload(await calibration_file.read()) if calibration_file else pd.DataFrame()
        session_meta = {}
        if session_meta_file is not None:
            session_meta = json.loads((await session_meta_file.read()).decode("utf-8"))
        task_tables, summary_df, result = run_end_to_end_from_raw(events_df, gaze_df, MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Session-file prediction failed: {exc}") from exc

    run_id = str(uuid.uuid4())
    _persist_run_artifacts(
        run_id=run_id,
        session_meta=session_meta,
        events_df=events_df,
        gaze_df=gaze_df,
        calibration_df=calibration_df,
        task_tables=task_tables,
        summary_df=summary_df,
        result=result,
    )

    return {
        "run_id": run_id,
        "status": "completed",
        "result": result,
        "summary_feature_columns": list(summary_df.columns),
    }


@app.get("/runs/{run_id}/download")
def download_run_artifacts(run_id: str):
    run_dir = _run_dir(run_id, create=False)
    if not os.path.isdir(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(run_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                arcname = os.path.relpath(full_path, run_dir)
                zf.write(full_path, arcname=arcname)

    zbuf.seek(0)
    return StreamingResponse(
        zbuf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_artifacts.zip"'},
    )
