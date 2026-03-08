# ADHD Eye Tracking Inference Backend

This backend is built for the **single-backend flow**:

1. The frontend collects the full game session locally while the child plays.
2. After the final game ends, the frontend sends **one final payload** to the backend.
3. The backend converts raw `events` and `gaze` into task feature tables.
4. The backend summarizes those tables into the **same `task__column__mean/std/min/max` format** used during training.
5. The backend runs `model.joblib` and returns the final result.

## Recommended endpoint

`POST /analyze-session`

### JSON body

```json
{
  "sessionMeta": {
    "participantId": "test23",
    "age": 8,
    "condition": "Unsure"
  },
  "events": [
    {
      "t_ms": 27261.2,
      "name": "target_on",
      "payload": {
        "game": 2,
        "trial_id": "f_1",
        "side": "left",
        "isTarget": false,
        "x": 258.72,
        "y": 287.6
      }
    }
  ],
  "gaze": [
    {
      "t_ms": 27275.0,
      "x": 580.1,
      "y": 250.4,
      "valid": true,
      "raw": { "xNorm": 0.51, "yNorm": 0.33 }
    }
  ],
  "calibration": []
}
```

## Other endpoints

- `GET /health`
- `POST /predict-from-features` → upload the 4 task CSVs directly
- `POST /predict-from-session-files` → upload `events.csv` and `gaze.csv` for debugging
- `GET /runs/{run_id}/download` → download the saved artifacts for a completed run

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Important compatibility note

This backend assumes the model was trained on the feature schema found in the current `model.joblib` file. The current model expects these families of summarized features:

- `prosaccade__correct__*`
- `prosaccade__peak_speed__*`
- `antisaccade__correct__*`
- `antisaccade__peak_speed__*`
- `fixation__t0_ms__*`
- `fixation__t1_ms__*`
- `fixation__n__*`
- `fixation__dispersion__*`
- `fixation__pct_missing__*`
- `smooth_pursuit__t0_ms__*`
- `smooth_pursuit__t1_ms__*`
- `smooth_pursuit__n_samples__*`
- `smooth_pursuit__corr_x__*`
- `smooth_pursuit__mae__*`
- `smooth_pursuit__rmse__*`

The backend generates those names before inference.


## Troubleshooting

If you see an error like `numpy._core ... multiarray`, delete your virtual environment and recreate it with the pinned requirements from this project. On Windows:

```powershell
rmdir /s /q .venv
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/health` and confirm the backend reports the pinned NumPy and scikit-learn versions.
