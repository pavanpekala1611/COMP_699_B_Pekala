from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent / "all_merged.csv"
# Or, if you insist on an absolute path, make sure this is REAL:
DATA_PATH = Path("/Users/pavan/Documents/Capstone_Project/Dataset/all_merged.csv")

@app.get("/rtmms_dashboard/summary")
def get_summary():
    try:
        if not DATA_PATH.exists():
            return {
                "monitored_patients": 0,
                "active_alerts_5min": 0,
                "median_response_sec": 0,
                "critical_5min": 0,
                "warnings_5min": 0,
                "error": f"Data file not found at: {DATA_PATH}",
            }

        df = pd.read_csv(DATA_PATH)

        # Try to parse timestamp if present
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=5)
            last5 = df[df["ts"] >= window_start]
        else:
            last5 = df

        monitored_patients = (
            df["patient_id"].nunique() if "patient_id" in df.columns else 0
        )
        active_alerts = len(last5)

        if "response_time_sec" in df.columns:
            median_response = df["response_time_sec"].median()
        else:
            median_response = 0

        if "severity" in last5.columns:
            critical_5 = (last5["severity"] == "critical").sum()
            warning_5 = (last5["severity"] == "warning").sum()
        else:
            critical_5 = 0
            warning_5 = 0

        return {
            "monitored_patients": int(monitored_patients),
            "active_alerts_5min": int(active_alerts),
            "median_response_sec": float(median_response) if pd.notna(median_response) else 0,
            "critical_5min": int(critical_5),
            "warnings_5min": int(warning_5),
        }

    except Exception as e:
        # Print full traceback in terminal for you
        traceback.print_exc()
        # And return a JSON error to the frontend instead of crashing the server
        return {
            "monitored_patients": 0,
            "active_alerts_5min": 0,
            "median_response_sec": 0,
            "critical_5min": 0,
            "warnings_5min": 0,
            "error": f"{type(e).__name__}: {e}",
        }

@app.get("/")
def root():
    return {"status": "ok", "data_path": str(DATA_PATH)}
