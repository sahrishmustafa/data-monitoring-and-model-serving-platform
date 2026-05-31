# ingestion/ingestion.py
"""
Main ingestion script. Polls the data API, saves records to CSV,
monitors for schema changes and distribution drift, and triggers
retraining when conditions are met.
"""
import os
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from drift_detector import DriftDetector

# Load .env variables
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────
DATA_API_URL = os.getenv("DATA_API_URL", "http://149.40.228.124:6500/records")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
OUTPUT_CSV = Path("data/records.csv")
RETRAIN_THRESHOLD_ROWS = 100   # trigger retraining after this many new rows

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Slack helper (sends raw webhook message) ────────────────────────
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

def send_slack_alert(message: str) -> None:
    """Post a plain-text message to the configured Slack webhook."""
    if not SLACK_WEBHOOK_URL:
        log.warning("SLACK_WEBHOOK_URL not set — skipping Slack alert")
        return
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as exc:
        log.error(f"Failed to send Slack alert: {exc}")

# ── Lazy import of Prometheus metrics ──────────────────────────────
# We import here so the ingestion script can be tested independently
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from exporter.metrics import (
    records_processed_total,
    feature_added,
    feature_removed,
    datalake_unavailable,
    distribution_drift_detected,
)

# ── State ──────────────────────────────────────────────────────────
known_schema: list[str] = []          # last seen list of feature names
drift_detector = DriftDetector()
new_rows_since_retrain: int = 0


def fetch_batch() -> dict | None:
    """
    GET /records. Returns parsed JSON on 200, None on 503 or error.
    Increments datalake_unavailable counter on 503.
    """
    try:
        resp = requests.get(DATA_API_URL, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 503:
            log.warning("API returned 503 — data source unavailable")
            datalake_unavailable.inc()
            send_slack_alert(
                "⚠️ Data source returned 503. Check API availability."
            )
        else:
            log.warning(f"Unexpected HTTP {resp.status_code}")
    except requests.RequestException as exc:
        log.error(f"Request failed: {exc}")
        datalake_unavailable.inc()
        send_slack_alert("⚠️ Data source unreachable. Check API availability.")
    return None


def check_schema(new_schema: list[str]) -> None:
    """
    Compare new_schema to known_schema.
    Fire metrics and Slack alerts for any differences.
    """
    global known_schema
    if not known_schema:
        known_schema = new_schema
        log.info(f"Initial schema recorded: {known_schema}")
        return

    added = set(new_schema) - set(known_schema)
    removed = set(known_schema) - set(new_schema)

    for col in added:
        log.warning(f"Schema change — feature ADDED: {col}")
        feature_added.inc()
        send_slack_alert(
            f"🆕 New feature detected in schema: '{col}'. "
            "Retraining may be required."
        )

    for col in removed:
        log.warning(f"Schema change — feature REMOVED: {col}")
        feature_removed.inc()
        send_slack_alert(
            f"❌ Feature dropped from schema: '{col}'. "
            "Verify pipeline compatibility."
        )

    if added or removed:
        # Schema changed — trigger retraining
        trigger_retrain(reason="schema_change")

    known_schema = new_schema


def save_records(schema: list[str], records: list[dict]) -> None:
    """Append a batch of records to the local CSV file."""
    global new_rows_since_retrain
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=schema)

    # Append — write header only if file is new
    write_header = not OUTPUT_CSV.exists()
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)

    records_processed_total.inc(len(records))
    new_rows_since_retrain += len(records)
    log.info(f"Saved {len(records)} records (total new: {new_rows_since_retrain})")


def trigger_retrain(reason: str) -> None:
    """
    Invoke the retraining orchestrator as a subprocess.
    Resets the new-rows counter afterwards.
    """
    global new_rows_since_retrain
    log.info(f"Triggering retraining — reason: {reason}")
    import subprocess
    subprocess.Popen(
        [sys.executable, str(Path(__file__).parent.parent / "model" / "retrain_trigger.py"),
         "--reason", reason]
    )
    new_rows_since_retrain = 0


def ingest_once() -> None:
    """One full ingestion cycle: fetch → schema check → drift → save → maybe retrain."""
    batch = fetch_batch()
    if batch is None:
        return

    schema: list[str] = []
    records: list[dict] = []

    if isinstance(batch, list):
        if batch:
            # Determine maximum number of features from items in batch
            num_features = 0
            for item in batch:
                if isinstance(item, dict) and "features" in item and isinstance(item["features"], list):
                    num_features = max(num_features, len(item["features"]))
            
            schema = [f"feat_{i}" for i in range(num_features)] + ["label"]
            for item in batch:
                if isinstance(item, dict):
                    feat_vals = item.get("features", [])
                    label_val = item.get("label")
                    # Construct record mapping feat_0, feat_1, ... and label
                    record_dict = {f"feat_{i}": (feat_vals[i] if i < len(feat_vals) else None) for i in range(num_features)}
                    record_dict["label"] = label_val
                    records.append(record_dict)
    elif isinstance(batch, dict):
        schema = batch.get("schema", [])
        records = batch.get("records", [])
    else:
        log.warning(f"Unexpected batch format: {type(batch)}")
        return

    if not schema or not records:
        log.warning("Empty batch received — skipping")
        return

    # Schema monitoring
    check_schema(schema)

    # Build numeric-only dict for drift detection
    numeric_data: dict[str, list] = {}
    for col in schema:
        col_values = []
        for row in records:
            val = row.get(col)
            if val is not None:
                try:
                    col_values.append(float(val))
                except (TypeError, ValueError):
                    pass  # skip non-numeric columns
        if col_values:
            numeric_data[col] = col_values

    # Distribution drift detection
    drift_found, drifted_cols = drift_detector.detect(numeric_data)
    if drift_found:
        log.warning(f"Drift detected in columns: {drifted_cols}")
        distribution_drift_detected.set(1)
        send_slack_alert(
            f"📊 Data distribution drift detected in: {drifted_cols}. "
            "Model may be stale."
        )
        trigger_retrain(reason="distribution_drift")
    else:
        distribution_drift_detected.set(0)

    # Save data
    save_records(schema, records)

    # Retrain if enough new rows have accumulated
    if new_rows_since_retrain >= RETRAIN_THRESHOLD_ROWS:
        trigger_retrain(reason="new_data_accumulated")


def main():
    log.info(f"Starting ingestion loop — polling every {POLL_INTERVAL}s")
    while True:
        ingest_once()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
