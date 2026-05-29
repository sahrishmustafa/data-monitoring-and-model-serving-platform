# model/retrain_trigger.py
"""
Orchestrates retraining: runs train.py, updates Prometheus accuracy metric,
redeploys to AWS, and sends a Slack notification.
"""
import os
import sys
import logging
import argparse
import subprocess
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from exporter.metrics import retrain_count_total, model_accuracy

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
DEPLOY_SCRIPT = Path(__file__).parent.parent / "deploy" / "deploy.sh"


def send_slack(msg: str) -> None:
    if not SLACK_WEBHOOK_URL:
        return
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": msg}, timeout=5)
    except Exception as exc:
        log.error(f"Slack error: {exc}")


def run_training() -> tuple[float, str]:
    """Execute train.py and parse its output for accuracy and model path."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "train.py")],
        capture_output=True, text=True
    )
    log.info(result.stdout)
    if result.returncode != 0:
        log.error(result.stderr)
        raise RuntimeError("Training failed")

    # Parse last line: "Training complete. Accuracy=0.8500, Model=model/model_v2.pkl"
    last_line = result.stdout.strip().split("\n")[-1]
    parts = last_line.split(", ")
    acc = float(parts[0].split("=")[1])
    model_path = parts[1].split("=")[1]
    return acc, model_path


def redeploy() -> None:
    """Re-run the deployment script so EC2 gets the new model."""
    if not DEPLOY_SCRIPT.exists():
        log.warning("deploy.sh not found — skipping redeploy")
        return
    result = subprocess.run(["bash", str(DEPLOY_SCRIPT)], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Redeployment succeeded")
    else:
        log.error(f"Redeployment failed:\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", default="manual",
                        help="Why retraining was triggered")
    args = parser.parse_args()

    log.info(f"Retraining triggered. Reason: {args.reason}")
    retrain_count_total.inc()

    try:
        acc, model_path = run_training()
        model_accuracy.set(acc)
        log.info(f"New accuracy: {acc:.4f}, model: {model_path}")
        redeploy()
        send_slack(
            f"🤖 Model retrained successfully!\n"
            f"• Reason: {args.reason}\n"
            f"• New accuracy: {acc:.4f}\n"
            f"• Model: {model_path}"
        )
    except Exception as exc:
        log.error(f"Retraining failed: {exc}")
        send_slack(f"🚨 Retraining FAILED! Reason: {args.reason}. Error: {exc}")


if __name__ == "__main__":
    main()
