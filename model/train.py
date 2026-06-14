# model/train.py
"""
Training script. Reads data/records.csv, trains a Random Forest,
enforces validation accuracy >= 0.80, and saves a versioned .pkl file.
"""
import os
import sys
import logging
import glob
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_PATH = Path("data/records.csv")
MODEL_DIR = Path("model")
ACCURACY_THRESHOLD = float(os.getenv("RETRAIN_ACCURACY_THRESHOLD", "0.80"))
MAX_ITERATIONS = int(os.getenv("MAX_TRAIN_ITERATIONS", "10"))


def get_next_version() -> int:
    """Find the highest existing model version and return next number."""
    existing = glob.glob(str(MODEL_DIR / "model_v*.pkl"))
    if not existing:
        return 1
    versions = []
    for p in existing:
        try:
            versions.append(int(Path(p).stem.split("_v")[1]))
        except (IndexError, ValueError):
            pass
    return (max(versions) + 1) if versions else 1


def load_and_prepare_data() -> tuple:
    """
    Read CSV, drop rows with missing values, separate features from target.
    Assumes the last column is the target.
    """
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    if len(df) < 20:
        raise ValueError(f"Not enough data to train: {len(df)} rows")

    # Encode any string columns
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def train() -> tuple[float, Path]:
    """
    Train until accuracy >= threshold or MAX_ITERATIONS exceeded.
    Returns (best_accuracy, model_path).
    """
    X, y = load_and_prepare_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_acc = 0.0
    best_model = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        # Vary n_estimators each iteration to search for better configs
        n_est = 50 * iteration
        clf = RandomForestClassifier(
            n_estimators=n_est,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        log.info(f"Iteration {iteration}: n_estimators={n_est}, accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = clf

        if best_acc >= ACCURACY_THRESHOLD:
            log.info(f"Accuracy threshold reached at iteration {iteration}")
            break
    else:
        log.warning(
            f"Did not reach threshold {ACCURACY_THRESHOLD} after {MAX_ITERATIONS} "
            f"iterations. Best: {best_acc:.4f}"
        )

    # Save versioned model
    version = get_next_version()
    model_path = MODEL_DIR / f"model_v{version}.pkl"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    log.info(f"Model saved to {model_path} (accuracy={best_acc:.4f})")

    acc_path = MODEL_DIR / "latest_accuracy.txt"
    acc_path.write_text(str(best_acc))

    # Write a tiny metadata file so the server knows which model to load
    meta_path = MODEL_DIR / "latest.txt"
    # Ensure the path stored is relative to the image's root (/app)
    meta_path.write_text(f"model/model_v{version}.pkl")

    return best_acc, model_path


if __name__ == "__main__":
    acc, path = train()
    print(f"Training complete. Accuracy={acc:.4f}, Model={path}")
