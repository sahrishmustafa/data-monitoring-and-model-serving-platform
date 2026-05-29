# ingestion/drift_detector.py
"""
Detects distribution drift by comparing the mean and std of each feature
in the new batch against a stored baseline. If the z-score of the mean
difference exceeds the threshold for any feature, drift is flagged.
"""
import numpy as np
import os

# How many standard deviations away counts as drift (configurable via env)
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


class DriftDetector:
    def __init__(self, threshold: float = DRIFT_THRESHOLD):
        self.threshold = threshold
        # Baseline stats: {feature_name: {"mean": float, "std": float}}
        self.baseline: dict = {}

    def set_baseline(self, data: dict[str, list]) -> None:
        """
        Store the initial distribution of each numeric feature.
        Call this on the first successful batch.
        """
        self.baseline = {}
        for col, values in data.items():
            arr = np.array(values, dtype=float)
            if arr.size > 0:
                self.baseline[col] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)) or 1.0  # avoid division by zero
                }

    def detect(self, data: dict[str, list]) -> tuple[bool, list[str]]:
        """
        Compare current batch statistics to the baseline.
        Returns (drift_detected: bool, drifted_features: list[str]).
        """
        if not self.baseline:
            # No baseline yet — set it and report no drift
            self.set_baseline(data)
            return False, []

        drifted = []
        for col, values in data.items():
            if col not in self.baseline:
                continue  # new column — handled by schema monitor
            arr = np.array(values, dtype=float)
            if arr.size == 0:
                continue
            current_mean = float(np.mean(arr))
            baseline_mean = self.baseline[col]["mean"]
            baseline_std = self.baseline[col]["std"]
            # Compute normalised shift
            shift = abs(current_mean - baseline_mean) / baseline_std
            if shift > self.threshold:
                drifted.append(col)

        return len(drifted) > 0, drifted
