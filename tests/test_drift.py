# tests/test_drift.py
"""Test distribution drift detection logic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.drift_detector import DriftDetector


def test_no_drift_similar_distribution():
    """Data with same distribution should not trigger drift."""
    detector = DriftDetector(threshold=0.5)
    baseline = {
        "age":    [25, 30, 28, 32, 27],
        "income": [50000, 55000, 52000, 48000, 53000]
    }
    detector.set_baseline(baseline)

    # Slightly different but within threshold
    current = {
        "age":    [26, 31, 29, 33, 28],
        "income": [51000, 56000, 53000, 49000, 54000]
    }
    drift, cols = detector.detect(current)
    assert drift is False, f"Should not detect drift, but got drifted cols: {cols}"


def test_drift_detected_large_shift():
    """Data with a large mean shift should trigger drift."""
    detector = DriftDetector(threshold=0.2)
    baseline = {
        "age": [25, 30, 28, 32, 27]
    }
    detector.set_baseline(baseline)

    # Mean shifts from ~28 to ~80 — clearly above threshold
    shifted = {
        "age": [80, 85, 82, 88, 83]
    }
    drift, cols = detector.detect(shifted)
    assert drift is True, "Should detect drift for large mean shift"
    assert "age" in cols


def test_drift_baseline_set_on_first_call():
    """First call should set baseline and return no drift."""
    detector = DriftDetector(threshold=0.2)
    first_batch = {"temperature": [20, 22, 21, 23, 20]}
    drift, cols = detector.detect(first_batch)
    assert drift is False, "First call should always return no drift"
    assert detector.baseline != {}
