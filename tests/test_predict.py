# tests/test_predict.py
"""Test the /predict FastAPI endpoint with a mocked model."""
import sys
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_predict_returns_correct_schema():
    """
    Mock the model, call /predict, and assert the response has
    the expected 'prediction' and 'confidence' keys.
    """
    # Create a mock sklearn-compatible model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    # Patch load_latest_model so the app uses our mock
    with patch("serving.app.load_latest_model", return_value=mock_model):
        from serving.app import app
        client = TestClient(app)
        response = client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0, 4.0]}
        )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data, "Response must contain 'prediction'"
    assert "confidence" in data, "Response must contain 'confidence'"
    assert isinstance(data["confidence"], float)


def test_health_endpoint():
    """Health endpoint should always return 200 with status ok."""
    with patch("serving.app.load_latest_model", return_value=MagicMock()):
        from serving.app import app
        client = TestClient(app)
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
