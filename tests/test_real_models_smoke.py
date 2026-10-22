"""
tests/test_real_models_smoke.py
===============================
Deployment Readiness and Real Model Smoke Tests.

Validates that:
1. Real ONNX model weights (YuNet, SFace, MiniFASNet V2, MiniFASNet V1SE) can be loaded
   and executed via native OpenCV on CPU without mock injection.
2. System health check endpoints (/health, /face/health, /liveness/health) accurately
   report operational status, model availability, and configuration parameters.
3. Missing or corrupted models fail-closed safely without returning fake LIVE outcomes.
4. Temporary frame processing leaves zero persisted image files or leaked biometric data on disk.
"""

import os
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.anti_spoofing_service import (
    MiniFASNetAntiSpoofingDetector,
    AntiSpoofingOutcome,
)
from app.services.face_verifier import get_face_verifier, YuNetSFaceEngine
from app.services.liveness_service import get_liveness_service, YuNetLivenessDetector


@pytest.fixture
def test_client():
    return TestClient(app)


def test_ocr_health_endpoint(test_client):
    """GET /api/v1/health reports engine status and device."""
    resp = test_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["ocr_engine"] == "PaddleOCR"
    assert data["device"] == "CPU"


def test_face_health_endpoint_with_real_models(test_client):
    """GET /api/v1/face/health reports YuNet & SFace models loaded."""
    resp = test_client.get("/api/v1/face/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert data["engine_type"] == "real"
    assert "YuNet" in data["detector"]
    assert "SFace" in data["recognizer"]


def test_liveness_health_endpoint_with_real_models(test_client):
    """GET /api/v1/liveness/health reports both Active Liveness and Passive PAD ready."""
    resp = test_client.get("/api/v1/liveness/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "active_liveness_ready" in data
    assert "passive_anti_spoofing_ready" in data
    assert "fail_closed" in data
    assert data["fail_closed"] is True
    assert "thresholds" in data
    assert "pad_threshold_live" in data["thresholds"]
    assert "models" in data
    assert "yunet" in data["models"]
    assert "anti_spoofing" in data["models"]


def test_real_yunet_detector_forward_pass():
    """Verify real YuNet model executes on CPU with OpenCV FaceDetectorYN."""
    yunet_path = os.path.join(settings.FACE_MODEL_DIR, settings.FACE_DETECTOR_MODEL)
    if not os.path.exists(yunet_path):
        pytest.skip(f"YuNet model file not found at {yunet_path}")

    detector = YuNetLivenessDetector(yunet_path)
    dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
    # Forward pass on blank image should safely return empty list without crash
    faces = detector.detect_face_and_landmarks(dummy_img)
    assert isinstance(faces, list)
    assert len(faces) == 0


def test_real_sface_recognizer_forward_pass():
    """Verify real SFace model executes on CPU with OpenCV FaceRecognizerSF."""
    sface_path = os.path.join(settings.FACE_MODEL_DIR, settings.FACE_RECOGNIZER_MODEL)
    if not os.path.exists(sface_path):
        pytest.skip(f"SFace model file not found at {sface_path}")

    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    dummy_crop = np.zeros((112, 112, 3), dtype=np.uint8)
    feature = recognizer.feature(dummy_crop)
    assert feature is not None
    assert feature.shape[1] == 128  # 128-dimensional embedding


def test_real_minifasnet_dual_scale_forward_pass():
    """Verify real MiniFASNet dual-scale models execute via OpenCV DNN on CPU."""
    v2_path = settings.ANTI_SPOOFING_MODEL_PATH_V2
    v1se_path = settings.ANTI_SPOOFING_MODEL_PATH_V1SE

    if not os.path.exists(v2_path) or not os.path.exists(v1se_path):
        pytest.skip("MiniFASNet models not found locally")

    detector = MiniFASNetAntiSpoofingDetector(v2_path, v1se_path)
    assert detector.is_available is True

    # Create dummy face frame
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    face_box = (150, 100, 200, 220)

    result = detector.predict(img, face_box)
    assert result.outcome in (
        AntiSpoofingOutcome.LIVE,
        AntiSpoofingOutcome.SPOOF,
        AntiSpoofingOutcome.UNCERTAIN,
    )
    # Probabilities must sum approximately to 1.0
    total_prob = result.score_real + result.score_paper + result.score_screen
    assert abs(total_prob - 1.0) < 1e-4


def test_missing_model_fail_closed_safe_behavior():
    """Verify that pointing to nonexistent model files triggers ERROR and never fake LIVE."""
    missing_detector = MiniFASNetAntiSpoofingDetector(
        model_path_v2="models/face/nonexistent_v2.onnx",
        model_path_v1se="models/face/nonexistent_v1se.onnx",
    )
    assert missing_detector.is_available is False
    assert missing_detector.init_error is not None

    img = np.zeros((320, 320, 3), dtype=np.uint8)
    res = missing_detector.predict(img, (50, 50, 100, 100))

    assert res.outcome == AntiSpoofingOutcome.ERROR
    assert res.is_live is False
    assert "unavailable" in (res.error_message or "").lower()


def test_zero_disk_leakage_after_processing(test_client):
    """Verify that processing liveness requests leaves no temporary images in the repository."""
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")

    # Baseline file counts
    initial_uploads = len(os.listdir(uploads_dir)) if os.path.exists(uploads_dir) else 0
    initial_temp = len(os.listdir(temp_dir)) if os.path.exists(temp_dir) else 0

    # Create session
    create_resp = test_client.post("/api/v1/liveness/session/create")
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    # Post invalid frames to trigger error handling
    test_client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=[])

    current_uploads = len(os.listdir(uploads_dir)) if os.path.exists(uploads_dir) else 0
    current_temp = len(os.listdir(temp_dir)) if os.path.exists(temp_dir) else 0

    assert current_uploads == initial_uploads
    assert current_temp == initial_temp
