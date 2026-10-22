"""
tests/test_anti_spoofing.py
===========================
Comprehensive unit, API, and real-model regression tests for Passive Face Anti-Spoofing (PAD).

Test Categories:
1. Deterministic Unit Tests (Mocked Detector):
   - LIVE, SPOOF, UNCERTAIN, and ERROR outcomes.
   - Fail-closed behavior on technical errors.
   - Model unavailability handling.
2. Active Liveness + Passive PAD Cooperation (API Tests):
   - Prerecorded video attack with valid head-turn is rejected by anti-spoofing.
   - Uncertain anti-spoofing fails step without granting pass.
   - Technical inference error fails closed.
   - Genuine user with LIVE passive PAD and valid head turn passes.
3. Real MiniFASNet ONNX Model Tests (Real Inference):
   - Loading real models from models/face/.
   - Validating bounding-box expansion geometry (scale 2.7 and 4.0).
   - Real inference probability distribution and execution on CPU.
"""

import io
import time
import os
import cv2
import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from app.main import app
from app.config import settings
from app.schemas.liveness import (
    ChallengeAction,
    AntiSpoofingOutcome,
)
from app.services.liveness_service import (
    LivenessService,
    LivenessSessionState,
    MockLivenessDetector,
    get_liveness_service,
    set_liveness_service,
)
from app.services.anti_spoofing_service import (
    AntiSpoofingResult,
    BaseAntiSpoofingDetector,
    MockAntiSpoofingDetector,
    MiniFASNetAntiSpoofingDetector,
    get_anti_spoofing_detector,
    set_anti_spoofing_detector,
)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def create_test_frame_bytes(width: int = 640, height: int = 480) -> bytes:
    arr = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.circle(arr, (width // 2, height // 2), 60, (200, 200, 200), -1)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# =========================================================================
# 1. Deterministic Unit Tests for AntiSpoofing Outcomes (Mocked)
# =========================================================================

def test_anti_spoofing_live_outcome():
    """Real face with high score is classified as LIVE."""
    mock = MockAntiSpoofingDetector(default_outcome=AntiSpoofingOutcome.LIVE)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = mock.predict(dummy_img, (200, 100, 240, 260))

    assert result.outcome == AntiSpoofingOutcome.LIVE
    assert result.is_live is True
    assert result.score_real >= settings.ANTI_SPOOFING_THRESHOLD_LIVE


def test_anti_spoofing_spoof_outcome():
    """Screen or print attack with high spoof score is classified as SPOOF."""
    mock = MockAntiSpoofingDetector(default_outcome=AntiSpoofingOutcome.SPOOF)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = mock.predict(dummy_img, (200, 100, 240, 260))

    assert result.outcome == AntiSpoofingOutcome.SPOOF
    assert result.is_live is False
    assert result.score_screen >= 0.70 or result.score_paper >= 0.70


def test_anti_spoofing_uncertain_outcome():
    """Score falling into intermediate range is classified as UNCERTAIN."""
    mock = MockAntiSpoofingDetector(default_outcome=AntiSpoofingOutcome.UNCERTAIN)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = mock.predict(dummy_img, (200, 100, 240, 260))

    assert result.outcome == AntiSpoofingOutcome.UNCERTAIN
    assert result.is_live is False


def test_anti_spoofing_technical_error_fails_closed():
    """Inference failure returns ERROR and does not claim LIVE."""
    mock = MockAntiSpoofingDetector(default_outcome=AntiSpoofingOutcome.ERROR)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = mock.predict(dummy_img, (200, 100, 240, 260))

    assert result.outcome == AntiSpoofingOutcome.ERROR
    assert result.is_live is False
    assert result.error_message is not None


def test_anti_spoofing_missing_model_file_unavailability():
    """MiniFASNet detector gracefully handles missing model file with is_available=False."""
    detector = MiniFASNetAntiSpoofingDetector(
        model_path_v2="models/face/nonexistent_v2.onnx",
        model_path_v1se="models/face/nonexistent_v1se.onnx",
    )
    assert detector.is_available is False

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.predict(dummy_img, (200, 100, 240, 260))
    assert result.outcome == AntiSpoofingOutcome.ERROR
    assert result.is_live is False
    assert "unavailable" in result.error_message.lower()


# =========================================================================
# 2. Integration with Active Liveness Challenge (Cooperation & Security)
# =========================================================================

def test_liveness_step_spoof_detected_rejects_even_with_valid_head_turn(client):
    """
    CRITICAL SECURITY INVARIANT:
    A prerecorded video where the attacker turns their head left satisfies the active motion yaw check,
    BUT passive anti-spoofing classifies the video presentation attack as SPOOF.
    The challenge MUST be rejected with SPOOF_DETECTED.
    """
    original_service = get_liveness_service()
    mock_liveness_det = MockLivenessDetector()
    mock_pad_det = MockAntiSpoofingDetector(default_outcome=AntiSpoofingOutcome.SPOOF)

    service = LivenessService(
        detector=mock_liveness_det,
        anti_spoofing_detector=mock_pad_det,
    )
    set_liveness_service(service)

    try:
        session_id = "test_spoof_rejection"
        service._sessions[session_id] = LivenessSessionState(
            session_id=session_id,
            challenges=[ChallengeAction.TURN_HEAD_LEFT],
        )

        # 4 frames: Centered face, perfectly valid yaw motion turning left
        box = (200, 100, 240, 260, 0.99)
        frames_data = [
            [{"box": box, "landmarks": [(260, 180), (360, 180), (310, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (325, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (335, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (340, 230), (270, 300), (350, 300)]}],
        ]
        mock_liveness_det.set_preset_results(frames_data)

        frame_bytes = create_test_frame_bytes()
        files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
        res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

        assert res.status_code == 200
        data = res.json()
        assert data["step_passed"] is False
        assert data["reason_code"] == "SPOOF_DETECTED"
        assert "giả mạo" in data["message"].lower()
        assert data["anti_spoofing_outcome"] == "SPOOF"

    finally:
        set_liveness_service(original_service)


def test_liveness_step_uncertain_anti_spoofing_fails_step(client):
    """When anti-spoofing outcome is UNCERTAIN, verification does not pass."""
    original_service = get_liveness_service()
    mock_liveness_det = MockLivenessDetector()
    mock_pad_det = MockAntiSpoofingDetector(default_outcome=AntiSpoofingOutcome.UNCERTAIN)

    service = LivenessService(
        detector=mock_liveness_det,
        anti_spoofing_detector=mock_pad_det,
    )
    set_liveness_service(service)

    try:
        session_id = "test_uncertain_rejection"
        service._sessions[session_id] = LivenessSessionState(
            session_id=session_id,
            challenges=[ChallengeAction.TURN_HEAD_LEFT],
        )

        box = (200, 100, 240, 260, 0.99)
        frames_data = [
            [{"box": box, "landmarks": [(260, 180), (360, 180), (310, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (340, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (340, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (340, 230), (270, 300), (350, 300)]}],
        ]
        mock_liveness_det.set_preset_results(frames_data)

        frame_bytes = create_test_frame_bytes()
        files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
        res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

        assert res.status_code == 200
        data = res.json()
        assert data["step_passed"] is False
        assert data["reason_code"] == "UNCERTAIN_LIVENESS"
        assert data["anti_spoofing_outcome"] == "UNCERTAIN"

    finally:
        set_liveness_service(original_service)


def test_liveness_step_anti_spoofing_technical_error_fails_closed(client):
    """When anti-spoofing fails with ERROR, verification fails closed."""
    original_service = get_liveness_service()
    mock_liveness_det = MockLivenessDetector()
    mock_pad_det = MockAntiSpoofingDetector(default_outcome=AntiSpoofingOutcome.ERROR)

    service = LivenessService(
        detector=mock_liveness_det,
        anti_spoofing_detector=mock_pad_det,
    )
    set_liveness_service(service)

    try:
        session_id = "test_error_rejection"
        service._sessions[session_id] = LivenessSessionState(
            session_id=session_id,
            challenges=[ChallengeAction.TURN_HEAD_LEFT],
        )

        box = (200, 100, 240, 260, 0.99)
        frames_data = [
            [{"box": box, "landmarks": [(260, 180), (360, 180), (310, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (340, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (340, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (340, 230), (270, 300), (350, 300)]}],
        ]
        mock_liveness_det.set_preset_results(frames_data)

        frame_bytes = create_test_frame_bytes()
        files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
        res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

        assert res.status_code == 200
        data = res.json()
        assert data["step_passed"] is False
        assert data["reason_code"] == "ANTI_SPOOFING_ERROR"
        assert data["anti_spoofing_outcome"] == "ERROR"

    finally:
        set_liveness_service(original_service)


def test_liveness_step_genuine_live_and_valid_head_turn_passes(client):
    """When passive anti-spoofing is LIVE AND head-turn matches, step passes."""
    original_service = get_liveness_service()
    mock_liveness_det = MockLivenessDetector()
    mock_pad_det = MockAntiSpoofingDetector(default_outcome=AntiSpoofingOutcome.LIVE)

    service = LivenessService(
        detector=mock_liveness_det,
        anti_spoofing_detector=mock_pad_det,
    )
    set_liveness_service(service)

    try:
        session_id = "test_live_pass"
        service._sessions[session_id] = LivenessSessionState(
            session_id=session_id,
            challenges=[ChallengeAction.TURN_HEAD_LEFT],
        )

        box = (200, 100, 240, 260, 0.99)
        frames_data = [
            [{"box": box, "landmarks": [(260, 180), (360, 180), (310, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (325, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (335, 230), (270, 300), (350, 300)]}],
            [{"box": box, "landmarks": [(260, 180), (360, 180), (340, 230), (270, 300), (350, 300)]}],
        ]
        mock_liveness_det.set_preset_results(frames_data)

        frame_bytes = create_test_frame_bytes()
        files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
        res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

        assert res.status_code == 200
        data = res.json()
        assert data["step_passed"] is True
        assert data["all_passed"] is True
        assert data["anti_spoofing_outcome"] == "LIVE"

    finally:
        set_liveness_service(original_service)


# =========================================================================
# 3. Real MiniFASNet ONNX Model Tests (Non-mocked, Actual Inference)
# =========================================================================

def test_real_minifasnet_initialization():
    """Real MiniFASNet detector loads both V2 and V1SE ONNX models successfully."""
    detector = MiniFASNetAntiSpoofingDetector()
    assert detector.is_available is True
    assert detector._net_v2 is not None
    assert detector._net_v1se is not None


def test_real_minifasnet_bounding_box_expansion_geometry():
    """Validates _get_new_box expansion calculation for scale 2.7 and 4.0."""
    src_w, src_h = 640, 480
    bbox = (200, 100, 100, 100)

    # Scale 2.7
    x1, y1, x2, y2 = MiniFASNetAntiSpoofingDetector._get_new_box(src_w, src_h, bbox, scale=2.7)
    assert x1 >= 0 and y1 >= 0
    assert x2 <= src_w - 1 and y2 <= src_h - 1
    # Center should remain approximately 250, 150
    assert abs((x1 + x2) / 2.0 - 250) <= 2
    assert abs((y1 + y2) / 2.0 - 150) <= 2

    # Scale 4.0
    x1_4, y1_4, x2_4, y2_4 = MiniFASNetAntiSpoofingDetector._get_new_box(src_w, src_h, bbox, scale=4.0)
    assert x1_4 <= x1 and y1_4 <= y1
    assert x2_4 >= x2 and y2_4 >= y2


def test_real_minifasnet_inference_execution_and_speed():
    """
    Executes real inference through both ONNX models on CPU.
    Verifies that probabilities sum to 1.0 and latency is well under 50ms per frame.
    """
    detector = MiniFASNetAntiSpoofingDetector()
    assert detector.is_available is True

    test_image = np.full((480, 640, 3), 128, dtype=np.uint8)
    cv2.circle(test_image, (320, 240), 70, (220, 180, 160), -1)
    bbox = (220, 140, 200, 200)

    t0 = time.time()
    result = detector.predict(test_image, bbox)
    t1 = time.time()

    latency_ms = (t1 - t0) * 1000.0
    assert latency_ms < 50.0  # Must be fast on CPU (typically ~2-10ms)

    assert result.outcome in [
        AntiSpoofingOutcome.LIVE,
        AntiSpoofingOutcome.SPOOF,
        AntiSpoofingOutcome.UNCERTAIN,
    ]
    total_score = result.score_real + result.score_paper + result.score_screen
    assert 0.99 <= total_score <= 1.01
