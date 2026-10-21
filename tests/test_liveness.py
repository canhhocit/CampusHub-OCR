import io
import time
import cv2
import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.schemas.liveness import ChallengeAction
from app.services.liveness_service import (
    LivenessService,
    LivenessSessionState,
    MockLivenessDetector,
    get_liveness_service,
    set_liveness_service,
)


@pytest.fixture
def mock_liveness():
    """Provides an isolated LivenessService with a Mock detector for deterministic testing."""
    original_service = get_liveness_service()
    mock_detector = MockLivenessDetector()
    service = LivenessService(detector=mock_detector)
    set_liveness_service(service)
    yield service, mock_detector
    set_liveness_service(original_service)


def create_synthetic_frame_bytes(width: int = 400, height: int = 400) -> bytes:
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    # Add simple gradient to ensure realistic byte payload
    cv2.circle(arr, (width // 2, height // 2), 50, (200, 200, 200), -1)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# --- 1. Session Lifecycle Tests ---

def test_liveness_create_session(client, mock_liveness):
    service, _ = mock_liveness
    res = client.post("/api/v1/liveness/session/create")
    assert res.status_code == 200
    data = res.json()

    assert data["session_id"].startswith("live_")
    assert data["total_steps"] == 2
    assert data["current_step"] == 1
    assert data["expires_in_seconds"] > 0
    assert data["challenge"]["action"] in [
        ChallengeAction.TURN_HEAD_LEFT.value,
        ChallengeAction.TURN_HEAD_RIGHT.value,
    ]
    assert data["challenge"]["instruction"] != ""


def test_liveness_get_session_status(client, mock_liveness):
    service, _ = mock_liveness
    create_res = client.post("/api/v1/liveness/session/create")
    session_id = create_res.json()["session_id"]

    status_res = client.get(f"/api/v1/liveness/session/{session_id}")
    assert status_res.status_code == 200
    data = status_res.json()
    assert data["session_id"] == session_id
    assert data["status"] == "IN_PROGRESS"
    assert data["passed"] is False
    assert data["total_steps"] == 2
    assert "prototype" in data["disclaimer"] or "thử nghiệm" in data["disclaimer"]


def test_liveness_get_nonexistent_session_404(client, mock_liveness):
    res = client.get("/api/v1/liveness/session/live_nonexistent_token")
    assert res.status_code == 404
    assert "không tồn tại" in res.json()["detail"]


def test_liveness_session_expiration(client, mock_liveness):
    service, _ = mock_liveness
    create_res = client.post("/api/v1/liveness/session/create")
    session_id = create_res.json()["session_id"]

    # Force expiration by setting expires_at in the past
    session = service._sessions[session_id]
    session.expires_at = time.time() - 10

    # Query status
    status_res = client.get(f"/api/v1/liveness/session/{session_id}")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "EXPIRED"

    # Attempting to submit frames to expired session
    frame_bytes = create_synthetic_frame_bytes()
    verify_res = client.post(
        f"/api/v1/liveness/session/{session_id}/verify-step",
        files=[("frames", ("f1.jpg", frame_bytes, "image/jpeg")) for _ in range(4)],
    )
    assert verify_res.status_code == 200
    data = verify_res.json()
    assert data["step_passed"] is False
    assert data["status"] == "EXPIRED"
    assert data["reason_code"] == "SESSION_EXPIRED"


# --- 2. Frame Sequence Validation & Input Boundary Tests ---

def test_liveness_verify_insufficient_frames_rejected(client, mock_liveness):
    create_res = client.post("/api/v1/liveness/session/create")
    session_id = create_res.json()["session_id"]

    frame_bytes = create_synthetic_frame_bytes()
    # Sending only 2 frames (< LIVENESS_MIN_FRAMES 3)
    res = client.post(
        f"/api/v1/liveness/session/{session_id}/verify-step",
        files=[
            ("frames", ("f1.jpg", frame_bytes, "image/jpeg")),
            ("frames", ("f2.jpg", frame_bytes, "image/jpeg")),
        ],
    )
    assert res.status_code == 400
    assert "không đủ" in res.json()["detail"]


def test_liveness_verify_excessive_frames_rejected(client, mock_liveness):
    create_res = client.post("/api/v1/liveness/session/create")
    session_id = create_res.json()["session_id"]

    frame_bytes = create_synthetic_frame_bytes()
    # Sending 16 frames (> LIVENESS_MAX_FRAMES 15)
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(16)]
    res = client.post(
        f"/api/v1/liveness/session/{session_id}/verify-step",
        files=files,
    )
    assert res.status_code == 400
    assert "vượt quá" in res.json()["detail"]


def test_liveness_verify_invalid_session_404(client, mock_liveness):
    frame_bytes = create_synthetic_frame_bytes()
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res = client.post(
        "/api/v1/liveness/session/live_invalid_12345/verify-step",
        files=files,
    )
    assert res.status_code == 404


# --- 3. Biometric Invariant Tests (Face count & static presentation attacks) ---

def test_liveness_no_face_detected_in_frame(client, mock_liveness):
    service, detector = mock_liveness
    create_res = client.post("/api/v1/liveness/session/create")
    session_id = create_res.json()["session_id"]

    # Frame 0, 1, 2 have faces, but frame 3 has 0 faces
    detector.set_preset_results([
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}],
        [],  # 0 faces in frame 3!
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}],
    ])

    frame_bytes = create_synthetic_frame_bytes()
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["step_passed"] is False
    assert data["reason_code"] == "NO_FACE_DETECTED"
    assert "Không tìm thấy khuôn mặt" in data["message"]


def test_liveness_multiple_faces_detected(client, mock_liveness):
    service, detector = mock_liveness
    create_res = client.post("/api/v1/liveness/session/create")
    session_id = create_res.json()["session_id"]

    # Frame has 2 faces
    detector.set_preset_results([
        [
            {"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]},
            {"box": (200, 50, 100, 100, 0.99), "landmarks": [(220, 70), (280, 70), (250, 100), (230, 130), (270, 130)]},
        ],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}],
    ])

    frame_bytes = create_synthetic_frame_bytes()
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["step_passed"] is False
    assert data["reason_code"] == "MULTIPLE_FACES_DETECTED"
    assert "nhiều khuôn mặt" in data["message"]


def test_liveness_static_image_presentation_attack_rejected(client, mock_liveness):
    """
    Submitting a sequence of identical frames (e.g. holding up a photo) has 0 motion delta.
    The system MUST reject this presentation attack with STATIC_IMAGE_DETECTED.
    """
    service, detector = mock_liveness
    create_res = client.post("/api/v1/liveness/session/create")
    session_id = create_res.json()["session_id"]

    # All frames have the exact same frontal landmarks (yaw_ratio = 0.50, delta = 0.0)
    static_face = [
        {"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}
    ]
    detector.set_preset_results([static_face, static_face, static_face, static_face])

    frame_bytes = create_synthetic_frame_bytes()
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["step_passed"] is False
    assert data["reason_code"] == "STATIC_IMAGE_DETECTED"
    assert "khung hình tĩnh" in data["message"]


# --- 4. Active Challenge Dynamics & Progression Tests ---

def test_liveness_challenge_mismatch_rejected(client, mock_liveness):
    """
    When challenge requires TURN_HEAD_LEFT, but user turns head RIGHT.
    The system must detect the mismatch and explain it clearly.
    """
    service, detector = mock_liveness

    # Set up session with specific challenge TURN_HEAD_LEFT
    session_id = "live_test_mismatch"
    service._sessions[session_id] = LivenessSessionState(
        session_id=session_id,
        challenges=[ChallengeAction.TURN_HEAD_LEFT, ChallengeAction.TURN_HEAD_RIGHT],
    )

    # Simulate head turning RIGHT:
    # Right eye x=70, left eye x=130 (inter-eye=60).
    # Nose starts at 100 (yaw = (100-70)/60 = 0.50), then moves to 80 (yaw = 10/60 = 0.166 < 0.35)
    detector.set_preset_results([
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (95, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (88, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (80, 100), (80, 130), (120, 130)]}],
    ])

    frame_bytes = create_synthetic_frame_bytes()
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["step_passed"] is False
    assert data["reason_code"] == "CHALLENGE_MISMATCH"
    assert "PHẢI" in data["message"]


def test_liveness_successful_full_challenge_flow(client, mock_liveness):
    """
    Simulates a full 2-step active challenge flow:
    Step 1: TURN_HEAD_LEFT
    Step 2: TURN_HEAD_RIGHT
    Verifies that session transitions to PASSED and replay is blocked.
    """
    service, detector = mock_liveness

    session_id = "live_test_full_pass"
    service._sessions[session_id] = LivenessSessionState(
        session_id=session_id,
        challenges=[ChallengeAction.TURN_HEAD_LEFT, ChallengeAction.TURN_HEAD_RIGHT],
    )

    frame_bytes = create_synthetic_frame_bytes()

    # --- STEP 1: Execute TURN_HEAD_LEFT ---
    # Right eye x=70, left eye x=130 (inter-eye=60).
    # Nose starts at 100 (yaw 0.50), then moves to 115 (yaw = 45/60 = 0.75 >= 0.65)
    detector.set_preset_results([
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (105, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (110, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (115, 100), (80, 130), (120, 130)]}],
    ])

    files_step1 = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res1 = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files_step1)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["step_passed"] is True
    assert data1["all_passed"] is False
    assert data1["current_step"] == 1
    assert data1["status"] == "IN_PROGRESS"
    assert data1["next_challenge"]["action"] == ChallengeAction.TURN_HEAD_RIGHT.value

    # --- STEP 2: Execute TURN_HEAD_RIGHT ---
    # Nose starts near 100 (yaw 0.50), then moves to 80 (yaw = 10/60 = 0.166 <= 0.35)
    detector.set_preset_results([
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (92, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (85, 100), (80, 130), (120, 130)]}],
        [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (80, 100), (80, 130), (120, 130)]}],
    ])

    files_step2 = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res2 = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files_step2)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["step_passed"] is True
    assert data2["all_passed"] is True
    assert data2["status"] == "PASSED"
    assert data2["reason_code"] == "LIVENESS_VERIFIED"
    assert "thành công" in data2["message"]

    # Verify session status endpoint reports PASSED
    status_res = client.get(f"/api/v1/liveness/session/{session_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["status"] == "PASSED"
    assert status_data["passed"] is True
    assert len(status_data["history"]) == 2

    # --- REPLAY TEST: Attempting to verify-step on an already completed session ---
    replay_res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files_step2)
    assert replay_res.status_code == 200
    replay_data = replay_res.json()
    assert replay_data["step_passed"] is False
    assert replay_data["reason_code"] == "SESSION_ALREADY_COMPLETED"
    assert "ngăn chặn" in replay_data["message"]


def test_liveness_max_step_attempts_exhausted(client, mock_liveness):
    """Failing a step 3 times causes the session to permanently fail with MAX_ATTEMPTS_EXCEEDED."""
    service, detector = mock_liveness
    create_res = client.post("/api/v1/liveness/session/create")
    session_id = create_res.json()["session_id"]

    # Static frame sequence produces STATIC_IMAGE_DETECTED
    static_face = [{"box": (50, 50, 100, 100, 0.99), "landmarks": [(70, 70), (130, 70), (100, 100), (80, 130), (120, 130)]}]
    detector.set_preset_results([static_face] * 4)

    frame_bytes = create_synthetic_frame_bytes()
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]

    # Attempt 1
    r1 = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)
    assert r1.json()["status"] == "IN_PROGRESS"
    assert "Lần thử 1/3" in r1.json()["message"]

    # Attempt 2
    detector.set_preset_results([static_face] * 4)
    r2 = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)
    assert r2.json()["status"] == "IN_PROGRESS"
    assert "Lần thử 2/3" in r2.json()["message"]

    # Attempt 3 -> Fails session
    detector.set_preset_results([static_face] * 4)
    r3 = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)
    assert r3.json()["status"] == "FAILED"
    assert r3.json()["reason_code"] == "MAX_ATTEMPTS_EXCEEDED"


def test_real_yunet_liveness_detector_initialization():
    """YuNetLivenessDetector initializes properly with default OpenCV Zoo model."""
    from app.services.liveness_service import YuNetLivenessDetector
    detector = YuNetLivenessDetector()
    assert detector._detector is not None

    # Test detection on synthetic blank image (should return 0 faces without crashing)
    blank_img = np.zeros((320, 320, 3), dtype=np.uint8)
    res = detector.detect_face_and_landmarks(blank_img)
    assert isinstance(res, list)
    assert len(res) == 0


def test_missing_yunet_liveness_model_raises_filenotfound():
    """YuNetLivenessDetector raises FileNotFoundError when given invalid path."""
    from app.services.liveness_service import YuNetLivenessDetector
    with pytest.raises(FileNotFoundError):
        YuNetLivenessDetector(model_path="models/face/nonexistent_liveness.onnx")

