import io
import time
import cv2
import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.schemas.liveness import ChallengeAction, AntiSpoofingOutcome
from app.services.liveness_service import (
    LivenessService,
    LivenessSessionState,
    MockLivenessDetector,
    FaceFramingValidator,
    FramingValidationResult,
    get_liveness_service,
    set_liveness_service,
)
from app.services.anti_spoofing_service import MockAntiSpoofingDetector


@pytest.fixture
def mock_liveness():
    """Provides an isolated LivenessService with a Mock detector for deterministic testing."""
    original_service = get_liveness_service()
    mock_detector = MockLivenessDetector()
    mock_anti_spoofing = MockAntiSpoofingDetector(default_outcome=AntiSpoofingOutcome.LIVE)
    service = LivenessService(detector=mock_detector, anti_spoofing_detector=mock_anti_spoofing)
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


# =========================================================================
# 5. Face-Framing Validation Regression Tests (Task Requirements)
# =========================================================================

def test_face_framing_validator_full_face_correctly_positioned():
    """1. Full face correctly positioned with normal proportions passes framing validation."""
    # 640x480 frame, centered face: box at (200, 100), size (240, 260) -> center (320, 230) -> (0.50, 0.479)
    box = (200, 100, 240, 260, 0.99)
    landmarks = [
        (260, 180),  # right eye
        (380, 180),  # left eye
        (320, 230),  # nose tip
        (270, 300),  # right mouth
        (370, 300),  # left mouth
    ]
    res = FaceFramingValidator.validate_frame(box, landmarks, img_w=640, img_h=480)
    assert res.is_valid is True
    assert res.reason_code is None
    assert res.details["center_x"] == 0.5
    assert 0.40 <= res.details["center_y"] <= 0.55


def test_face_framing_validator_face_shifted_too_far_down_rejected(client, mock_liveness):
    """2. Face shifted too far down is rejected with FACE_OUT_OF_GUIDE."""
    # 640x480 frame, face located near bottom: center_y = (340 + 60) / 480 = 0.833 > 0.65
    box = (220, 340, 200, 120, 0.99)
    landmarks = [
        (270, 370),
        (370, 370),
        (320, 400),
        (280, 430),
        (360, 430),
    ]
    # Unit validator test
    res = FaceFramingValidator.validate_frame(box, landmarks, img_w=640, img_h=480)
    assert res.is_valid is False
    assert res.reason_code == "FACE_OUT_OF_GUIDE"
    assert "quá thấp" in res.message

    # End-to-end API test
    service, detector = mock_liveness
    create_res = client.post("/api/v1/liveness/session/create")
    session_id = create_res.json()["session_id"]

    detector.set_preset_results([[{"box": box, "landmarks": landmarks}]] * 4)
    frame_bytes = create_synthetic_frame_bytes(width=640, height=480)
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    api_res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

    assert api_res.status_code == 200
    data = api_res.json()
    assert data["step_passed"] is False
    assert data["reason_code"] == "FACE_OUT_OF_GUIDE"
    assert "quá thấp" in data["message"]


def test_face_framing_validator_only_upper_face_or_eyes_in_guide_rejected(client, mock_liveness):
    """
    3. User ducks down so only eyes/upper face appear in guide oval (original failure reproduction).
    The bottom of the face / mouth is near or crossing the frame bottom.
    System MUST reject with FACE_PARTIALLY_VISIBLE or FACE_OUT_OF_GUIDE.
    """
    # Chin/mouth crosses frame bottom boundary: box y=320, h=160 -> bottom = 480 (edge)
    box = (220, 320, 200, 160, 0.99)
    landmarks = [
        (270, 360),
        (370, 360),
        (320, 420),
        (280, 475),  # mouth near frame bottom
        (360, 475),
    ]
    # Unit test
    res = FaceFramingValidator.validate_frame(box, landmarks, img_w=640, img_h=480)
    assert res.is_valid is False
    assert res.reason_code in ["FACE_PARTIALLY_VISIBLE", "FACE_OUT_OF_GUIDE"]

    # End-to-end API test verifying bug cannot pass
    service, detector = mock_liveness
    create_res = client.post("/api/v1/liveness/session/create")
    session_id = create_res.json()["session_id"]

    detector.set_preset_results([[{"box": box, "landmarks": landmarks}]] * 4)
    frame_bytes = create_synthetic_frame_bytes(width=640, height=480)
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    api_res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

    assert api_res.status_code == 200
    data = api_res.json()
    assert data["step_passed"] is False
    assert data["reason_code"] in ["FACE_PARTIALLY_VISIBLE", "FACE_OUT_OF_GUIDE"]


def test_face_framing_validator_face_shifted_too_far_up_or_sideways_rejected():
    """4. Face shifted too far up or sideways is rejected."""
    # A. Too far up: center_y = (15 + 40) / 480 = 0.114 < 0.20
    box_up = (220, 15, 200, 80, 0.99)
    landmarks_up = [(270, 40), (370, 40), (320, 60), (280, 80), (360, 80)]
    res_up = FaceFramingValidator.validate_frame(box_up, landmarks_up, img_w=640, img_h=480)
    assert res_up.is_valid is False
    assert res_up.reason_code in ["FACE_OUT_OF_GUIDE", "FACE_PARTIALLY_VISIBLE"]

    # B. Too far left: center_x = (20 + 75) / 640 = 0.148 < 0.22
    box_left = (20, 120, 150, 200, 0.99)
    landmarks_left = [(50, 180), (120, 180), (80, 220), (60, 260), (110, 260)]
    res_left = FaceFramingValidator.validate_frame(box_left, landmarks_left, img_w=640, img_h=480)
    assert res_left.is_valid is False
    assert res_left.reason_code in ["FACE_OUT_OF_GUIDE", "FACE_PARTIALLY_VISIBLE"]

    # C. Too far right: center_x = (470 + 75) / 640 = 0.851 > 0.78
    box_right = (470, 120, 150, 200, 0.99)
    landmarks_right = [(500, 180), (570, 180), (540, 220), (510, 260), (560, 260)]
    res_right = FaceFramingValidator.validate_frame(box_right, landmarks_right, img_w=640, img_h=480)
    assert res_right.is_valid is False
    assert res_right.reason_code in ["FACE_OUT_OF_GUIDE", "FACE_PARTIALLY_VISIBLE"]


def test_face_framing_validator_face_clipped_by_boundaries_rejected():
    """5. Face partially clipped at any frame boundary is rejected with FACE_PARTIALLY_VISIBLE."""
    # Top border clipping (box starts at y=5 < 0.03 * 480)
    box_top_clipped = (220, 5, 200, 220, 0.99)
    landmarks = [(270, 80), (370, 80), (320, 130), (280, 170), (360, 170)]
    res_top = FaceFramingValidator.validate_frame(box_top_clipped, landmarks, img_w=640, img_h=480)
    assert res_top.is_valid is False
    assert res_top.reason_code == "FACE_PARTIALLY_VISIBLE"

    # Bottom border clipping (box bottom reaches 475 > 480 * 0.97 = 465.6)
    box_bottom_clipped = (220, 260, 200, 215, 0.99)
    res_bottom = FaceFramingValidator.validate_frame(box_bottom_clipped, landmarks, img_w=640, img_h=480)
    assert res_bottom.is_valid is False
    assert res_bottom.reason_code == "FACE_PARTIALLY_VISIBLE"

    # Left border clipping (box left = 5 < 640 * 0.03 = 19.2)
    box_left_clipped = (5, 120, 200, 220, 0.99)
    res_left = FaceFramingValidator.validate_frame(box_left_clipped, landmarks, img_w=640, img_h=480)
    assert res_left.is_valid is False
    assert res_left.reason_code == "FACE_PARTIALLY_VISIBLE"

    # Right border clipping (box right = 440 + 195 = 635 > 640 * 0.97 = 620.8)
    box_right_clipped = (440, 120, 195, 220, 0.99)
    res_right = FaceFramingValidator.validate_frame(box_right_clipped, landmarks, img_w=640, img_h=480)
    assert res_right.is_valid is False
    assert res_right.reason_code == "FACE_PARTIALLY_VISIBLE"


def test_liveness_step_face_positioned_correctly_turning_left_passes(client, mock_liveness):
    """6. Face positioned correctly while turning left passes when challenge requires TURN_HEAD_LEFT."""
    service, detector = mock_liveness

    session_id = "live_test_turn_left"
    service._sessions[session_id] = LivenessSessionState(
        session_id=session_id,
        challenges=[ChallengeAction.TURN_HEAD_LEFT],
    )

    # 4 frames: Centered face, yaw turning left (nose moves towards left eye)
    # Right eye x=260, Left eye x=360 (inter-eye=100)
    # Frame 0: nose x=310 (yaw 0.50)
    # Frame 1: nose x=325 (yaw 0.65)
    # Frame 2: nose x=335 (yaw 0.75)
    # Frame 3: nose x=340 (yaw 0.80)
    box = (200, 100, 240, 260, 0.99)
    frames_data = [
        [{"box": box, "landmarks": [(260, 180), (360, 180), (310, 230), (270, 300), (350, 300)]}],
        [{"box": box, "landmarks": [(260, 180), (360, 180), (325, 230), (270, 300), (350, 300)]}],
        [{"box": box, "landmarks": [(260, 180), (360, 180), (335, 230), (270, 300), (350, 300)]}],
        [{"box": box, "landmarks": [(260, 180), (360, 180), (340, 230), (270, 300), (350, 300)]}],
    ]
    detector.set_preset_results(frames_data)

    frame_bytes = create_synthetic_frame_bytes(width=640, height=480)
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["step_passed"] is True
    assert data["all_passed"] is True
    assert data["status"] == "PASSED"


def test_liveness_step_face_positioned_correctly_turning_right_passes(client, mock_liveness):
    """7. Face positioned correctly while turning right passes when challenge requires TURN_HEAD_RIGHT."""
    service, detector = mock_liveness

    session_id = "live_test_turn_right"
    service._sessions[session_id] = LivenessSessionState(
        session_id=session_id,
        challenges=[ChallengeAction.TURN_HEAD_RIGHT],
    )

    # 4 frames: Centered face, yaw turning right (nose moves towards right eye)
    # Right eye x=260, Left eye x=360 (inter-eye=100)
    # Frame 0: nose x=310 (yaw 0.50)
    # Frame 1: nose x=295 (yaw 0.35)
    # Frame 2: nose x=285 (yaw 0.25)
    # Frame 3: nose x=280 (yaw 0.20)
    box = (200, 100, 240, 260, 0.99)
    frames_data = [
        [{"box": box, "landmarks": [(260, 180), (360, 180), (310, 230), (270, 300), (350, 300)]}],
        [{"box": box, "landmarks": [(260, 180), (360, 180), (295, 230), (270, 300), (350, 300)]}],
        [{"box": box, "landmarks": [(260, 180), (360, 180), (285, 230), (270, 300), (350, 300)]}],
        [{"box": box, "landmarks": [(260, 180), (360, 180), (280, 230), (270, 300), (350, 300)]}],
    ]
    detector.set_preset_results(frames_data)

    frame_bytes = create_synthetic_frame_bytes(width=640, height=480)
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["step_passed"] is True
    assert data["all_passed"] is True


def test_liveness_natural_head_turn_small_motion_does_not_false_reject(client, mock_liveness):
    """8. Face moves naturally (slight translation along with rotation) without false framing rejection."""
    service, detector = mock_liveness

    session_id = "live_test_natural_shift"
    service._sessions[session_id] = LivenessSessionState(
        session_id=session_id,
        challenges=[ChallengeAction.TURN_HEAD_LEFT],
    )

    # Face naturally translates by 15-20px as head turns
    frames_data = [
        [{"box": (200, 100, 240, 260, 0.99), "landmarks": [(260, 180), (360, 180), (310, 230), (270, 300), (350, 300)]}],
        [{"box": (208, 103, 240, 260, 0.99), "landmarks": [(268, 183), (368, 183), (328, 233), (278, 303), (358, 303)]}],
        [{"box": (215, 105, 240, 260, 0.99), "landmarks": [(275, 185), (375, 185), (342, 235), (285, 305), (365, 305)]}],
        [{"box": (220, 108, 240, 260, 0.99), "landmarks": [(280, 188), (380, 188), (355, 238), (290, 308), (370, 308)]}],
    ]
    detector.set_preset_results(frames_data)

    frame_bytes = create_synthetic_frame_bytes(width=640, height=480)
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["step_passed"] is True


def test_liveness_framing_enforced_on_every_frame(client, mock_liveness):
    """Sequence fails if any frame violates framing rules (even if first 2 frames were valid)."""
    service, detector = mock_liveness

    session_id = "live_test_frame_fail_middle"
    service._sessions[session_id] = LivenessSessionState(
        session_id=session_id,
        challenges=[ChallengeAction.TURN_HEAD_LEFT],
    )

    valid_box = (200, 100, 240, 260, 0.99)
    valid_landmarks = [(260, 180), (360, 180), (310, 230), (270, 300), (350, 300)]

    # Frame 0 and 1 are valid, but Frame 2 drops down so only eyes appear
    invalid_box = (220, 340, 200, 130, 0.99)
    invalid_landmarks = [(270, 370), (370, 370), (320, 410), (280, 440), (360, 440)]

    detector.set_preset_results([
        [{"box": valid_box, "landmarks": valid_landmarks}],
        [{"box": valid_box, "landmarks": valid_landmarks}],
        [{"box": invalid_box, "landmarks": invalid_landmarks}],  # VIOLATION on frame 3
        [{"box": valid_box, "landmarks": valid_landmarks}],
    ])

    frame_bytes = create_synthetic_frame_bytes(width=640, height=480)
    files = [("frames", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(4)]
    res = client.post(f"/api/v1/liveness/session/{session_id}/verify-step", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["step_passed"] is False
    assert data["reason_code"] in ["FACE_PARTIALLY_VISIBLE", "FACE_OUT_OF_GUIDE"]
    assert "Khung hình thứ 3" in data["message"]


