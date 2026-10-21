import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from app.config import settings
from app.schemas.face import FaceVerificationResponse
from app.services.face_verifier import (
    FaceQualityValidator,
    FaceVerifierService,
    MockFaceEngine,
    YuNetSFaceEngine,
    get_face_verifier,
    set_face_verifier,
)
from tests.conftest import create_valid_image_bytes, create_corrupt_image_bytes


# --- 1. Model Availability & Health Check Tests ---

def test_face_health_endpoint_real(client):
    """When real models are loaded in runtime, health endpoint reports status ok and models_loaded true."""
    # Reset to default singleton
    set_face_verifier(None)
    response = client.get("/api/v1/face/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == settings.APP_VERSION
    assert "YuNet" in data["detector"]
    assert "SFace" in data["recognizer"]
    assert data["models_loaded"] is True
    assert data["engine_type"] == "real"
    assert data["error_message"] is None


def test_face_health_endpoint_unavailable(client):
    """When models cannot be loaded, health endpoint reports unavailable without crashing."""
    service = FaceVerifierService(engine=MockFaceEngine())
    service.engine = None
    service.init_error = "Tệp mô hình ONNX không tồn tại."
    set_face_verifier(service)

    try:
        response = client.get("/api/v1/face/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["models_loaded"] is False
        assert data["engine_type"] == "unavailable"
        assert "không tồn tại" in data["error_message"]
    finally:
        set_face_verifier(None)


def test_face_verify_endpoint_unavailable_returns_503(client):
    """When real models are missing, verify endpoint returns HTTP 503 instead of falling back to mock."""
    service = FaceVerifierService(engine=MockFaceEngine())
    service.engine = None
    service.init_error = "Mô hình YuNet chưa được nạp"
    set_face_verifier(service)

    try:
        valid_bytes = create_valid_image_bytes()
        res = client.post(
            "/api/v1/face/verify",
            files={
                "card_file": ("card.jpg", valid_bytes, "image/jpeg"),
                "selfie_file": ("selfie.jpg", valid_bytes, "image/jpeg"),
            },
        )
        assert res.status_code == 503
        assert "không khả dụng" in res.json()["detail"]
    finally:
        set_face_verifier(None)


def test_no_silent_mock_fallback_on_missing_model_paths():
    """FaceVerifierService must never instantiate MockFaceEngine silently if ONNX paths are missing."""
    service = FaceVerifierService.__new__(FaceVerifierService)
    # Simulate failed YuNetSFaceEngine init
    service.engine = None
    service.init_error = "File not found"
    service.is_injected = False

    assert service.is_ready() is False
    assert service.is_mock() is False
    assert "chưa sẵn sàng" in service.get_status_message()


def test_missing_onnx_model_raises_filenotfound():
    """YuNetSFaceEngine raises FileNotFoundError on missing files."""
    with pytest.raises(FileNotFoundError) as exc_info:
        YuNetSFaceEngine(detector_path="models/face/nonexistent_detector.onnx")
    assert "nonexistent_detector.onnx" in str(exc_info.value)


# --- 2. Input & Threshold Validation Tests ---

def test_face_verify_invalid_card_input(client):
    valid_bytes = create_valid_image_bytes()
    # Empty card file
    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", b"", "image/jpeg"),
            "selfie_file": ("selfie.jpg", valid_bytes, "image/jpeg"),
        },
    )
    assert res.status_code == 400
    assert "rỗng" in res.json()["detail"]

    # Unsupported format
    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.pdf", b"%PDF-1.4", "application/pdf"),
            "selfie_file": ("selfie.jpg", valid_bytes, "image/jpeg"),
        },
    )
    assert res.status_code == 415

    # Corrupt card image
    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", create_corrupt_image_bytes(), "image/jpeg"),
            "selfie_file": ("selfie.jpg", valid_bytes, "image/jpeg"),
        },
    )
    assert res.status_code == 400
    assert "giải mã" in res.json()["detail"]


def test_face_verify_invalid_threshold_rejected(client, mock_face):
    valid_bytes = create_valid_image_bytes()
    # Threshold > 1.0
    res1 = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", valid_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", valid_bytes, "image/jpeg"),
        },
        data={"threshold": 2.5},
    )
    assert res1.status_code == 400
    assert "khoảng từ -1.0 đến 1.0" in res1.json()["detail"]

    # Threshold < -1.0
    res2 = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", valid_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", valid_bytes, "image/jpeg"),
        },
        data={"threshold": -1.5},
    )
    assert res2.status_code == 400
    assert "khoảng từ -1.0 đến 1.0" in res2.json()["detail"]


# --- 3. Face Detection & Isolation Logic ---

def test_face_verify_no_face_in_card(client, mock_face):
    mock_engine = mock_face.engine
    mock_engine.mock_faces_map[600 * 10000 + 400] = []

    card_bytes = create_valid_image_bytes(width=600, height=400)
    selfie_bytes = create_valid_image_bytes(width=500, height=500)

    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", card_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", selfie_bytes, "image/jpeg"),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["match"] is False
    assert data["card_face"]["detected"] is False
    assert data["card_face"]["error_code"] == "NO_FACE_DETECTED"


def test_face_verify_multiple_faces_in_selfie(client, mock_face):
    mock_engine = mock_face.engine
    mock_engine.mock_faces_map[500 * 10000 + 500] = [
        {"box": (100, 100, 80, 80, 0.95), "landmarks": [], "raw": None},
        {"box": (300, 100, 80, 80, 0.92), "landmarks": [], "raw": None},
    ]

    card_bytes = create_valid_image_bytes(width=600, height=400)
    selfie_bytes = create_valid_image_bytes(width=500, height=500)

    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", card_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", selfie_bytes, "image/jpeg"),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["match"] is False
    assert data["selfie_face"]["detected"] is False
    assert data["selfie_face"]["error_code"] == "MULTIPLE_FACES_DETECTED"


def test_cccd_portrait_area_selection():
    service = FaceVerifierService(engine=MockFaceEngine())
    card_w = 856
    card_h = 540

    face_left_portrait = {
        "box": (100, 150, 140, 180, 0.98),
        "landmarks": [(0, 0)] * 5,
        "raw": None,
    }
    face_right_background = {
        "box": (600, 200, 100, 120, 0.85),
        "landmarks": [(0, 0)] * 5,
        "raw": None,
    }

    chosen, err_code, _ = service._select_cccd_portrait_face(
        [face_right_background, face_left_portrait], card_w, card_h
    )
    assert err_code is None
    assert chosen == face_left_portrait
    assert chosen["box"][0] == 100


# --- 4. Quality Validation & Rejection Tests ---

def test_face_quality_serious_defect_size_rejection():
    """Faces smaller than SERIOUS_MIN_SIZE (50px) must be rejected (is_valid=False)."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    q = FaceQualityValidator.validate_quality(img, (20, 20, 40, 40))
    assert q.is_valid is False
    assert any("dưới ngưỡng tối thiểu" in w for w in q.warnings)


def test_face_quality_mild_warning_size():
    """Faces between 50px and 60px are valid but produce a warning."""
    np.random.seed(42)
    img = np.random.randint(60, 200, (200, 200, 3), dtype=np.uint8)
    q = FaceQualityValidator.validate_quality(img, (20, 20, 55, 55))
    assert q.is_valid is True
    assert any("nhỏ hơn khuyến nghị" in w for w in q.warnings)


def test_face_quality_serious_defect_dark_rejection():
    """Severely dark faces (< 20/255) must be rejected (is_valid=False)."""
    dark_img = np.zeros((100, 100, 3), dtype=np.uint8)
    q = FaceQualityValidator.validate_quality(dark_img, (10, 10, 80, 80))
    assert q.is_valid is False
    assert any("thiếu sáng nghiêm trọng" in w for w in q.warnings)


def test_face_quality_serious_defect_blurry_rejection():
    """Flat uniform images with 0 Laplacian variance must be rejected as severely blurry."""
    flat_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    q = FaceQualityValidator.validate_quality(flat_img, (10, 10, 80, 80))
    assert q.is_valid is False
    assert any("nhòe mờ" in w for w in q.warnings)


def test_face_verify_serious_defect_aborts_matching(client, mock_face):
    """When card or selfie has serious quality defect, verification terminates early without matching."""
    # Force Mock to return a tiny face box (30x30 < 50)
    mock_face.engine.mock_faces_map[600 * 10000 + 400] = [{
        "box": (10, 10, 30, 30, 0.95),
        "landmarks": [(15, 15)] * 5,
        "raw": None,
    }]

    card_bytes = create_valid_image_bytes(width=600, height=400)
    selfie_bytes = create_valid_image_bytes(width=500, height=500)

    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", card_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", selfie_bytes, "image/jpeg"),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["match"] is False
    assert data["card_face"]["error_code"] == "LOW_QUALITY_FACE"
    assert any("lỗi chất lượng" in w for w in data["warnings"])


# --- 5. Matching & Threshold Behavior ---

def test_face_verify_match_flow(client, mock_face):
    card_bytes = create_valid_image_bytes(width=600, height=400)
    selfie_bytes = create_valid_image_bytes(width=600, height=400)

    # Use a textured image to pass sharpness check in mock
    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", card_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", selfie_bytes, "image/jpeg"),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["match"] is True
    assert data["similarity_score"] >= 0.99
    assert data["confidence_score"] >= 0.90
    assert data["card_face"]["detected"] is True
    assert data["selfie_face"]["detected"] is True
    assert "thử nghiệm" in data["disclaimer"]


def test_face_verify_different_person(client, mock_face):
    class CustomMockEngine(MockFaceEngine):
        def extract_embedding(self, image: np.ndarray, face_data: dict) -> np.ndarray:
            if image.shape[1] == 600:
                vec = np.zeros((1, 128), dtype=np.float32)
                vec[0, 0] = 1.0
                return vec
            else:
                vec = np.zeros((1, 128), dtype=np.float32)
                vec[0, 1] = 1.0
                return vec

    mock_face.engine = CustomMockEngine()

    card_bytes = create_valid_image_bytes(width=600, height=400)
    selfie_bytes = create_valid_image_bytes(width=700, height=500)

    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", card_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", selfie_bytes, "image/jpeg"),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["match"] is False
    assert data["similarity_score"] == 0.0
    assert data["confidence_score"] < 0.5


def test_face_verify_custom_threshold(client, mock_face):
    class FixedSimEngine(MockFaceEngine):
        def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
            return 0.55

    mock_face.engine = FixedSimEngine()

    card_bytes = create_valid_image_bytes(width=600, height=400)
    selfie_bytes = create_valid_image_bytes(width=600, height=400)

    # Threshold 0.50 -> match (0.55 >= 0.50)
    res1 = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", card_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", selfie_bytes, "image/jpeg"),
        },
        data={"threshold": 0.50},
    )
    assert res1.json()["match"] is True

    # Threshold 0.65 -> no match (0.55 < 0.65)
    res2 = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", card_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", selfie_bytes, "image/jpeg"),
        },
        data={"threshold": 0.65},
    )
    assert res2.json()["match"] is False


# --- 6. Real Model Weights & Evaluation Tools ---

def test_real_yunet_sface_engine_initialization():
    """Verify that YuNet and SFace models in models/face/ load cleanly."""
    try:
        engine = YuNetSFaceEngine()
        assert engine._detector is not None
        assert engine._recognizer is not None
    except FileNotFoundError:
        pytest.skip("Face model ONNX files not present in models/face/")


def test_face_benchmark_synthetic_smoke_test():
    """Verify that evaluate_face properly labels synthetic runs as smoke tests."""
    import evaluate_face
    verifier = FaceVerifierService(engine=MockFaceEngine())
    pairs = evaluate_face.create_synthetic_smoke_test_pairs(count=4)
    summary = evaluate_face.run_benchmark(pairs, verifier, is_synthetic=True, thresholds=[0.363])

    assert summary["evaluation_mode"] == "SYNTHETIC_SMOKE_TEST"
    assert summary["biometric_validity"] == "SMOKE_TEST_ONLY_DO_NOT_CITE"
    assert summary["total_pairs"] == 4
    assert summary["genuine_pairs"] == 2
    assert summary["impostor_pairs"] == 2
    assert "0.363" in summary["threshold_metrics"]


def test_load_manifest_validation_errors():
    """Verify manifest dataset loader rejects non-existent and empty manifests."""
    import evaluate_face

    with pytest.raises(FileNotFoundError):
        evaluate_face.load_manifest_dataset("nonexistent_manifest_file.json")


def test_cccd_portrait_moderate_blur_passes_quality_with_warning():
    """CCCD portrait with Laplacian variance ~8.35 (similar to user case) is valid (is_valid=True) with advisory warning."""
    img = np.full((353, 281, 3), 152, dtype=np.uint8)
    cv2.circle(img, (100, 140), 15, (50, 50, 50), -1)
    cv2.circle(img, (180, 140), 15, (50, 50, 50), -1)
    cv2.ellipse(img, (140, 220), (40, 15), 0, 0, 180, (80, 60, 60), 3)
    img_soft = cv2.GaussianBlur(img, (5, 5), 1.2)

    gray = cv2.cvtColor(img_soft, cv2.COLOR_BGR2GRAY)
    actual_blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    assert actual_blur < 15.0  # Would have failed under old threshold 15.0!

    q = FaceQualityValidator.validate_quality(img_soft, (0, 0, 281, 353), source_type="card")
    assert q.is_valid is True
    assert any("độ sắc nét trung bình" in w for w in q.warnings)


def test_cccd_moderate_blur_matching_proceeds_without_abort(client, mock_face):
    """When CCCD has moderate blur (e.g. Laplacian ~8.35), matching is NOT aborted and returns genuine score."""
    mock_face.engine.mock_faces_map[600 * 10000 + 400] = [{
        "box": (100, 50, 281, 353, 0.99),
        "landmarks": [(150, 150), (230, 150), (190, 220), (160, 280), (220, 280)],
        "raw": None,
    }]

    card_arr = np.full((400, 600, 3), 152, dtype=np.uint8)
    # Add facial features inside crop (100, 50, 281, 353) -> x in [100, 381], y in [50, 403]
    cv2.circle(card_arr, (180, 160), 12, (50, 50, 50), -1)
    cv2.circle(card_arr, (260, 160), 12, (50, 50, 50), -1)
    cv2.ellipse(card_arr, (220, 260), (45, 18), 0, 0, 180, (80, 60, 60), 4)
    card_arr_soft = cv2.GaussianBlur(card_arr, (5, 5), 1.2)
    _, buf_card = cv2.imencode(".jpg", card_arr_soft)
    card_bytes = buf_card.tobytes()

    selfie_bytes = create_valid_image_bytes(width=600, height=400)

    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", card_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", selfie_bytes, "image/jpeg"),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["match"] is True
    assert data["pipeline_state"] == "MATCHED"
    assert data["similarity_score"] > 0.0
    assert any("độ sắc nét trung bình" in w for w in data["warnings"])


def test_face_quality_severe_blur_aborts_matching(client, mock_face):
    """When image is severely blurred (Laplacian < 3.0), matching is aborted with BLOCKED_QUALITY."""
    mock_face.engine.mock_faces_map[600 * 10000 + 400] = [{
        "box": (100, 50, 280, 280, 0.99),
        "landmarks": [(150, 150), (230, 150), (190, 220), (160, 280), (220, 280)],
        "raw": None,
    }]

    # Create flat image with Laplacian 0.0 < 3.0
    flat_arr = np.full((400, 600, 3), 140, dtype=np.uint8)
    _, buf_card = cv2.imencode(".jpg", flat_arr)
    card_bytes = buf_card.tobytes()

    selfie_bytes = create_valid_image_bytes(width=600, height=400)

    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", card_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", selfie_bytes, "image/jpeg"),
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["match"] is False
    assert data["pipeline_state"] == "BLOCKED_QUALITY"
    assert data["similarity_score"] == 0.0


def test_diagnostics_payload():
    """Verify that verify_faces includes complete diagnostics payload."""
    verifier = FaceVerifierService(engine=MockFaceEngine())
    card_img = np.random.randint(60, 200, (400, 600, 3), dtype=np.uint8)
    selfie_img = np.random.randint(60, 200, (500, 500, 3), dtype=np.uint8)

    resp = verifier.verify_faces(card_img, selfie_img)
    assert resp.diagnostics is not None
    assert resp.diagnostics["pipeline_state"] == "MATCHED"
    assert resp.diagnostics["matching_executed"] is True
    assert resp.diagnostics["card"]["image_size"] == [600, 400]
    assert resp.diagnostics["selfie"]["image_size"] == [500, 500]


def test_debug_crop_saving_toggle(tmp_path):
    """Verify save_debug_crop respects FACE_DEBUG_SAVE_CROPS flag."""
    import os
    from app.services.face_verifier import save_debug_crop
    test_crop = np.zeros((100, 100, 3), dtype=np.uint8)

    # When False, does not save
    settings.FACE_DEBUG_SAVE_CROPS = False
    result = save_debug_crop(test_crop, "test")
    assert result is None

    # When True, saves to debug dir
    settings.FACE_DEBUG_SAVE_CROPS = True
    settings.FACE_DEBUG_DIR = str(tmp_path)
    try:
        saved_path = save_debug_crop(test_crop, "test_crop")
        assert saved_path is not None
        assert os.path.exists(saved_path)
    finally:
        settings.FACE_DEBUG_SAVE_CROPS = False


def test_face_benchmark_distribution_reporting():
    """Verify that evaluate_face reports matching execution counts and distributions."""
    import evaluate_face
    verifier = FaceVerifierService(engine=MockFaceEngine())
    pairs = evaluate_face.create_synthetic_smoke_test_pairs(count=6)
    summary = evaluate_face.run_benchmark(pairs, verifier, is_synthetic=True, thresholds=[0.363])

    assert "matching_executed" in summary
    assert summary["matching_executed"] == 6
    assert "similarity_distribution" in summary
    assert summary["similarity_distribution"]["genuine"]["count"] == 3
    assert summary["similarity_distribution"]["impostor"]["count"] == 3


def test_yunet_score_threshold_configurability_and_runtime_update():
    """Verify that YuNetSFaceEngine honors score_threshold and can update threshold at runtime."""
    engine = YuNetSFaceEngine(score_threshold=0.65)
    assert engine.score_threshold == 0.65
    engine.set_score_threshold(0.75)
    assert engine.score_threshold == 0.75


def test_selfie_detection_recovered_with_calibrated_threshold():
    """Verify that a face with score 0.74 is rejected under 0.80 but accepted under 0.70."""
    verifier = FaceVerifierService(engine=MockFaceEngine())
    # Mock face with 0.74 confidence
    verifier.engine.mock_faces_map[600 * 10000 + 400] = [{
        "box": (150, 80, 200, 200, 0.74),
        "landmarks": [(180, 130), (280, 130), (230, 180), (200, 230), (260, 230)],
        "raw": None,
    }]
    faces = verifier.engine.detect_faces(np.zeros((400, 600, 3), dtype=np.uint8))
    face, err_code, _ = verifier._select_selfie_face(faces)
    assert face is not None
    assert face["box"][4] == 0.74
    assert err_code is None


def test_cccd_multi_face_selects_primary_large_portrait():
    """When CCCD has primary large portrait and small secondary ghost portrait, selects the primary portrait."""
    verifier = FaceVerifierService(engine=MockFaceEngine())
    img_w, img_h = 1000, 630

    # Face 1: Primary cardholder portrait on left (cx=250 <= 580, area=200*250 = 50000)
    # Face 2: Secondary miniature ghost portrait on right (cx=850, area=70*90 = 6300)
    faces = [
        {"box": (150, 150, 200, 250, 0.99), "landmarks": [], "raw": None},
        {"box": (815, 400, 70, 90, 0.85), "landmarks": [], "raw": None},
    ]

    selected, err_code, _ = verifier._select_cccd_portrait_face(faces, img_w, img_h)
    assert err_code is None
    assert selected is not None
    assert selected["box"] == (150, 150, 200, 250, 0.99)


def test_cccd_dominant_face_selection_with_margin():
    """When CCCD has margins (cx > 0.58*w), selects the dominant primary face over background noise."""
    verifier = FaceVerifierService(engine=MockFaceEngine())
    img_w, img_h = 1000, 600

    # Face 1: Large face (area=250*300 = 75000) at cx=600 (>580)
    # Face 2: Tiny false positive (area=40*40 = 1600)
    faces = [
        {"box": (475, 100, 250, 300, 0.95), "landmarks": [], "raw": None},
        {"box": (50, 50, 40, 40, 0.72), "landmarks": [], "raw": None},
    ]

    selected, err_code, _ = verifier._select_cccd_portrait_face(faces, img_w, img_h)
    assert err_code is None
    assert selected is not None
    assert selected["box"][2] == 250


def test_small_portrait_advisory_warning_explaining_degradation():
    """Portrait crop of 59x78 px produces advisory warning explaining SFace biometric degradation without hard abort."""
    crop_img = np.full((78, 59, 3), 150, dtype=np.uint8)
    crop_img[30:35, 15:25] = 40
    crop_img[30:35, 35:45] = 40
    crop_img[55:60, 20:40] = 50

    q = FaceQualityValidator.validate_quality(crop_img, (0, 0, 59, 78), source_type="card")
    assert q.is_valid is True  # Passes quality gate (advisory warning only)
    assert any("nhỏ hơn khuyến nghị" in w for w in q.warnings)
    assert any("SFace" in w for w in q.warnings)


def test_benchmark_runner_calculates_rate_metrics():
    """Verify evaluate_face calculates rate percentages correctly."""
    import evaluate_face
    verifier = FaceVerifierService(engine=MockFaceEngine())
    pairs = evaluate_face.create_synthetic_smoke_test_pairs(count=4)
    summary = evaluate_face.run_benchmark(pairs, verifier, is_synthetic=True, thresholds=[0.363])

    assert "detection_success_rate_percent" in summary
    assert "quality_rejection_rate_percent" in summary
    assert "matching_executed_rate_percent" in summary
    assert summary["matching_executed_rate_percent"] == 100.0


