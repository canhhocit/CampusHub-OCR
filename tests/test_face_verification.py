import cv2
import numpy as np
import pytest
from app.config import settings
from app.schemas.face import FaceVerificationResponse
from app.services.face_verifier import (
    FaceQualityValidator,
    FaceVerifierService,
    MockFaceEngine,
    YuNetSFaceEngine,
)
from tests.conftest import create_valid_image_bytes, create_corrupt_image_bytes


# --- 1. API Endpoint Tests ---

def test_face_health_endpoint(client):
    response = client.get("/api/v1/face/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == settings.APP_VERSION
    assert "YuNet" in data["detector"]
    assert "SFace" in data["recognizer"]
    assert data["threshold"] == settings.FACE_MATCH_THRESHOLD


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


def test_face_verify_invalid_selfie_input(client):
    valid_bytes = create_valid_image_bytes()
    # Empty selfie file
    res = client.post(
        "/api/v1/face/verify",
        files={
            "card_file": ("card.jpg", valid_bytes, "image/jpeg"),
            "selfie_file": ("selfie.jpg", b"", "image/jpeg"),
        },
    )
    assert res.status_code == 400
    assert "rỗng" in res.json()["detail"]


# --- 2. Face Detection Edge Cases ---

def test_face_verify_no_face_in_card(client, mock_face):
    mock_engine = mock_face.engine
    # Set mock to return empty list for card (dimensions 600x400)
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
    assert any("thẻ CCCD" in w for w in data["warnings"])


def test_face_verify_no_face_in_selfie(client, mock_face):
    mock_engine = mock_face.engine
    # Set mock to return empty list for selfie (dimensions 500x500)
    mock_engine.mock_faces_map[500 * 10000 + 500] = []

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
    assert data["selfie_face"]["error_code"] == "NO_FACE_DETECTED"
    assert any("selfie" in w for w in data["warnings"])


def test_face_verify_multiple_faces_in_selfie(client, mock_face):
    mock_engine = mock_face.engine
    # Return 2 faces for selfie
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
    assert data["selfie_face"]["count"] == 2
    assert data["selfie_face"]["error_code"] == "MULTIPLE_FACES_DETECTED"


# --- 3. CCCD Portrait Isolation Logic ---

def test_cccd_portrait_area_selection():
    service = FaceVerifierService(engine=MockFaceEngine())
    # Card is 856 x 540 (standard CCCD ratio)
    card_w = 856
    card_h = 540

    face_left_portrait = {
        "box": (100, 150, 140, 180, 0.98),  # cx=170 <= 0.52 * 856 = 445 -> Card portrait
        "landmarks": [(0, 0)] * 5,
        "raw": None,
    }
    face_right_background = {
        "box": (600, 200, 100, 120, 0.85),  # cx=650 > 445 -> Outside portrait zone
        "landmarks": [(0, 0)] * 5,
        "raw": None,
    }

    # When both are detected, it should isolate the portrait on the left
    chosen, err_code, _ = service._select_cccd_portrait_face(
        [face_right_background, face_left_portrait], card_w, card_h
    )
    assert err_code is None
    assert chosen == face_left_portrait
    assert chosen["box"][0] == 100


# --- 4. Quality Validation ---

def test_face_quality_small_face():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    # 40x40 is < settings.FACE_MIN_SIZE (60)
    q = FaceQualityValidator.validate_quality(img, (20, 20, 40, 40))
    assert q.is_valid is True
    assert any("quá nhỏ" in w for w in q.warnings)


def test_face_quality_dark_and_bright():
    # Very dark crop
    dark_img = np.zeros((100, 100, 3), dtype=np.uint8)
    q_dark = FaceQualityValidator.validate_quality(dark_img, (10, 10, 80, 80))
    assert any("quá tối" in w for w in q_dark.warnings)

    # Very bright/overexposed crop
    bright_img = np.full((100, 100, 3), 250, dtype=np.uint8)
    q_bright = FaceQualityValidator.validate_quality(bright_img, (10, 10, 80, 80))
    assert any("lóa sáng" in w for w in q_bright.warnings)


def test_face_quality_blurry():
    # Solid flat gray crop has 0 Laplacian variance (severely blurry / no texture)
    flat_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    q = FaceQualityValidator.validate_quality(flat_img, (10, 10, 80, 80))
    assert any("mờ nét" in w for w in q.warnings)


# --- 5. Matching & Threshold Behavior ---

def test_face_verify_match_flow(client, mock_face):
    # Using default MockFaceEngine, identical image size produces identical embeddings
    card_bytes = create_valid_image_bytes(width=600, height=400)
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
    assert data["similarity_score"] >= 0.99
    assert data["confidence_score"] >= 0.90
    assert data["card_face"]["detected"] is True
    assert data["selfie_face"]["detected"] is True
    assert "thử nghiệm" in data["disclaimer"]


def test_face_verify_different_person(client, mock_face):
    class CustomMockEngine(MockFaceEngine):
        def extract_embedding(self, image: np.ndarray, face_data: dict) -> np.ndarray:
            # Return orthogonal embeddings based on image size
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


# --- 6. Real Model Weights Loading ---

def test_real_yunet_sface_engine_initialization():
    """Verify that YuNet and SFace models in models/face/ load cleanly."""
    try:
        engine = YuNetSFaceEngine()
        assert engine._detector is not None
        assert engine._recognizer is not None
    except FileNotFoundError:
        pytest.skip("Face model ONNX files not present in models/face/")


def test_face_benchmark_evaluation_tool():
    """Verify that evaluate_face.run_benchmark computes metrics properly."""
    import evaluate_face
    verifier = FaceVerifierService(engine=MockFaceEngine())
    pairs = evaluate_face.create_synthetic_benchmark_pairs(count=4)
    summary = evaluate_face.run_benchmark(pairs, verifier, thresholds=[0.363])
    assert summary["total_pairs"] == 4
    assert summary["total_positives"] == 2
    assert summary["total_negatives"] == 2
    assert "0.363" in summary["threshold_metrics"]
    assert "latencies_ms" in summary

