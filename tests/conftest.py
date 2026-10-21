import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.services.ocr_engine import MockOCREngine, set_ocr_engine, BaseOCREngine, get_ocr_engine
from app.services.face_verifier import (
    FaceVerifierService,
    MockFaceEngine,
    get_face_verifier,
    set_face_verifier,
)
from app.schemas.ocr import RawDetection

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_ocr():
    """Provides a controlled mock OCR engine for fast and deterministic unit testing."""
    original_engine = get_ocr_engine()
    mock = MockOCREngine()
    set_ocr_engine(mock)
    yield mock
    set_ocr_engine(original_engine)

@pytest.fixture
def mock_face():
    """Provides a controlled mock Face verifier service for fast and deterministic unit testing."""
    original_verifier = get_face_verifier()
    mock_engine = MockFaceEngine()
    mock_service = FaceVerifierService(engine=mock_engine)
    set_face_verifier(mock_service)
    yield mock_service
    set_face_verifier(original_verifier)

def create_valid_image_bytes(width: int = 600, height: int = 400, format: str = "JPEG") -> bytes:
    """Creates synthetic valid image bytes with realistic texture without saving to disk."""
    np.random.seed(42)
    arr = np.random.randint(60, 200, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def create_corrupt_image_bytes() -> bytes:
    """Returns malformed bytes that simulate a broken or truncated file."""
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00corruptcontenthere"
