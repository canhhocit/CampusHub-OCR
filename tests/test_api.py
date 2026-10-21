import io
from app.schemas.ocr import RawDetection
from tests.conftest import create_valid_image_bytes

def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["ocr_engine"] == "PaddleOCR"

def test_index_ui_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CampusHub-OCR" in response.text
    assert "Bắt đầu Trích xuất OCR" in response.text
    assert "So khớp Khuôn mặt 1:1" in response.text
    assert "faceCardFileInput" in response.text
    assert "selfieFileInput" in response.text
    assert "openWebcamBtn" in response.text
    assert "submitFaceBtn" in response.text

def test_ocr_endpoint_validation_error_on_empty(client):
    response = client.post(
        "/api/v1/ocr",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        data={"side": "AUTO"}
    )
    assert response.status_code == 400
    assert "rỗng" in response.json()["detail"]

def test_ocr_endpoint_validation_error_on_unsupported_type(client):
    response = client.post(
        "/api/v1/ocr",
        files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")},
        data={"side": "AUTO"}
    )
    assert response.status_code == 415

def test_ocr_endpoint_with_mock_engine(client, mock_ocr):
    # Set mock detections
    mock_ocr.set_preset_detections([
        RawDetection(text="CĂN CƯỚC CÔNG DÂN", confidence=0.99),
        RawDetection(text="Số: 012345678912", confidence=0.99),
        RawDetection(text="Họ và tên: LÊ THỊ HOA", confidence=0.98),
        RawDetection(text="Ngày sinh: 10/10/2000", confidence=0.97),
    ])

    img_bytes = create_valid_image_bytes(width=800, height=500)
    response = client.post(
        "/api/v1/ocr",
        files={"file": ("cccd.jpg", img_bytes, "image/jpeg")},
        data={"side": "FRONT", "preprocess": "true"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["side"] == "FRONT"
    assert data["processing_time_ms"] > 0
    assert data["fields"]["id_number"]["value"] == "012345678912"
    assert data["fields"]["full_name"]["value"] == "LÊ THỊ HOA"
    assert len(data["raw_detections"]) == 4
    assert data["disclaimer"] != ""
