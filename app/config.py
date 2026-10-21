import os
from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "CampusHub-OCR"
    APP_VERSION: str = "1.0.0"
    HOST: str = os.getenv("OCR_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("OCR_PORT", "8001"))
    DEBUG: bool = os.getenv("OCR_DEBUG", "false").lower() == "true"
    
    # Image constraints
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MIN_IMAGE_WIDTH: int = 100
    MIN_IMAGE_HEIGHT: int = 100
    MAX_IMAGE_WIDTH: int = 6000
    MAX_IMAGE_HEIGHT: int = 6000
    ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
    ALLOWED_MIME_TYPES: set[str] = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "application/octet-stream",  # Fallback for some clients, checked by magic bytes
    }
    
    # OCR settings
    OCR_LANG: str = os.getenv("OCR_LANG", "vi")
    USE_GPU: bool = os.getenv("OCR_USE_GPU", "false").lower() == "true"
    ENABLE_PREPROCESSING: bool = True
    
    # Face Verification settings
    FACE_MODEL_DIR: str = os.getenv("FACE_MODEL_DIR", "models/face")
    FACE_DETECTOR_MODEL: str = "face_detection_yunet_2023mar.onnx"
    FACE_RECOGNIZER_MODEL: str = "face_recognition_sface_2021dec.onnx"
    FACE_MATCH_THRESHOLD: float = float(os.getenv("FACE_MATCH_THRESHOLD", "0.363"))
    FACE_SCORE_THRESHOLD: float = 0.80
    FACE_MIN_SIZE: int = 60
    ALLOW_MOCK_FACE: bool = os.getenv("ALLOW_MOCK_FACE", "false").lower() == "true"
    
    # Face Quality Validator thresholds (domain-specific: printed CCCD vs live selfie)
    FACE_SERIOUS_MIN_SIZE: int = 50
    FACE_CARD_SERIOUS_MIN_BLUR: float = 3.0       # Severe blur threshold for printed CCCD card portraits
    FACE_CARD_RECOMMENDED_MIN_BLUR: float = 12.0  # Advisory warning threshold for soft card portraits
    FACE_SELFIE_SERIOUS_MIN_BLUR: float = 5.0     # Severe blur threshold for live selfies
    FACE_SELFIE_RECOMMENDED_MIN_BLUR: float = 25.0 # Advisory warning threshold for live selfies
    FACE_SERIOUS_MIN_BRIGHTNESS: float = 15.0
    FACE_SERIOUS_MAX_BRIGHTNESS: float = 245.0
    FACE_RECOMMENDED_MIN_BRIGHTNESS: float = 35.0
    FACE_RECOMMENDED_MAX_BRIGHTNESS: float = 225.0

    # Diagnostic & Debug options (strictly for local development, disabled by default)
    FACE_DEBUG_SAVE_CROPS: bool = os.getenv("FACE_DEBUG_SAVE_CROPS", "false").lower() == "true"
    FACE_DEBUG_DIR: str = os.getenv("FACE_DEBUG_DIR", "scratch/face_debug")

    FACE_DISCLAIMER: str = (
        "Kết quả so khớp khuôn mặt 1:1 mang tính chất thử nghiệm thuật toán và đối sánh sinh trắc học, "
        "không cấu thành việc định danh pháp lý hoặc kiểm tra chống giả mạo (liveness/anti-spoofing)."
    )

    # Disclaimer
    DISCLAIMER: str = (
        "Kết quả OCR mang tính chất thử nghiệm và trích xuất dữ liệu tự động, "
        "không chứng minh hoặc thay thế việc xác thực tính pháp lý của giấy tờ công dân."
    )

settings = Settings()
