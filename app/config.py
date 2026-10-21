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
    FACE_SCORE_THRESHOLD: float = float(os.getenv("FACE_SCORE_THRESHOLD", "0.70"))
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

    # Liveness Detection settings (Active Challenge Prototype V1)
    LIVENESS_SESSION_TIMEOUT_SEC: int = int(os.getenv("LIVENESS_SESSION_TIMEOUT_SEC", "120"))
    LIVENESS_MAX_SESSIONS: int = int(os.getenv("LIVENESS_MAX_SESSIONS", "100"))
    LIVENESS_MIN_FRAMES: int = int(os.getenv("LIVENESS_MIN_FRAMES", "3"))
    LIVENESS_MAX_FRAMES: int = int(os.getenv("LIVENESS_MAX_FRAMES", "15"))
    LIVENESS_MAX_PAYLOAD_BYTES: int = 15 * 1024 * 1024  # 15 MB
    LIVENESS_MAX_STEP_ATTEMPTS: int = 3
    LIVENESS_YAW_CENTER_MIN: float = 0.40
    LIVENESS_YAW_CENTER_MAX: float = 0.60
    LIVENESS_YAW_TURN_LEFT_MIN: float = 0.65
    LIVENESS_YAW_TURN_RIGHT_MAX: float = 0.35
    LIVENESS_MIN_MOTION_VARIANCE: float = 0.04

    # Face-framing validation settings (normalized coordinates in [0.0, 1.0])
    LIVENESS_FACE_MIN_HEIGHT_RATIO: float = float(os.getenv("LIVENESS_FACE_MIN_HEIGHT_RATIO", "0.20"))
    LIVENESS_FACE_MAX_HEIGHT_RATIO: float = float(os.getenv("LIVENESS_FACE_MAX_HEIGHT_RATIO", "0.80"))
    LIVENESS_FACE_MIN_WIDTH_RATIO: float = float(os.getenv("LIVENESS_FACE_MIN_WIDTH_RATIO", "0.15"))
    LIVENESS_FACE_MAX_WIDTH_RATIO: float = float(os.getenv("LIVENESS_FACE_MAX_WIDTH_RATIO", "0.75"))
    LIVENESS_FACE_CENTER_X_MIN: float = float(os.getenv("LIVENESS_FACE_CENTER_X_MIN", "0.22"))
    LIVENESS_FACE_CENTER_X_MAX: float = float(os.getenv("LIVENESS_FACE_CENTER_X_MAX", "0.78"))
    LIVENESS_FACE_CENTER_Y_MIN: float = float(os.getenv("LIVENESS_FACE_CENTER_Y_MIN", "0.20"))
    LIVENESS_FACE_CENTER_Y_MAX: float = float(os.getenv("LIVENESS_FACE_CENTER_Y_MAX", "0.65"))
    LIVENESS_FRAME_MARGIN_RATIO: float = float(os.getenv("LIVENESS_FRAME_MARGIN_RATIO", "0.03"))
    LIVENESS_MAX_EYE_Y_RATIO: float = float(os.getenv("LIVENESS_MAX_EYE_Y_RATIO", "0.58"))
    LIVENESS_MAX_MOUTH_Y_RATIO: float = float(os.getenv("LIVENESS_MAX_MOUTH_Y_RATIO", "0.88"))

    LIVENESS_DISCLAIMER: str = (
        "Kết quả kiểm tra thực thể sống V1 (Liveness Detection) dựa trên thách thức chủ động (Active Challenge). "
        "Đây là phiên bản thử nghiệm thuật toán (research prototype), không thay thế chứng thực sinh trắc học chuẩn quốc tế "
        "hoặc chống lại các cuộc tấn công giả mạo tinh vi (deepfake/replay)."
    )

    # Disclaimer
    DISCLAIMER: str = (
        "Kết quả OCR mang tính chất thử nghiệm và trích xuất dữ liệu tự động, "
        "không chứng minh hoặc thay thế việc xác thực tính pháp lý của giấy tờ công dân."
    )

settings = Settings()
