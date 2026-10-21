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
    
    # Disclaimer
    DISCLAIMER: str = (
        "Kết quả OCR mang tính chất thử nghiệm và trích xuất dữ liệu tự động, "
        "không chứng minh hoặc thay thế việc xác thực tính pháp lý của giấy tờ công dân."
    )

settings = Settings()
