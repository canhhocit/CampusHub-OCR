from typing import Optional, Tuple
from pydantic import BaseModel, Field

class FaceBoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    confidence: float = Field(..., ge=0.0, le=1.0)

class FaceLandmarks(BaseModel):
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    nose_tip: Tuple[int, int]
    mouth_left: Tuple[int, int]
    mouth_right: Tuple[int, int]

class FaceQualityCheck(BaseModel):
    is_valid: bool
    width: int
    height: int
    brightness: float
    blur_score: float
    aspect_ratio: float
    warnings: list[str] = Field(default_factory=list)

class FaceDetectionResult(BaseModel):
    detected: bool
    count: int = 0
    box: Optional[FaceBoundingBox] = None
    landmarks: Optional[FaceLandmarks] = None
    quality: Optional[FaceQualityCheck] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

class FaceVerificationResponse(BaseModel):
    success: bool
    match: bool
    similarity_score: float = Field(
        ...,
        description="Cosine similarity giữa 2 vector đặc trưng SFace trong khoảng [-1.0, 1.0]."
    )
    confidence_score: float = Field(
        ...,
        description="Điểm hiển thị trực quan (heuristic display score) được chuẩn hóa từ cosine similarity quanh ngưỡng quyết định; không phải xác suất thống kê đã chuẩn hóa (calibrated probability)."
    )
    threshold: float
    card_face: FaceDetectionResult
    selfie_face: FaceDetectionResult
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: float
    disclaimer: str

class FaceHealthResponse(BaseModel):
    status: str
    version: str
    detector: str
    recognizer: str
    device: str
    models_loaded: bool
    threshold: float
    engine_type: str = "real"
    error_message: Optional[str] = None

