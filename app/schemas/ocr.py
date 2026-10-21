from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field

class CardSide(str, Enum):
    FRONT = "FRONT"
    BACK = "BACK"
    AUTO = "AUTO"

class FieldStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    MISSING = "MISSING"
    INVALID = "INVALID"
    # Backward-compatible aliases
    VALID = "EXTRACTED"
    NOT_DETECTED = "MISSING"
    UNREADABLE = "INVALID"

class ExtractedField(BaseModel):
    name: str
    label: str
    value: Optional[str] = None
    status: FieldStatus
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    extraction_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    raw_text: Optional[str] = None
    validation_message: Optional[str] = None

class RawDetection(BaseModel):
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: Optional[list[list[float]]] = None

class ImageMetadata(BaseModel):
    width: int
    height: int
    channels: int
    size_bytes: int
    format: str

class OCRResponse(BaseModel):
    success: bool
    side: CardSide
    processing_time_ms: float
    fields: dict[str, ExtractedField]
    raw_detections: list[RawDetection]
    image_info: Optional[ImageMetadata] = None
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str

class HealthResponse(BaseModel):
    status: str
    version: str
    ocr_engine: str
    device: str
    supported_formats: list[str]
