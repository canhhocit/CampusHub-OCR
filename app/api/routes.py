import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.config import settings
from app.schemas.ocr import (
    CardSide,
    OCRResponse,
    HealthResponse,
    FieldStatus,
)
from app.services.image_processor import ImageProcessor, ImageValidationError
from app.services.ocr_engine import get_ocr_engine
from app.services.cccd_extractor import CCCDExtractor

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        ocr_engine="PaddleOCR",
        device="GPU" if settings.USE_GPU else "CPU",
        supported_formats=list(settings.ALLOWED_EXTENSIONS),
    )

@router.post("/ocr", response_model=OCRResponse)
async def process_ocr(
    file: UploadFile = File(..., description="Tệp ảnh CCCD định dạng JPG, JPEG hoặc PNG"),
    side: CardSide = Form(default=CardSide.AUTO, description="Mặt thẻ (FRONT, BACK hoặc AUTO)"),
    preprocess: bool = Form(default=True, description="Kích hoạt tiền xử lý hình ảnh tối ưu OCR"),
):
    start_time = time.perf_counter()
    warnings: list[str] = []

    # 1. Read file bytes safely into memory
    try:
        content = await file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể đọc nội dung tệp tải lên."
        )

    # 2. Validate image structure, format, dimensions, and file size
    try:
        image, metadata = ImageProcessor.validate_image_bytes(
            data=content,
            filename=file.filename or "upload.jpg"
        )
    except ImageValidationError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message
        )
    finally:
        # Avoid retaining raw upload content in memory
        del content

    # 3. Optional image preprocessing
    try:
        processed_img = ImageProcessor.preprocess_for_ocr(image) if preprocess else image
    except Exception:
        processed_img = image
        warnings.append("Tiền xử lý ảnh thất bại, hệ thống sử dụng ảnh gốc để nhận dạng.")

    # 4. OCR Inference
    try:
        engine = get_ocr_engine()
        detections = engine.detect_and_recognize(processed_img)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Quá trình nhận dạng ký tự (OCR) gặp sự cố nội bộ. Vui lòng thử lại với ảnh chất lượng hơn."
        )

    # 5. Field Extraction
    effective_side, fields = CCCDExtractor.extract_fields(detections, requested_side=side)

    # 6. Check warnings
    if not detections:
        warnings.append("Không nhận diện được bất kỳ ký tự văn bản nào trong ảnh. Hãy kiểm tra độ sáng và góc chụp.")
    else:
        # Check if ID number was found on front side
        if effective_side == CardSide.FRONT:
            id_f = fields.get("id_number")
            if not id_f or id_f.status != FieldStatus.VALID:
                warnings.append("Không phát hiện được số CCCD (12 chữ số). Ảnh có thể bị mờ, chói lóa hoặc bị che khuất.")
        else:
            issue_d = fields.get("issue_date")
            if not issue_d or issue_d.status != FieldStatus.VALID:
                warnings.append("Không phát hiện được ngày cấp trên mặt sau của thẻ.")

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return OCRResponse(
        success=True,
        side=effective_side,
        processing_time_ms=elapsed_ms,
        fields=fields,
        raw_detections=detections,
        image_info=metadata,
        warnings=warnings,
        disclaimer=settings.DISCLAIMER,
    )
