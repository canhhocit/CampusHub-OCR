import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.config import settings
from app.schemas.face import (
    FaceVerificationResponse,
    FaceHealthResponse,
)
from app.services.image_processor import ImageProcessor, ImageValidationError
from app.services.face_verifier import get_face_verifier, YuNetSFaceEngine

face_router = APIRouter()


@face_router.get("/health", response_model=FaceHealthResponse)
def face_health_check():
    verifier = get_face_verifier()
    is_yunet = isinstance(verifier.engine, YuNetSFaceEngine)
    return FaceHealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        detector="YuNet (ONNX - OpenCV Zoo)",
        recognizer="SFace (ONNX - OpenCV Zoo)",
        device="CPU",
        models_loaded=is_yunet,
        threshold=settings.FACE_MATCH_THRESHOLD,
    )


@face_router.post("/verify", response_model=FaceVerificationResponse)
async def verify_faces(
    card_file: UploadFile = File(..., description="Tệp ảnh mặt trước CCCD có chân dung in (JPG, PNG)"),
    selfie_file: UploadFile = File(..., description="Tệp ảnh chân dung chụp từ webcam (JPG, PNG)"),
    threshold: Optional[float] = Form(default=None, description="Ngưỡng so khớp cosine (mặc định 0.363)"),
):
    """Xác thực đối sánh khuôn mặt 1:1 giữa ảnh chân dung trên CCCD và ảnh selfie webcam."""
    start_time = time.perf_counter()

    # 1. Read card file
    try:
        card_bytes = await card_file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể đọc nội dung tệp ảnh CCCD tải lên."
        )

    # 2. Read selfie file
    try:
        selfie_bytes = await selfie_file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể đọc nội dung tệp ảnh selfie tải lên."
        )

    # 3. Validate card image
    try:
        card_image, _ = ImageProcessor.validate_image_bytes(
            data=card_bytes,
            filename=card_file.filename or "card.jpg"
        )
    except ImageValidationError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=f"Ảnh thẻ CCCD không hợp lệ: {e.message}"
        )
    finally:
        del card_bytes

    # 4. Validate selfie image
    try:
        selfie_image, _ = ImageProcessor.validate_image_bytes(
            data=selfie_bytes,
            filename=selfie_file.filename or "selfie.jpg"
        )
    except ImageValidationError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=f"Ảnh chân dung selfie không hợp lệ: {e.message}"
        )
    finally:
        del selfie_bytes

    # 5. Execute face verification pipeline
    try:
        verifier = get_face_verifier()
        result = verifier.verify_faces(
            card_image=card_image,
            selfie_image=selfie_image,
            threshold=threshold,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Quá trình đối sánh khuôn mặt gặp sự cố xử lý nội bộ. Vui lòng thử lại với ảnh chất lượng hơn."
        )
