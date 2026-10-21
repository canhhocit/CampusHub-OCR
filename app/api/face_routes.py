import time
import math
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
    """Kiểm tra trạng thái sẵn sàng của mô hình và runtime nhận diện khuôn mặt."""
    verifier = get_face_verifier()

    if not verifier.is_ready():
        return FaceHealthResponse(
            status="unavailable",
            version=settings.APP_VERSION,
            detector="YuNet (ONNX - OpenCV Zoo)",
            recognizer="SFace (ONNX - OpenCV Zoo)",
            device="CPU",
            models_loaded=False,
            threshold=settings.FACE_MATCH_THRESHOLD,
            engine_type="unavailable",
            error_message=verifier.init_error or "Mô hình YuNet hoặc SFace chưa được nạp.",
        )

    if verifier.is_mock():
        return FaceHealthResponse(
            status="ok",
            version=settings.APP_VERSION,
            detector="MockFaceEngine (Testing)",
            recognizer="MockFaceEngine (Testing)",
            device="CPU",
            models_loaded=False,
            threshold=settings.FACE_MATCH_THRESHOLD,
            engine_type="mock",
            error_message="Hệ thống đang chạy MockFaceEngine được tiêm thủ công cho môi trường kiểm thử.",
        )

    # Real YuNet + SFace engine active
    return FaceHealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        detector="YuNet (ONNX - OpenCV Zoo)",
        recognizer="SFace (ONNX - OpenCV Zoo)",
        device="CPU",
        models_loaded=True,
        threshold=settings.FACE_MATCH_THRESHOLD,
        engine_type="real",
        error_message=None,
    )


@face_router.post("/verify", response_model=FaceVerificationResponse)
async def verify_faces(
    card_file: UploadFile = File(..., description="Tệp ảnh mặt trước CCCD có chân dung in (JPG, PNG)"),
    selfie_file: UploadFile = File(..., description="Tệp ảnh chân dung chụp từ webcam (JPG, PNG)"),
    threshold: Optional[float] = Form(default=None, description="Ngưỡng so khớp cosine tùy chỉnh [-1.0, 1.0] (mặc định 0.363)"),
):
    """Xác thực đối sánh khuôn mặt 1:1 giữa ảnh chân dung trên CCCD và ảnh selfie webcam."""
    verifier = get_face_verifier()

    # 1. HARDENING: Fail fast if real engine is not operational
    if not verifier.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Dịch vụ so khớp khuôn mặt hiện không khả dụng. "
                f"Lý do: {verifier.init_error or 'Chưa cài đặt mô hình YuNet/SFace'}. "
                "Vui lòng kiểm tra thư mục models/face/ hoặc liên hệ quản trị viên."
            ),
        )

    # 2. HARDENING: Disallow un-injected mock engine in production runtime
    if verifier.is_mock() and not verifier.is_injected and not settings.ALLOW_MOCK_FACE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình nhận diện khuôn mặt giả lập (Mock) bị vô hiệu hóa trong môi trường thực thi runtime.",
        )

    # 3. HARDENING: Validate custom threshold range
    if threshold is not None:
        if math.isnan(threshold) or math.isinf(threshold) or threshold < -1.0 or threshold > 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ngưỡng so khớp (threshold) không hợp lệ. Giá trị phải là số thực trong khoảng từ -1.0 đến 1.0.",
            )

    # 4. Read card file
    try:
        card_bytes = await card_file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể đọc nội dung tệp ảnh CCCD tải lên."
        )

    # 5. Read selfie file
    try:
        selfie_bytes = await selfie_file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể đọc nội dung tệp ảnh selfie tải lên."
        )

    # 6. Validate card image structure and headers
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

    # 7. Validate selfie image structure and headers
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

    # 8. Execute hardened face verification pipeline
    try:
        result = verifier.verify_faces(
            card_image=card_image,
            selfie_image=selfie_image,
            threshold=threshold,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Quá trình đối sánh khuôn mặt gặp sự cố xử lý nội bộ. Vui lòng thử lại với ảnh chất lượng hơn."
        )
