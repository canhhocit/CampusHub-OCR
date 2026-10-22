import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.config import settings
from app.schemas.liveness import (
    LivenessSessionCreateResponse,
    LivenessStepVerifyResponse,
    LivenessSessionStatusResponse,
    LivenessHealthResponse,
)
from app.services.image_processor import ImageProcessor, ImageValidationError
from app.services.liveness_service import get_liveness_service

liveness_router = APIRouter()


@liveness_router.get("/health", response_model=LivenessHealthResponse)
def liveness_health_check():
    """Kiểm tra trạng thái sẵn sàng của mô hình và dịch vụ kiểm tra thực thể sống & chống giả mạo."""
    service = get_liveness_service()
    return service.get_health_status()


@liveness_router.post("/session/create", response_model=LivenessSessionCreateResponse)
def create_liveness_session():
    """Khởi tạo phiên kiểm tra thực thể sống V1 mới với chuỗi thử thách ngẫu nhiên."""
    service = get_liveness_service()
    return service.create_session()


@liveness_router.get("/session/{session_id}", response_model=LivenessSessionStatusResponse)
def get_liveness_session_status(session_id: str):
    """Truy vấn trạng thái, tiến độ và kết quả của phiên kiểm tra thực thể sống."""
    service = get_liveness_service()
    try:
        return service.get_session_status(session_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Phiên kiểm tra không tồn tại hoặc đã hết hạn: {session_id}",
        )


@liveness_router.post("/session/{session_id}/verify-step", response_model=LivenessStepVerifyResponse)
async def verify_liveness_step(
    session_id: str,
    frames: List[UploadFile] = File(
        ...,
        description=f"Chuỗi khung hình webcam liên tiếp ({settings.LIVENESS_MIN_FRAMES} - {settings.LIVENESS_MAX_FRAMES} khung hình)",
    ),
):
    """
    Xác minh một bước thử thách cử động (quay đầu, gật đầu) dựa trên chuỗi khung hình liên tiếp.
    Yêu cầu chuỗi khung hình có sự chuyển động thực thể sống, không nhận ảnh tĩnh đơn lẻ.
    """
    # 1. Enforce sequence frame count limits
    if not frames or len(frames) < settings.LIVENESS_MIN_FRAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Số lượng khung hình không đủ. Yêu cầu tối thiểu {settings.LIVENESS_MIN_FRAMES} khung hình liên tiếp.",
        )

    if len(frames) > settings.LIVENESS_MAX_FRAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Số lượng khung hình vượt quá giới hạn tối đa ({settings.LIVENESS_MAX_FRAMES} khung hình).",
        )

    service = get_liveness_service()

    # Fast-check session existence before reading large payloads
    try:
        current_status = service.get_session_status(session_id)
        if current_status.status in ("PASSED", "EXPIRED", "FAILED"):
            return service.verify_step(session_id, [])
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Phiên kiểm tra không tồn tại hoặc đã hết hạn: {session_id}",
        )

    # 2. Read and validate each frame safely in-memory
    decoded_frames: List[np.ndarray] = []
    total_bytes = 0

    for idx, upload_file in enumerate(frames):
        try:
            content = await upload_file.read()
            total_bytes += len(content)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Không thể đọc dữ liệu khung hình thứ {idx + 1}.",
            )

        if total_bytes > settings.LIVENESS_MAX_PAYLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Tổng dung lượng chuỗi khung hình vượt quá giới hạn ({settings.LIVENESS_MAX_PAYLOAD_BYTES // (1024 * 1024)} MB).",
            )

        try:
            frame_np, _ = ImageProcessor.validate_image_bytes(
                content,
                filename=upload_file.filename or f"frame_{idx}.jpg",
            )
            decoded_frames.append(frame_np)
        except ImageValidationError as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=f"Khung hình thứ {idx + 1} không hợp lệ: {e.message}",
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Lỗi giải mã khung hình thứ {idx + 1}.",
            )

    # 3. Verify step with service
    return service.verify_step(session_id, decoded_frames)
