import cv2
import numpy as np
import os
from typing import Tuple
from app.config import settings
from app.schemas.ocr import ImageMetadata

class ImageValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class ImageProcessor:
    @staticmethod
    def validate_image_bytes(data: bytes, filename: str = "upload.jpg") -> Tuple[np.ndarray, ImageMetadata]:
        # 1. Check empty file
        if not data or len(data) == 0:
            raise ImageValidationError("Tệp tải lên rỗng (0 bytes).", status_code=400)
            
        # 2. Check maximum file size
        size_bytes = len(data)
        if size_bytes > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise ImageValidationError(
                f"Kích thước tệp vượt quá giới hạn cho phép ({max_mb} MB).",
                status_code=413
            )
            
        # 3. Check file extension
        _, ext = os.path.splitext(filename.lower())
        if ext and ext not in settings.ALLOWED_EXTENSIONS:
            allowed = ", ".join(settings.ALLOWED_EXTENSIONS)
            raise ImageValidationError(
                f"Định dạng tệp không được hỗ trợ ({ext}). Hỗ trợ: {allowed}",
                status_code=415
            )
            
        # 4. Check Magic bytes header (JPEG / PNG)
        is_jpeg = data.startswith(b"\xff\xd8\xff")
        is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
        if not (is_jpeg or is_png):
            raise ImageValidationError(
                "Nội dung tệp không phải là hình ảnh JPEG hoặc PNG hợp lệ.",
                status_code=400
            )
            
        # 5. Decode image using OpenCV
        np_arr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ImageValidationError(
                "Không thể giải mã dữ liệu hình ảnh (tệp bị lỗi hoặc hỏng cấu trúc).",
                status_code=400
            )
            
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        # 6. Check dimensions constraints
        if width < settings.MIN_IMAGE_WIDTH or height < settings.MIN_IMAGE_HEIGHT:
            raise ImageValidationError(
                f"Kích thước hình ảnh quá nhỏ ({width}x{height}px). "
                f"Tối thiểu {settings.MIN_IMAGE_WIDTH}x{settings.MIN_IMAGE_HEIGHT}px.",
                status_code=400
            )
            
        if width > settings.MAX_IMAGE_WIDTH or height > settings.MAX_IMAGE_HEIGHT:
            raise ImageValidationError(
                f"Kích thước hình ảnh quá lớn ({width}x{height}px). "
                f"Tối đa {settings.MAX_IMAGE_WIDTH}x{settings.MAX_IMAGE_HEIGHT}px.",
                status_code=400
            )
            
        metadata = ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            size_bytes=size_bytes,
            format="JPEG" if is_jpeg else "PNG"
        )
        
        return image, metadata

    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Subtle preprocessing to improve text recognition without distorting cards.
        PaddleOCR works effectively on RGB/BGR inputs, so we avoid destructive binarization.
        """
        h, w = image.shape[:2]
        processed = image.copy()
        
        # If image is very small (width < 800), upscale with cubic interpolation
        if w < 800:
            scale = 800.0 / w
            new_w = 800
            new_h = int(h * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
        # If image is excessively large for OCR inference (w > 2500), downscale smoothly
        elif w > 2500:
            scale = 2500.0 / w
            new_w = 2500
            new_h = int(h * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return processed
