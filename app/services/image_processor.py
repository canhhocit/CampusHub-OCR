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
    def deskew_image(image: np.ndarray, max_angle: float = 30.0) -> np.ndarray:
        """
        Automatically straightens tilted or rotated cards (hand-held mobile captures).
        Uses Otsu inverted binary thresholding + minAreaRect to estimate the skew angle.
        Safety guard: only rotates if 0.7° <= |angle| <= max_angle, avoiding distortion
        of already straight images. Returns original image on any failure.
        """
        try:
            if image is None or image.size == 0:
                return image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            coords = np.column_stack(np.where(thresh > 0))
            if coords.shape[0] < 50:
                return image

            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Safe rotation guard: skip small noise (<0.7 deg) or extreme angles (>max_angle)
            if abs(angle) < 0.7 or abs(angle) > max_angle:
                return image

            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image,
                rot_mat,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        except Exception:
            return image

    @staticmethod
    def clean_shadows_and_contrast(image: np.ndarray) -> np.ndarray:
        """
        Removes uneven lighting/shadow gradients across card and equalizes contrast
        using morphological dilation background estimation and cumulative histogram clipping.
        Returns enhanced BGR/RGB image.
        """
        try:
            if image is None or image.size == 0:
                return image
            is_color = len(image.shape) == 3
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image.copy()

            # Background estimation via dilation and median blur
            dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(gray, bg)

            # Contrast stretch via cumulative histogram clipping (1% clip)
            hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
            acc = np.cumsum(hist)
            max_val = acc[-1]
            clip_val = max_val * 0.005
            min_gray = np.searchsorted(acc, clip_val)
            max_gray = np.searchsorted(acc, max_val - clip_val)
            
            if max_gray > min_gray:
                alpha = 255.0 / (max_gray - min_gray)
                beta = -min_gray * alpha
                enhanced_gray = cv2.convertScaleAbs(diff, alpha=alpha, beta=beta)
            else:
                enhanced_gray = diff

            return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR) if is_color else enhanced_gray
        except Exception:
            return image

    @classmethod
    def preprocess_for_ocr(cls, image: np.ndarray, deskew: bool = True) -> np.ndarray:
        """
        Preprocessing pipeline for OCR:
        1. Auto-deskew hand-held angle tilt (if enabled and within safe angle bounds).
        2. Proportional upscale for small images (w < 800) and smooth downscale for huge images (w > 2500).
        """
        processed = image.copy()
        
        # 1. Deskew
        if deskew:
            processed = cls.deskew_image(processed)

        # 2. Resize constraints
        h, w = processed.shape[:2]
        if w < 800:
            scale = 800.0 / w
            new_w = 800
            new_h = int(h * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif w > 2500:
            scale = 2500.0 / w
            new_w = 2500
            new_h = int(h * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return processed
