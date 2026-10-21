import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
import numpy as np

from app.config import settings
from app.schemas.face import (
    FaceBoundingBox,
    FaceLandmarks,
    FaceQualityCheck,
    FaceDetectionResult,
    FaceVerificationResponse,
)


class FaceQualityValidator:
    """Validates face crop resolution, sharpness, illumination, and geometry.

    Distinguishes strictly between:
    - Serious quality defects (is_valid = False): causes automatic rejection.
    - Mild sub-optimal warnings (is_valid = True): warns user while allowing matching.
    """

    # Serious rejection criteria thresholds
    SERIOUS_MIN_SIZE: int = 50
    SERIOUS_MIN_BLUR: float = 15.0
    SERIOUS_MIN_BRIGHTNESS: float = 20.0
    SERIOUS_MAX_BRIGHTNESS: float = 240.0

    @classmethod
    def validate_quality(cls, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> FaceQualityCheck:
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        # Bound coordinates safely within image dimensions
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(img_w, x + w)
        y1 = min(img_h, y + h)

        crop_w = max(0, x1 - x0)
        crop_h = max(0, y1 - y0)
        aspect_ratio = round(crop_w / max(1, crop_h), 2)
        warnings: list[str] = []
        serious_failures: list[str] = []

        if crop_w == 0 or crop_h == 0:
            return FaceQualityCheck(
                is_valid=False,
                width=crop_w,
                height=crop_h,
                brightness=0.0,
                blur_score=0.0,
                aspect_ratio=aspect_ratio,
                warnings=["Tọa độ khuôn mặt nằm hoàn toàn ngoài phạm vi ảnh."],
            )

        crop = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 1. Size / Resolution check
        if crop_w < cls.SERIOUS_MIN_SIZE or crop_h < cls.SERIOUS_MIN_SIZE:
            serious_failures.append(
                f"Kích thước khuôn mặt quá nhỏ ({crop_w}x{crop_h}px) dưới ngưỡng tối thiểu cho phép ({cls.SERIOUS_MIN_SIZE}px)."
            )
        elif crop_w < settings.FACE_MIN_SIZE or crop_h < settings.FACE_MIN_SIZE:
            warnings.append(
                f"Kích thước khuôn mặt ({crop_w}x{crop_h}px) nhỏ hơn khuyến nghị ({settings.FACE_MIN_SIZE}px)."
            )

        # 2. Brightness / Illumination check
        brightness = round(float(np.mean(gray)), 2)
        if brightness < cls.SERIOUS_MIN_BRIGHTNESS:
            serious_failures.append(
                f"Khuôn mặt quá tối/thiếu sáng nghiêm trọng ({brightness}/255 < {cls.SERIOUS_MIN_BRIGHTNESS})."
            )
        elif brightness > cls.SERIOUS_MAX_BRIGHTNESS:
            serious_failures.append(
                f"Khuôn mặt bị lóa sáng/cháy sáng nghiêm trọng ({brightness}/255 > {cls.SERIOUS_MAX_BRIGHTNESS})."
            )
        elif brightness < 40.0:
            warnings.append(f"Khuôn mặt thiếu sáng ({brightness}/255). Nên chụp ở nơi đủ sáng hơn.")
        elif brightness > 225.0:
            warnings.append(f"Khuôn mặt có vùng bị lóa sáng ({brightness}/255).")

        # 3. Sharpness / Blur check (Laplacian variance)
        blur_score = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
        if blur_score < cls.SERIOUS_MIN_BLUR:
            serious_failures.append(
                f"Khuôn mặt bị nhòe mờ/mất nét nghiêm trọng (Laplacian: {blur_score} < {cls.SERIOUS_MIN_BLUR})."
            )
        elif blur_score < 35.0:
            warnings.append(f"Khuôn mặt hơi mờ nét hoặc có hiện tượng rung tay (Laplacian: {blur_score}).")

        # 4. Aspect ratio check
        if aspect_ratio < 0.50 or aspect_ratio > 1.50:
            warnings.append(f"Tỷ lệ khung hình khuôn mặt bất thường ({aspect_ratio}), có thể do góc chụp quá nghiêng.")

        is_valid = len(serious_failures) == 0
        all_warnings = serious_failures + warnings

        return FaceQualityCheck(
            is_valid=is_valid,
            width=crop_w,
            height=crop_h,
            brightness=brightness,
            blur_score=blur_score,
            aspect_ratio=aspect_ratio,
            warnings=all_warnings,
        )


class FaceEngineInterface(ABC):
    """Abstract interface for face detection and feature extraction engines."""

    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> list[dict]:
        """Detect faces in an image."""
        pass

    @abstractmethod
    def extract_embedding(self, image: np.ndarray, face_data: dict) -> np.ndarray:
        """Align face crop and extract a 128-dimensional embedding vector."""
        pass

    @abstractmethod
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two 128-dimensional embeddings."""
        pass


class YuNetSFaceEngine(FaceEngineInterface):
    """OpenCV Zoo native implementation using YuNet (detector) and SFace (recognizer)."""

    def __init__(
        self,
        detector_path: Optional[str] = None,
        recognizer_path: Optional[str] = None,
    ):
        base_dir = settings.FACE_MODEL_DIR
        self.detector_path = detector_path or os.path.join(base_dir, settings.FACE_DETECTOR_MODEL)
        self.recognizer_path = recognizer_path or os.path.join(base_dir, settings.FACE_RECOGNIZER_MODEL)

        if not os.path.exists(self.detector_path):
            raise FileNotFoundError(
                f"Tệp mô hình YuNet không tồn tại tại: {self.detector_path}. "
                f"Vui lòng tải từ OpenCV Zoo và lưu vào {base_dir}/."
            )
        if not os.path.exists(self.recognizer_path):
            raise FileNotFoundError(
                f"Tệp mô hình SFace không tồn tại tại: {self.recognizer_path}. "
                f"Vui lòng tải từ OpenCV Zoo và lưu vào {base_dir}/."
            )

        # Initialize detector with default shape (will update per image)
        self._detector = cv2.FaceDetectorYN.create(
            self.detector_path,
            "",
            (320, 320),
            score_threshold=settings.FACE_SCORE_THRESHOLD,
            nms_threshold=0.3,
            top_k=5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(self.recognizer_path, "")

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)

        if faces is None or len(faces) == 0:
            return []

        results = []
        for face_row in faces:
            x, y, bw, bh = int(face_row[0]), int(face_row[1]), int(face_row[2]), int(face_row[3])
            score = float(face_row[14])
            landmarks = [
                (int(face_row[4]), int(face_row[5])),    # right eye
                (int(face_row[6]), int(face_row[7])),    # left eye
                (int(face_row[8]), int(face_row[9])),    # nose tip
                (int(face_row[10]), int(face_row[11])),  # right mouth corner
                (int(face_row[12]), int(face_row[13])),  # left mouth corner
            ]
            results.append({
                "box": (x, y, bw, bh, score),
                "landmarks": landmarks,
                "raw": face_row,
            })
        return results

    def extract_embedding(self, image: np.ndarray, face_data: dict) -> np.ndarray:
        aligned = self._recognizer.alignCrop(image, face_data["raw"])
        feat = self._recognizer.feature(aligned)
        return feat

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        cosine_sim = self._recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(cosine_sim)


class MockFaceEngine(FaceEngineInterface):
    """Deterministic mock engine ONLY for unit testing and CI test environments.

    MUST NOT be used as a silent fallback during standard application execution.
    """

    def __init__(self):
        self.mock_faces_map: dict[int, list[dict]] = {}
        self.mock_embeddings_map: dict[str, np.ndarray] = {}

    def set_mock_detections(self, image_hash: int, detections: list[dict]):
        self.mock_faces_map[image_hash] = detections

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        img_h, img_w = image.shape[:2]
        key = img_w * 10000 + img_h
        if key in self.mock_faces_map:
            return self.mock_faces_map[key]

        if img_w < 50 or img_h < 50:
            return []

        w = int(img_w * 0.4)
        h = int(img_h * 0.4)
        x = int((img_w - w) / 2)
        y = int((img_h - h) / 2)

        return [{
            "box": (x, y, w, h, 0.98),
            "landmarks": [
                (x + int(w * 0.3), y + int(h * 0.35)),
                (x + int(w * 0.7), y + int(h * 0.35)),
                (x + int(w * 0.5), y + int(h * 0.55)),
                (x + int(w * 0.35), y + int(h * 0.8)),
                (x + int(w * 0.65), y + int(h * 0.8)),
            ],
            "raw": np.array([x, y, w, h, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.98], dtype=np.float32),
        }]

    def extract_embedding(self, image: np.ndarray, face_data: dict) -> np.ndarray:
        mean_val = float(np.mean(image))
        vec = np.ones((1, 128), dtype=np.float32) * (mean_val / 255.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        dot = float(np.dot(feat1.flatten(), feat2.flatten()))
        norm1 = float(np.linalg.norm(feat1))
        norm2 = float(np.linalg.norm(feat2))
        if norm1 > 0 and norm2 > 0:
            return dot / (norm1 * norm2)
        return 0.0


class FaceVerifierService:
    """Orchestrates 1:1 face verification between CCCD front-side portrait and selfie."""

    def __init__(self, engine: Optional[FaceEngineInterface] = None):
        self.engine: Optional[FaceEngineInterface] = None
        self.init_error: Optional[str] = None
        self.is_injected: bool = False

        if engine is not None:
            self.engine = engine
            self.is_injected = True
        else:
            # Normal startup: attempt to load real YuNet + SFace models.
            # HARDENING: NEVER silently fall back to MockFaceEngine in runtime.
            try:
                self.engine = YuNetSFaceEngine()
            except Exception as e:
                self.engine = None
                self.init_error = str(e)

    def is_ready(self) -> bool:
        """Returns True if a valid execution engine is loaded and operational."""
        return self.engine is not None

    def is_mock(self) -> bool:
        """Returns True if the active engine is a mock implementation."""
        return isinstance(self.engine, MockFaceEngine)

    def get_status_message(self) -> str:
        """Returns diagnostic message about the engine state."""
        if not self.is_ready():
            return f"Mô hình nhận dạng khuôn mặt chưa sẵn sàng: {self.init_error}"
        if self.is_mock():
            return "Đang sử dụng MockFaceEngine (môi trường kiểm thử có tiêm mock)"
        return "Mô hình thực tế (YuNet + SFace) đang hoạt động bình thường"

    def _select_cccd_portrait_face(
        self,
        faces: list[dict],
        img_w: int,
        img_h: int,
    ) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
        """Filter face detection for the printed cardholder portrait on Vietnamese CCCD."""
        if not faces:
            return None, "NO_FACE_DETECTED", "Không tìm thấy khuôn mặt chân dung trong ảnh thẻ CCCD."

        candidates = []
        for f in faces:
            x, y, w, h, _ = f["box"]
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Cardholder portrait zone constraint
            if (cx <= 0.52 * img_w) and (0.12 * img_h <= cy <= 0.92 * img_h):
                candidates.append(f)

        if len(candidates) == 1:
            return candidates[0], None, None
        elif len(candidates) > 1:
            candidates.sort(key=lambda c: c["box"][2] * c["box"][3], reverse=True)
            return candidates[0], None, None
        else:
            if len(faces) == 1:
                return faces[0], None, None
            return (
                None,
                "CARD_PORTRAIT_NOT_FOUND",
                "Phát hiện nhiều khuôn mặt ngoài vùng ảnh chân dung chuẩn của thẻ CCCD.",
            )

    def _select_selfie_face(
        self,
        faces: list[dict],
    ) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
        """Validate webcam selfie image containing exactly one person."""
        if not faces:
            return None, "NO_FACE_DETECTED", "Không tìm thấy khuôn mặt trong ảnh chân dung selfie."
        if len(faces) > 1:
            return (
                None,
                "MULTIPLE_FACES_DETECTED",
                f"Phát hiện {len(faces)} khuôn mặt trong ảnh selfie. Vui lòng chỉ chụp một người trước ống kính.",
            )
        return faces[0], None, None

    def _to_detection_result(
        self,
        face_data: Optional[dict],
        image: np.ndarray,
        count: int,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> FaceDetectionResult:
        if face_data is None:
            return FaceDetectionResult(
                detected=False,
                count=count,
                box=None,
                landmarks=None,
                quality=None,
                error_code=error_code,
                error_message=error_message,
            )

        x, y, w, h, score = face_data["box"]
        quality = FaceQualityValidator.validate_quality(image, (x, y, w, h))

        # Check for serious quality failure
        effective_error_code = error_code
        effective_error_message = error_message
        if not quality.is_valid:
            effective_error_code = "LOW_QUALITY_FACE"
            effective_error_message = "; ".join(quality.warnings)

        raw_landmarks = face_data.get("landmarks", [])
        landmarks_obj = None
        if len(raw_landmarks) >= 5:
            landmarks_obj = FaceLandmarks(
                right_eye=raw_landmarks[0],
                left_eye=raw_landmarks[1],
                nose_tip=raw_landmarks[2],
                mouth_right=raw_landmarks[3],
                mouth_left=raw_landmarks[4],
            )

        return FaceDetectionResult(
            detected=quality.is_valid,
            count=count,
            box=FaceBoundingBox(x=x, y=y, width=w, height=h, confidence=round(score, 4)),
            landmarks=landmarks_obj,
            quality=quality,
            error_code=effective_error_code,
            error_message=effective_error_message,
        )

    def verify_faces(
        self,
        card_image: np.ndarray,
        selfie_image: np.ndarray,
        threshold: Optional[float] = None,
    ) -> FaceVerificationResponse:
        start_time = time.perf_counter()

        # HARDENING: Check engine readiness
        if not self.is_ready():
            raise RuntimeError(
                f"FaceVerifierService chưa sẵn sàng: {self.init_error or 'Mô hình chưa được nạp'}"
            )

        # HARDENING: Check mock execution authorization
        if self.is_mock() and not self.is_injected and not settings.ALLOW_MOCK_FACE:
            raise RuntimeError(
                "MockFaceEngine bị vô hiệu hóa trong môi trường thực thi runtime. "
                "Cần cài đặt mô hình YuNet/SFace hoặc bật ALLOW_MOCK_FACE=true cho test."
            )

        effective_threshold = threshold if threshold is not None else settings.FACE_MATCH_THRESHOLD
        if effective_threshold < -1.0 or effective_threshold > 1.0:
            raise ValueError(
                f"Ngưỡng so khớp (threshold) không hợp lệ ({effective_threshold}). Phải nằm trong khoảng [-1.0, 1.0]."
            )

        card_h, card_w = card_image.shape[:2]
        selfie_h, selfie_w = selfie_image.shape[:2]
        warnings: list[str] = []

        # 1. Detect faces in Card image
        try:
            card_all_faces = self.engine.detect_faces(card_image)
        except Exception as e:
            return self._build_error_response(
                f"Lỗi hệ thống khi nhận diện khuôn mặt trên thẻ CCCD: {str(e)}",
                start_time,
                effective_threshold,
            )

        card_face, card_err_code, card_err_msg = self._select_cccd_portrait_face(
            card_all_faces, card_w, card_h
        )
        card_result = self._to_detection_result(
            card_face, card_image, len(card_all_faces), card_err_code, card_err_msg
        )

        # 2. Detect faces in Selfie image
        try:
            selfie_all_faces = self.engine.detect_faces(selfie_image)
        except Exception as e:
            return self._build_error_response(
                f"Lỗi hệ thống khi nhận diện khuôn mặt trong ảnh selfie: {str(e)}",
                start_time,
                effective_threshold,
            )

        selfie_face, selfie_err_code, selfie_err_msg = self._select_selfie_face(selfie_all_faces)
        selfie_result = self._to_detection_result(
            selfie_face, selfie_image, len(selfie_all_faces), selfie_err_code, selfie_err_msg
        )

        # 3. Check for abort due to detection failure OR serious quality defect
        card_has_serious_defect = card_result.quality and not card_result.quality.is_valid
        selfie_has_serious_defect = selfie_result.quality and not selfie_result.quality.is_valid

        if not card_result.detected or not selfie_result.detected or card_has_serious_defect or selfie_has_serious_defect:
            if card_has_serious_defect and card_result.quality:
                for w in card_result.quality.warnings:
                    warnings.append(f"Ảnh thẻ CCCD lỗi chất lượng: {w}")
            elif not card_result.detected and card_result.error_message:
                warnings.append(f"Ảnh thẻ CCCD: {card_result.error_message}")

            if selfie_has_serious_defect and selfie_result.quality:
                for w in selfie_result.quality.warnings:
                    warnings.append(f"Ảnh selfie lỗi chất lượng: {w}")
            elif not selfie_result.detected and selfie_result.error_message:
                warnings.append(f"Ảnh selfie: {selfie_result.error_message}")

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return FaceVerificationResponse(
                success=False,
                match=False,
                similarity_score=0.0,
                confidence_score=0.0,
                threshold=effective_threshold,
                card_face=card_result,
                selfie_face=selfie_result,
                warnings=warnings,
                processing_time_ms=elapsed_ms,
                disclaimer=settings.FACE_DISCLAIMER,
            )

        # 4. Append mild quality warnings (if any)
        if card_result.quality and card_result.quality.warnings:
            for w in card_result.quality.warnings:
                warnings.append(f"Ảnh thẻ: {w}")
        if selfie_result.quality and selfie_result.quality.warnings:
            for w in selfie_result.quality.warnings:
                warnings.append(f"Ảnh selfie: {w}")

        # 5. Extract embeddings and calculate cosine similarity
        try:
            feat_card = self.engine.extract_embedding(card_image, card_face)
            feat_selfie = self.engine.extract_embedding(selfie_image, selfie_face)
            sim_score = self.engine.compute_similarity(feat_card, feat_selfie)
        except Exception as e:
            return self._build_error_response(
                f"Lỗi khi trích xuất hoặc so khớp vector đặc trưng khuôn mặt: {str(e)}",
                start_time,
                effective_threshold,
                card_result,
                selfie_result,
            )

        sim_score = round(float(sim_score), 4)
        is_match = bool(sim_score >= effective_threshold)

        # Compute heuristic display confidence score [0.0, 1.0]
        # (NOT a calibrated statistical posterior probability)
        if sim_score >= effective_threshold:
            denom = max(0.001, 1.0 - effective_threshold)
            confidence = 0.5 + 0.5 * min(1.0, (sim_score - effective_threshold) / denom)
        else:
            denom = max(0.001, effective_threshold - (-1.0))
            confidence = 0.5 * max(0.0, (sim_score - (-1.0)) / denom)
        confidence = round(float(confidence), 4)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return FaceVerificationResponse(
            success=True,
            match=is_match,
            similarity_score=sim_score,
            confidence_score=confidence,
            threshold=effective_threshold,
            card_face=card_result,
            selfie_face=selfie_result,
            warnings=warnings,
            processing_time_ms=elapsed_ms,
            disclaimer=settings.FACE_DISCLAIMER,
        )

    def _build_error_response(
        self,
        error_msg: str,
        start_time: float,
        threshold: float,
        card_result: Optional[FaceDetectionResult] = None,
        selfie_result: Optional[FaceDetectionResult] = None,
    ) -> FaceVerificationResponse:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        dummy_card = card_result or FaceDetectionResult(
            detected=False, count=0, error_code="ERROR", error_message=error_msg
        )
        dummy_selfie = selfie_result or FaceDetectionResult(
            detected=False, count=0, error_code="ERROR", error_message=error_msg
        )
        return FaceVerificationResponse(
            success=False,
            match=False,
            similarity_score=0.0,
            confidence_score=0.0,
            threshold=threshold,
            card_face=dummy_card,
            selfie_face=dummy_selfie,
            warnings=[error_msg],
            processing_time_ms=elapsed_ms,
            disclaimer=settings.FACE_DISCLAIMER,
        )


# Global instance
_face_verifier_instance: Optional[FaceVerifierService] = None


def get_face_verifier() -> FaceVerifierService:
    global _face_verifier_instance
    if _face_verifier_instance is None:
        _face_verifier_instance = FaceVerifierService()
    return _face_verifier_instance


def set_face_verifier(service: Optional[FaceVerifierService]):
    global _face_verifier_instance
    _face_verifier_instance = service
