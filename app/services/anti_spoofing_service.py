"""
app/services/anti_spoofing_service.py
======================================
Dedicated Passive Face Anti-Spoofing Service (Presentation Attack Detection - PAD).

Architecture:
- Utilizes MiniFASNet (Silent-Face-Anti-Spoofing, Apache-2.0 License by Minivision AI).
- Multi-scale model fusion using OpenCV DNN (cv2.dnn.readNetFromONNX):
    * Scale 2.7 (MiniFASNetV2, 80x80)
    * Scale 4.0 (MiniFASNetV1SE, 80x80)
- Detects physical presentation attacks:
    * Class 0: Paper Photo (in ấn)
    * Class 1: Real / Live Face (người thật trước camera)
    * Class 2: Screen Photo / Replay Attack (video phát lại trên màn hình điện thoại/máy tính)

Security & Governance:
- Fail-Closed: Technical errors or missing models do not silently pass as LIVE.
- Privacy-preserving: Operates strictly in-memory; does not persist or log facial images or embeddings.
- Explicit Outcomes: Returns LIVE, SPOOF, UNCERTAIN, or ERROR based on documented thresholds.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

from app.config import settings
from app.schemas.liveness import AntiSpoofingOutcome

logger = logging.getLogger(__name__)


@dataclass
class AntiSpoofingResult:
    """Encapsulates the evaluation outcome of passive face anti-spoofing."""

    outcome: AntiSpoofingOutcome
    is_live: bool
    score_real: float
    score_paper: float
    score_screen: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class BaseAntiSpoofingDetector(ABC):
    """Abstract interface for passive face anti-spoofing detection."""

    @abstractmethod
    def predict(
        self, img_bgr: np.ndarray, face_box: Tuple[int, int, int, int]
    ) -> AntiSpoofingResult:
        """
        Evaluate a single face crop for presentation attack spoofing.

        Args:
            img_bgr: BGR image array.
            face_box: (x, y, w, h) bounding box coordinates of the detected face.

        Returns:
            AntiSpoofingResult with explicit outcome (LIVE, SPOOF, UNCERTAIN, ERROR).
        """
        pass


class MiniFASNetAntiSpoofingDetector(BaseAntiSpoofingDetector):
    """
    Production-grade passive anti-spoofing detector running MiniFASNet models
    via OpenCV DNN on CPU without requiring heavy PyTorch/TensorFlow dependencies.
    """

    def __init__(
        self,
        model_path_v2: Optional[str] = None,
        model_path_v1se: Optional[str] = None,
    ):
        self.model_path_v2 = model_path_v2 or settings.ANTI_SPOOFING_MODEL_PATH_V2
        self.model_path_v1se = model_path_v1se or settings.ANTI_SPOOFING_MODEL_PATH_V1SE

        self._net_v2 = None
        self._net_v1se = None
        self._is_available = False
        self._init_error = None

        self._load_models()

    def _load_models(self) -> None:
        """Load ONNX models via cv2.dnn.readNetFromONNX."""
        try:
            if not os.path.exists(self.model_path_v2):
                raise FileNotFoundError(f"MiniFASNet V2 model not found at {self.model_path_v2}")
            if not os.path.exists(self.model_path_v1se):
                raise FileNotFoundError(f"MiniFASNet V1SE model not found at {self.model_path_v1se}")

            self._net_v2 = cv2.dnn.readNetFromONNX(self.model_path_v2)
            self._net_v1se = cv2.dnn.readNetFromONNX(self.model_path_v1se)

            # Set CPU backend
            self._net_v2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net_v2.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self._net_v1se.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net_v1se.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self._is_available = True
            logger.info(
                "MiniFASNet anti-spoofing models loaded successfully: %s, %s",
                self.model_path_v2,
                self.model_path_v1se,
            )
        except Exception as e:
            self._is_available = False
            self._init_error = str(e)
            logger.error("Failed to initialize MiniFASNet anti-spoofing detector: %s", e)

    @property
    def is_available(self) -> bool:
        return self._is_available

    @staticmethod
    def _get_new_box(
        src_w: int, src_h: int, bbox: Tuple[int, int, int, int], scale: float
    ) -> Tuple[int, int, int, int]:
        """
        Calculate expanded crop box around face according to MiniFASNet specification.

        Args:
            src_w: Image width
            src_h: Image height
            bbox: (x, y, w, h)
            scale: Expansion scale factor (e.g. 2.7 or 4.0)
        """
        x, y, box_w, box_h = bbox
        if box_w <= 0 or box_h <= 0:
            return 0, 0, 0, 0

        scale = min(
            (src_h - 1) / float(box_h),
            min((src_w - 1) / float(box_w), float(scale)),
        )

        new_width = box_w * scale
        new_height = box_h * scale
        center_x = box_w / 2.0 + x
        center_y = box_h / 2.0 + y

        left_top_x = center_x - new_width / 2.0
        left_top_y = center_y - new_height / 2.0
        right_bottom_x = center_x + new_width / 2.0
        right_bottom_y = center_y + new_height / 2.0

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return (
            int(round(left_top_x)),
            int(round(left_top_y)),
            int(round(right_bottom_x)),
            int(round(right_bottom_y)),
        )

    def _forward_model(
        self, net, img_bgr: np.ndarray, bbox: Tuple[int, int, int, int], scale: float
    ) -> np.ndarray:
        """Preprocesses expanded crop and runs forward inference through a single MiniFASNet network."""
        src_h, src_w = img_bgr.shape[:2]
        x1, y1, x2, y2 = self._get_new_box(src_w, src_h, bbox, scale)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid crop coordinates computed from bounding box.")

        crop = img_bgr[y1 : y2 + 1, x1 : x2 + 1]
        if crop.size == 0:
            raise ValueError("Empty crop obtained from image.")

        # Resize to fixed input size (80, 80)
        resized = cv2.resize(crop, (80, 80)).astype(np.float32)

        # Retain BGR channel order (as expected by original MiniFASNet training)
        # Convert HWC -> CHW -> NCHW
        chw = np.transpose(resized, (2, 0, 1))
        blob = np.expand_dims(chw, axis=0)

        net.setInput(blob)
        logits = net.forward()

        # Numerically stable Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs[0]  # shape: (3,) -> [paper, real, screen]

    def predict(
        self, img_bgr: np.ndarray, face_box: Tuple[int, int, int, int]
    ) -> AntiSpoofingResult:
        if not self._is_available:
            return AntiSpoofingResult(
                outcome=AntiSpoofingOutcome.ERROR,
                is_live=False,
                score_real=0.0,
                score_paper=0.0,
                score_screen=0.0,
                confidence=0.0,
                error_message=f"Anti-spoofing model unavailable: {self._init_error}",
            )

        if img_bgr is None or img_bgr.size == 0:
            return AntiSpoofingResult(
                outcome=AntiSpoofingOutcome.ERROR,
                is_live=False,
                score_real=0.0,
                score_paper=0.0,
                score_screen=0.0,
                confidence=0.0,
                error_message="Input image is empty or invalid.",
            )

        try:
            # Model 1: Scale 2.7 (MiniFASNetV2)
            probs_v2 = self._forward_model(self._net_v2, img_bgr, face_box, scale=2.7)

            # Model 2: Scale 4.0 (MiniFASNetV1SE)
            probs_v1se = self._forward_model(self._net_v1se, img_bgr, face_box, scale=4.0)

            # Ensemble average
            avg_probs = (probs_v2 + probs_v1se) / 2.0
            score_paper = float(round(float(avg_probs[0]), 4))
            score_real = float(round(float(avg_probs[1]), 4))
            score_screen = float(round(float(avg_probs[2]), 4))

            details = {
                "score_paper": score_paper,
                "score_real": score_real,
                "score_screen": score_screen,
                "v2_scores": [round(float(x), 4) for x in probs_v2],
                "v1se_scores": [round(float(x), 4) for x in probs_v1se],
                "threshold_live": settings.ANTI_SPOOFING_THRESHOLD_LIVE,
                "threshold_spoof": settings.ANTI_SPOOFING_THRESHOLD_SPOOF,
            }

            # Decision logic based on documented thresholds
            if score_real >= settings.ANTI_SPOOFING_THRESHOLD_LIVE:
                outcome = AntiSpoofingOutcome.LIVE
                is_live = True
                confidence = score_real
            elif (
                score_real <= settings.ANTI_SPOOFING_THRESHOLD_SPOOF
                or (score_paper + score_screen) >= settings.ANTI_SPOOFING_THRESHOLD_SPOOF_SUM
            ):
                outcome = AntiSpoofingOutcome.SPOOF
                is_live = False
                confidence = max(score_paper, score_screen)
            else:
                outcome = AntiSpoofingOutcome.UNCERTAIN
                is_live = False
                confidence = score_real

            return AntiSpoofingResult(
                outcome=outcome,
                is_live=is_live,
                score_real=score_real,
                score_paper=score_paper,
                score_screen=score_screen,
                confidence=round(confidence, 4),
                details=details,
            )

        except Exception as e:
            logger.error("Anti-spoofing inference error: %s", e)
            return AntiSpoofingResult(
                outcome=AntiSpoofingOutcome.ERROR,
                is_live=False,
                score_real=0.0,
                score_paper=0.0,
                score_screen=0.0,
                confidence=0.0,
                error_message=f"Inference exception: {str(e)}",
            )


class MockAntiSpoofingDetector(BaseAntiSpoofingDetector):
    """Deterministic mock anti-spoofing detector for isolated unit and API tests."""

    def __init__(self, default_outcome: AntiSpoofingOutcome = AntiSpoofingOutcome.LIVE):
        self.default_outcome = default_outcome
        self.preset_results: List[AntiSpoofingResult] = []
        self.call_count = 0

    def set_preset_results(self, results: List[AntiSpoofingResult]) -> None:
        self.preset_results = list(results)
        self.call_count = 0

    def predict(
        self, img_bgr: np.ndarray, face_box: Tuple[int, int, int, int]
    ) -> AntiSpoofingResult:
        self.call_count += 1
        if self.preset_results:
            return self.preset_results.pop(0)

        if self.default_outcome == AntiSpoofingOutcome.LIVE:
            return AntiSpoofingResult(
                outcome=AntiSpoofingOutcome.LIVE,
                is_live=True,
                score_real=0.98,
                score_paper=0.01,
                score_screen=0.01,
                confidence=0.98,
                details={"mock": True},
            )
        elif self.default_outcome == AntiSpoofingOutcome.SPOOF:
            return AntiSpoofingResult(
                outcome=AntiSpoofingOutcome.SPOOF,
                is_live=False,
                score_real=0.15,
                score_paper=0.05,
                score_screen=0.80,
                confidence=0.80,
                details={"mock": True},
            )
        elif self.default_outcome == AntiSpoofingOutcome.UNCERTAIN:
            return AntiSpoofingResult(
                outcome=AntiSpoofingOutcome.UNCERTAIN,
                is_live=False,
                score_real=0.60,
                score_paper=0.20,
                score_screen=0.20,
                confidence=0.60,
                details={"mock": True},
            )
        else:
            return AntiSpoofingResult(
                outcome=AntiSpoofingOutcome.ERROR,
                is_live=False,
                score_real=0.0,
                score_paper=0.0,
                score_screen=0.0,
                confidence=0.0,
                error_message="Mocked technical inference failure",
            )


# Global singleton provider
_global_anti_spoofing_detector: Optional[BaseAntiSpoofingDetector] = None


def get_anti_spoofing_detector() -> BaseAntiSpoofingDetector:
    global _global_anti_spoofing_detector
    if _global_anti_spoofing_detector is None:
        if settings.ANTI_SPOOFING_ENABLED:
            _global_anti_spoofing_detector = MiniFASNetAntiSpoofingDetector()
        else:
            # When disabled, default mock LIVE is used for development/testing
            _global_anti_spoofing_detector = MockAntiSpoofingDetector(
                default_outcome=AntiSpoofingOutcome.LIVE
            )
    return _global_anti_spoofing_detector


def set_anti_spoofing_detector(detector: Optional[BaseAntiSpoofingDetector]) -> None:
    global _global_anti_spoofing_detector
    _global_anti_spoofing_detector = detector
