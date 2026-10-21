from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from app.config import settings
from app.schemas.ocr import RawDetection

class BaseOCREngine(ABC):
    @abstractmethod
    def detect_and_recognize(self, image: np.ndarray) -> list[RawDetection]:
        """Runs OCR on the given image array and returns raw detections."""
        pass

class PaddleOCREngine(BaseOCREngine):
    def __init__(self, lang: Optional[str] = None):
        self.lang = lang or settings.OCR_LANG
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                show_log=False,
                use_gpu=settings.USE_GPU
            )
        return self._ocr

    def detect_and_recognize(self, image: np.ndarray) -> list[RawDetection]:
        ocr = self._get_ocr()
        # PaddleOCR expects BGR or RGB image
        result = ocr.ocr(image, cls=True)
        detections: list[RawDetection] = []
        
        if not result or not result[0]:
            return detections
            
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            box, (text, conf) = line[0], line[1]
            if text and text.strip():
                # Convert box coordinates to float list
                box_coords = [[float(pt[0]), float(pt[1])] for pt in box] if box else None
                detections.append(
                    RawDetection(
                        text=text.strip(),
                        confidence=round(float(conf), 4),
                        box=box_coords
                    )
                )
                
        # Sort reading order: primary by Y (top-to-bottom), secondary by X (left-to-right)
        detections.sort(key=lambda d: (d.box[0][1] if d.box else 0, d.box[0][0] if d.box else 0))
        return detections

class MockOCREngine(BaseOCREngine):
    """Deterministic mock OCR engine for automated testing and benchmarking."""
    def __init__(self, preset_detections: Optional[list[RawDetection]] = None):
        self.preset_detections = preset_detections or []

    def set_preset_detections(self, detections: list[RawDetection]):
        self.preset_detections = detections

    def detect_and_recognize(self, image: np.ndarray) -> list[RawDetection]:
        return self.preset_detections

# Singleton instance
_engine_instance: Optional[BaseOCREngine] = None

def get_ocr_engine() -> BaseOCREngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PaddleOCREngine()
    return _engine_instance

def set_ocr_engine(engine: BaseOCREngine):
    global _engine_instance
    _engine_instance = engine
