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

    @staticmethod
    def sort_detections_in_reading_order(detections: list[RawDetection]) -> list[RawDetection]:
        if not detections:
            return []
        has_boxes = any(d.box is not None and len(d.box) >= 4 for d in detections)
        if not has_boxes:
            return detections

        items = []
        no_box_items = []
        for d in detections:
            if d.box and len(d.box) >= 4:
                x_min = min(p[0] for p in d.box)
                y_min = min(p[1] for p in d.box)
                y_max = max(p[1] for p in d.box)
                h = max(y_max - y_min, 1.0)
                y_center = (y_min + y_max) / 2.0
                items.append((y_center, x_min, h, d))
            else:
                no_box_items.append(d)

        # Sort by y_center
        items.sort(key=lambda it: it[0])
        lines: list[list[tuple[float, float, float, RawDetection]]] = []
        for it in items:
            y_c, x_m, h, d = it
            placed = False
            for line in lines:
                avg_y = sum(x[0] for x in line) / len(line)
                avg_h = sum(x[2] for x in line) / len(line)
                threshold = max(0.55 * min(avg_h, h), 8.0)
                if abs(y_c - avg_y) <= threshold:
                    line.append(it)
                    placed = True
                    break
            if not placed:
                lines.append([it])

        sorted_res = []
        for line in lines:
            line.sort(key=lambda it: it[1])
            for it in line:
                sorted_res.append(it[3])

        return sorted_res + no_box_items

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
                box_coords = [[float(pt[0]), float(pt[1])] for pt in box] if box else None
                detections.append(
                    RawDetection(
                        text=text.strip(),
                        confidence=round(float(conf), 4),
                        box=box_coords
                    )
                )
                
        return self.sort_detections_in_reading_order(detections)

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
