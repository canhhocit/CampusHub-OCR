import os
import time
import math
import secrets
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

from app.config import settings
from app.schemas.liveness import (
    ChallengeAction,
    ChallengeInfo,
    LivenessSessionCreateResponse,
    LivenessStepVerifyResponse,
    LivenessSessionStatusResponse,
    AntiSpoofingOutcome,
)

logger = logging.getLogger(__name__)

# Human-readable instructions for each challenge action
CHALLENGE_INSTRUCTIONS: Dict[ChallengeAction, str] = {
    ChallengeAction.TURN_HEAD_LEFT: "Vui lòng nhìn thẳng, sau đó quay đầu từ từ sang bên TRÁI của bạn.",
    ChallengeAction.TURN_HEAD_RIGHT: "Vui lòng nhìn thẳng, sau đó quay đầu từ từ sang bên PHẢI của bạn.",
    ChallengeAction.NOD_HEAD: "Vui lòng nhìn thẳng, sau đó gật nhẹ đầu xuống dưới.",
    ChallengeAction.FACE_CENTER: "Vui lòng giữ khuôn mặt nhìn thẳng vào camera.",
}


class LivenessSessionState:
    """Represents an active in-memory liveness challenge session."""

    def __init__(
        self,
        session_id: str,
        challenges: List[ChallengeAction],
        timeout_seconds: int = 120,
    ):
        self.session_id: str = session_id
        self.challenges: List[ChallengeAction] = challenges
        self.total_steps: int = len(challenges)
        self.current_step_index: int = 0
        self.step_attempts: int = 0
        self.status: str = "IN_PROGRESS"  # IN_PROGRESS, PASSED, FAILED, EXPIRED
        self.created_at: float = time.time()
        self.expires_at: float = self.created_at + timeout_seconds
        self.history: List[Dict[str, Any]] = []

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def get_current_challenge(self) -> Optional[ChallengeInfo]:
        if self.current_step_index >= self.total_steps or self.status != "IN_PROGRESS":
            return None
        action = self.challenges[self.current_step_index]
        return ChallengeInfo(
            action=action,
            instruction=CHALLENGE_INSTRUCTIONS.get(action, "Thực hiện cử động theo yêu cầu."),
            step=self.current_step_index + 1,
            total_steps=self.total_steps,
            timeout_seconds=30,
        )


class FramingValidationResult:
    """Encapsulates the result of face position, size, and framing validation."""

    def __init__(
        self,
        is_valid: bool,
        reason_code: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.is_valid: bool = is_valid
        self.reason_code: Optional[str] = reason_code
        self.message: Optional[str] = message
        self.details: Dict[str, Any] = details or {}


class FaceFramingValidator:
    """
    Dedicated, deterministic validator evaluating face positioning, size,
    and boundary clipping against the active guide oval for webcam liveness frames.
    """

    @staticmethod
    def validate_frame(
        box: Tuple[int, int, int, int, float],
        landmarks: List[Tuple[int, int]],
        img_w: int,
        img_h: int,
    ) -> FramingValidationResult:
        if img_w <= 0 or img_h <= 0:
            return FramingValidationResult(
                is_valid=False,
                reason_code="INVALID_FRAME_DIMENSIONS",
                message="Kích thước khung hình không hợp lệ.",
            )

        bx, by, bw, bh = box[0], box[1], box[2], box[3]

        # 1. Normalize all coordinates against actual frame dimensions
        norm_x = bx / float(img_w)
        norm_y = by / float(img_h)
        norm_w = bw / float(img_w)
        norm_h = bh / float(img_h)
        box_right = (bx + bw) / float(img_w)
        box_bottom = (by + bh) / float(img_h)
        center_x = norm_x + (norm_w / 2.0)
        center_y = norm_y + (norm_h / 2.0)

        details: Dict[str, Any] = {
            "center_x": round(center_x, 4),
            "center_y": round(center_y, 4),
            "norm_w": round(norm_w, 4),
            "norm_h": round(norm_h, 4),
            "box_top": round(norm_y, 4),
            "box_bottom": round(box_bottom, 4),
            "box_left": round(norm_x, 4),
            "box_right": round(box_right, 4),
        }

        margin = settings.LIVENESS_FRAME_MARGIN_RATIO

        # 2. Frame boundary clipping checks (face cut off by image borders)
        if norm_y < margin:
            return FramingValidationResult(
                is_valid=False,
                reason_code="FACE_PARTIALLY_VISIBLE",
                message="Khuôn mặt bị cắt ở cạnh trên khung hình. Vui lòng hạ camera hoặc ngồi thấp hơn để thấy trọn vẹn khuôn mặt.",
                details=details,
            )

        if box_bottom > (1.0 - margin):
            return FramingValidationResult(
                is_valid=False,
                reason_code="FACE_PARTIALLY_VISIBLE",
                message="Khuôn mặt bị cắt ở cạnh dưới khung hình. Vui lòng nâng camera hoặc ngồi cao hơn để thấy trọn vẹn khuôn mặt.",
                details=details,
            )

        if norm_x < margin or box_right > (1.0 - margin):
            return FramingValidationResult(
                is_valid=False,
                reason_code="FACE_PARTIALLY_VISIBLE",
                message="Khuôn mặt bị cắt ở viền cạnh khung hình. Vui lòng căn chỉnh toàn bộ khuôn mặt vào giữa camera.",
                details=details,
            )

        # 3. Landmark clipping checks: All facial landmarks must be well inside frame
        if landmarks:
            for lx, ly in landmarks:
                nlx = lx / float(img_w)
                nly = ly / float(img_h)
                if nlx < 0.02 or nlx > 0.98 or nly < 0.02 or nly > 0.98:
                    return FramingValidationResult(
                        is_valid=False,
                        reason_code="FACE_PARTIALLY_VISIBLE",
                        message="Ngũ quan khuôn mặt bị tràn ra ngoài viền khung hình. Vui lòng lùi lại để mặt nằm trọn trong camera.",
                        details=details,
                    )

        # 4. Vertical landmark distribution checks (Eye and mouth placement)
        if len(landmarks) >= 5:
            re_y = landmarks[0][1] / float(img_h)
            le_y = landmarks[1][1] / float(img_h)
            eye_mid_y = (re_y + le_y) / 2.0

            rm_y = landmarks[3][1] / float(img_h)
            lm_y = landmarks[4][1] / float(img_h)
            mouth_mid_y = (rm_y + lm_y) / 2.0

            details["eye_mid_y"] = round(eye_mid_y, 4)
            details["mouth_mid_y"] = round(mouth_mid_y, 4)

            # Reject if only upper face/eyes are in frame or face is lowered
            if eye_mid_y > settings.LIVENESS_MAX_EYE_Y_RATIO:
                return FramingValidationResult(
                    is_valid=False,
                    reason_code="FACE_OUT_OF_GUIDE",
                    message="Khuôn mặt đang đặt quá thấp (chỉ thấy phần trên/mắt trong khung). Vui lòng nâng mặt hoặc đưa toàn bộ khuôn mặt vào khung hướng dẫn.",
                    details=details,
                )

            # Mouth must be well within image and not touching bottom boundary
            if mouth_mid_y > settings.LIVENESS_MAX_MOUTH_Y_RATIO:
                return FramingValidationResult(
                    is_valid=False,
                    reason_code="FACE_OUT_OF_GUIDE",
                    message="Phần miệng/cằm đang ở quá sát cạnh dưới. Vui lòng nâng camera lên để thấy rõ cả cằm và miệng.",
                    details=details,
                )

        # 5. Face size ratio checks (Relative to frame)
        if norm_h < settings.LIVENESS_FACE_MIN_HEIGHT_RATIO or norm_w < settings.LIVENESS_FACE_MIN_WIDTH_RATIO:
            return FramingValidationResult(
                is_valid=False,
                reason_code="FACE_OUT_OF_GUIDE",
                message="Khuôn mặt quá nhỏ hoặc ở quá xa khung hướng dẫn. Vui lòng tiến lại gần camera hơn.",
                details=details,
            )

        if norm_h > settings.LIVENESS_FACE_MAX_HEIGHT_RATIO or norm_w > settings.LIVENESS_FACE_MAX_WIDTH_RATIO:
            return FramingValidationResult(
                is_valid=False,
                reason_code="FACE_OUT_OF_GUIDE",
                message="Khuôn mặt quá lớn so với khung hướng dẫn. Vui lòng lùi ra xa camera một chút.",
                details=details,
            )

        # 6. Center positioning validation within allowed guide zone
        if center_y > settings.LIVENESS_FACE_CENTER_Y_MAX:
            return FramingValidationResult(
                is_valid=False,
                reason_code="FACE_OUT_OF_GUIDE",
                message="Khuôn mặt đang đặt quá thấp so với khung hướng dẫn. Vui lòng nâng mặt hoặc nâng camera lên.",
                details=details,
            )

        if center_y < settings.LIVENESS_FACE_CENTER_Y_MIN:
            return FramingValidationResult(
                is_valid=False,
                reason_code="FACE_OUT_OF_GUIDE",
                message="Khuôn mặt đang đặt quá cao so với khung hướng dẫn. Vui lòng hạ mặt hoặc hạ camera xuống.",
                details=details,
            )

        if center_x < settings.LIVENESS_FACE_CENTER_X_MIN or center_x > settings.LIVENESS_FACE_CENTER_X_MAX:
            return FramingValidationResult(
                is_valid=False,
                reason_code="FACE_OUT_OF_GUIDE",
                message="Khuôn mặt đang bị lệch sang một bên. Vui lòng căn chỉnh toàn bộ khuôn mặt vào giữa khung hướng dẫn.",
                details=details,
            )

        return FramingValidationResult(is_valid=True, details=details)


class BaseLivenessDetector:
    """Abstract base detector interface for frame analysis."""

    def detect_face_and_landmarks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        raise NotImplementedError


class YuNetLivenessDetector(BaseLivenessDetector):
    """Reuses YuNet ONNX face detector for liveness landmark analysis."""

    def __init__(self, model_path: Optional[str] = None):
        base_dir = settings.FACE_MODEL_DIR
        self.model_path = model_path or os.path.join(base_dir, settings.FACE_DETECTOR_MODEL)
        self.score_threshold = settings.FACE_SCORE_THRESHOLD

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Tệp mô hình YuNet không tồn tại tại: {self.model_path}")

        self._detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            (320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )

    def detect_face_and_landmarks(self, image: np.ndarray) -> List[Dict[str, Any]]:
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
            })
        return results


class MockLivenessDetector(BaseLivenessDetector):
    """Mock detector for fast, deterministic unit testing without requiring real face frames."""

    def __init__(self):
        self.preset_results: List[List[Dict[str, Any]]] = []
        self._call_count: int = 0

    def set_preset_results(self, results_per_frame: List[List[Dict[str, Any]]]):
        self.preset_results = results_per_frame
        self._call_count = 0

    def detect_face_and_landmarks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self._call_count < len(self.preset_results):
            res = self.preset_results[self._call_count]
            self._call_count += 1
            return res
        # Default fallback: 1 centered face
        h, w = image.shape[:2]
        return [{
            "box": (w // 4, h // 4, w // 2, h // 2, 0.99),
            "landmarks": [
                (w // 3, h // 3),        # right eye
                (2 * w // 3, h // 3),    # left eye
                (w // 2, h // 2),        # nose tip
                (w // 3 + 20, 2 * h // 3),
                (2 * w // 3 - 20, 2 * h // 3),
            ],
        }]


class LivenessService:
    """Core Liveness Detection service managing active sessions and frame verification."""

    def __init__(
        self,
        detector: Optional[BaseLivenessDetector] = None,
        anti_spoofing_detector: Optional[Any] = None,
    ):
        self._detector: Optional[BaseLivenessDetector] = detector
        self._anti_spoofing_detector = anti_spoofing_detector
        self._sessions: Dict[str, LivenessSessionState] = {}
        self._lock: threading.Lock = threading.Lock()

    def _get_detector(self) -> BaseLivenessDetector:
        if self._detector is None:
            self._detector = YuNetLivenessDetector()
        return self._detector

    def _get_anti_spoofing_detector(self):
        if self._anti_spoofing_detector is None:
            from app.services.anti_spoofing_service import get_anti_spoofing_detector
            self._anti_spoofing_detector = get_anti_spoofing_detector()
        return self._anti_spoofing_detector

    def set_detector(self, detector: Optional[BaseLivenessDetector]):
        """Injects a custom or mock detector for testing."""
        self._detector = detector

    def set_anti_spoofing_detector(self, detector: Optional[Any]):
        """Injects a custom or mock anti-spoofing detector for testing."""
        self._anti_spoofing_detector = detector

    def _cleanup_expired_sessions(self):
        """Purges expired sessions to ensure memory remains bounded."""
        now = time.time()
        expired_ids = [sid for sid, s in self._sessions.items() if s.expires_at < now]
        for sid in expired_ids:
            del self._sessions[sid]

        # Evict oldest if capacity exceeded
        if len(self._sessions) > settings.LIVENESS_MAX_SESSIONS:
            sorted_sessions = sorted(self._sessions.items(), key=lambda item: item[1].created_at)
            overflow_count = len(self._sessions) - settings.LIVENESS_MAX_SESSIONS
            for sid, _ in sorted_sessions[:overflow_count]:
                del self._sessions[sid]

    def create_session(self) -> LivenessSessionCreateResponse:
        """Creates a short-lived session with an unpredictably randomized challenge sequence."""
        with self._lock:
            self._cleanup_expired_sessions()

            # Generate random session ID
            session_id = f"live_{secrets.token_urlsafe(16)}"

            # Randomized challenge sequence (2 active steps):
            # E.g. [LEFT, RIGHT] or [RIGHT, LEFT]
            available_pairs = [
                [ChallengeAction.TURN_HEAD_LEFT, ChallengeAction.TURN_HEAD_RIGHT],
                [ChallengeAction.TURN_HEAD_RIGHT, ChallengeAction.TURN_HEAD_LEFT],
            ]
            chosen_challenges = secrets.choice(available_pairs)

            session = LivenessSessionState(
                session_id=session_id,
                challenges=chosen_challenges,
                timeout_seconds=settings.LIVENESS_SESSION_TIMEOUT_SEC,
            )
            self._sessions[session_id] = session

            challenge = session.get_current_challenge()
            assert challenge is not None

            return LivenessSessionCreateResponse(
                session_id=session_id,
                created_at=session.created_at,
                expires_in_seconds=int(session.expires_at - time.time()),
                total_steps=session.total_steps,
                current_step=1,
                challenge=challenge,
            )

    def get_session_status(self, session_id: str) -> LivenessSessionStatusResponse:
        """Retrieves current session state and progress."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise KeyError(f"Phiên kiểm tra không tồn tại hoặc đã hết hạn: {session_id}")

            if session.is_expired() and session.status == "IN_PROGRESS":
                session.status = "EXPIRED"

            return LivenessSessionStatusResponse(
                session_id=session.session_id,
                status=session.status,
                current_step=session.current_step_index + 1 if session.status == "IN_PROGRESS" else session.current_step_index,
                total_steps=session.total_steps,
                passed=(session.status == "PASSED"),
                created_at=session.created_at,
                expires_at=session.expires_at,
                history=session.history,
                disclaimer=settings.LIVENESS_DISCLAIMER,
            )

    @staticmethod
    def _compute_pose_metrics(landmarks: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        Computes scale-invariant Yaw and Pitch ratios from 5 facial landmarks:
        0: right eye, 1: left eye, 2: nose tip, 3: right mouth, 4: left mouth.
        """
        re_x, re_y = landmarks[0]
        le_x, le_y = landmarks[1]
        nt_x, nt_y = landmarks[2]
        rcm_x, rcm_y = landmarks[3]
        lcm_x, lcm_y = landmarks[4]

        # Horizontal yaw ratio (scale-normalized)
        # In OpenCV image coordinates:
        # re_x is right eye of subject (image left, smaller x)
        # le_x is left eye of subject (image right, larger x)
        min_eye_x = min(re_x, le_x)
        max_eye_x = max(re_x, le_x)
        inter_eye_dist = max(1, max_eye_x - min_eye_x)

        yaw_ratio = round(float((nt_x - min_eye_x) / inter_eye_dist), 4)

        # Vertical pitch ratio
        eye_mid_y = (re_y + le_y) / 2.0
        mouth_mid_y = (rcm_y + lcm_y) / 2.0
        vert_dist = max(1.0, mouth_mid_y - eye_mid_y)
        pitch_ratio = round(float((nt_y - eye_mid_y) / vert_dist), 4)

        return yaw_ratio, pitch_ratio

    def verify_step(
        self,
        session_id: str,
        frames: List[np.ndarray],
    ) -> LivenessStepVerifyResponse:
        """
        Validates an ordered sequence of webcam frames against the current active challenge.
        Requires continuous temporal motion and enforces biometric invariants.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return LivenessStepVerifyResponse(
                    session_id=session_id,
                    step_passed=False,
                    current_step=0,
                    total_steps=0,
                    all_passed=False,
                    status="FAILED",
                    reason_code="INVALID_SESSION",
                    message="Phiên kiểm tra không tồn tại hoặc đã bị xóa khỏi bộ nhớ.",
                )

            if session.is_expired():
                session.status = "EXPIRED"
                return LivenessStepVerifyResponse(
                    session_id=session_id,
                    step_passed=False,
                    current_step=session.current_step_index + 1,
                    total_steps=session.total_steps,
                    all_passed=False,
                    status="EXPIRED",
                    reason_code="SESSION_EXPIRED",
                    message="Phiên kiểm tra đã hết hạn thời gian. Vui lòng bắt đầu phiên mới.",
                )

            if session.status == "PASSED":
                return LivenessStepVerifyResponse(
                    session_id=session_id,
                    step_passed=False,
                    current_step=session.total_steps,
                    total_steps=session.total_steps,
                    all_passed=True,
                    status="PASSED",
                    reason_code="SESSION_ALREADY_COMPLETED",
                    message="Phiên kiểm tra này đã hoàn tất thành công trước đó (ngăn chặn hành vi phát lại replay).",
                )

            if session.status == "FAILED":
                return LivenessStepVerifyResponse(
                    session_id=session_id,
                    step_passed=False,
                    current_step=session.current_step_index + 1,
                    total_steps=session.total_steps,
                    all_passed=False,
                    status="FAILED",
                    reason_code="SESSION_ALREADY_FAILED",
                    message="Phiên kiểm tra đã thất bại do vượt quá số lần thử hoặc vi phạm điều kiện.",
                )

            current_step_num = session.current_step_index + 1
            expected_action = session.challenges[session.current_step_index]

            # 1. Enforce frame sequence count boundaries
            if len(frames) < settings.LIVENESS_MIN_FRAMES:
                return self._handle_step_failure(
                    session,
                    "INSUFFICIENT_FRAMES",
                    f"Số lượng khung hình không đủ (nhận {len(frames)} < {settings.LIVENESS_MIN_FRAMES} khung hình).",
                )

            if len(frames) > settings.LIVENESS_MAX_FRAMES:
                return self._handle_step_failure(
                    session,
                    "EXCESSIVE_FRAMES",
                    f"Số lượng khung hình vượt quá giới hạn cho phép ({len(frames)} > {settings.LIVENESS_MAX_FRAMES}).",
                )

            detector = self._get_detector()

            # 2. Analyze each frame sequentially
            yaw_ratios: List[float] = []
            pitch_ratios: List[float] = []
            box_centers: List[Tuple[float, float]] = []

            for idx, frame in enumerate(frames):
                if frame is None or frame.size == 0:
                    return self._handle_step_failure(
                        session,
                        "CORRUPTED_FRAME",
                        f"Khung hình thứ {idx + 1} bị hỏng hoặc dữ liệu rỗng.",
                    )

                img_h, img_w = frame.shape[:2]
                detected_faces = detector.detect_face_and_landmarks(frame)

                # Strict requirement: Exactly ONE face in every frame
                if len(detected_faces) == 0:
                    return self._handle_step_failure(
                        session,
                        "NO_FACE_DETECTED",
                        f"Không tìm thấy khuôn mặt trong khung hình thứ {idx + 1}. Vui lòng căn chỉnh mặt vào giữa camera.",
                    )

                if len(detected_faces) > 1:
                    return self._handle_step_failure(
                        session,
                        "MULTIPLE_FACES_DETECTED",
                        f"Phát hiện nhiều khuôn mặt ({len(detected_faces)} khuôn mặt) trong khung hình thứ {idx + 1}. Yêu cầu chỉ 1 người.",
                    )

                face = detected_faces[0]
                box = face["box"]
                bw, bh = box[2], box[3]

                # Minimal quality size check
                if bw < settings.FACE_SERIOUS_MIN_SIZE or bh < settings.FACE_SERIOUS_MIN_SIZE:
                    return self._handle_step_failure(
                        session,
                        "FACE_TOO_SMALL",
                        f"Khuôn mặt trong khung hình thứ {idx + 1} quá nhỏ ({bw}x{bh}px). Hãy tiến lại gần camera hơn.",
                    )

                landmarks = face["landmarks"]
                if len(landmarks) < 5:
                    return self._handle_step_failure(
                        session,
                        "INCOMPLETE_LANDMARKS",
                        f"Không đủ 5 điểm mốc khuôn mặt trong khung hình thứ {idx + 1}.",
                    )

                # Dedicated face-framing, position, and boundary clipping validation
                framing_result = FaceFramingValidator.validate_frame(box, landmarks, img_w, img_h)
                if not framing_result.is_valid:
                    return self._handle_step_failure(
                        session,
                        framing_result.reason_code or "FACE_OUT_OF_GUIDE",
                        f"Khung hình thứ {idx + 1}: {framing_result.message}",
                        metrics=framing_result.details,
                    )

                # Dedicated Passive Face Anti-Spoofing (PAD) check
                if settings.ANTI_SPOOFING_ENABLED:
                    pad_detector = self._get_anti_spoofing_detector()
                    pad_result = pad_detector.predict(frame, (box[0], box[1], bw, bh))

                    if pad_result.outcome == AntiSpoofingOutcome.ERROR:
                        if settings.ANTI_SPOOFING_FAIL_CLOSED:
                            return self._handle_step_failure(
                                session,
                                "ANTI_SPOOFING_ERROR",
                                f"Khung hình thứ {idx + 1}: {pad_result.error_message or 'Lỗi suy luận kiểm tra chống giả mạo'}. Hệ thống từ chối xác thực vì an toàn.",
                                metrics=pad_result.details,
                                anti_spoofing_outcome=pad_result.outcome.value,
                            )

                    elif pad_result.outcome == AntiSpoofingOutcome.SPOOF:
                        spoof_type = (
                            "màn hình thiết bị khác (video replay)"
                            if pad_result.score_screen >= pad_result.score_paper
                            else "ảnh in trên giấy"
                        )
                        return self._handle_step_failure(
                            session,
                            "SPOOF_DETECTED",
                            f"Phát hiện dấu hiệu giả mạo từ {spoof_type} trong khung hình thứ {idx + 1} (Độ tin cậy: {pad_result.confidence * 100:.1f}%). Vui lòng sử dụng khuôn mặt thật trực tiếp trước camera.",
                            metrics=pad_result.details,
                            anti_spoofing_outcome=pad_result.outcome.value,
                        )

                    elif pad_result.outcome == AntiSpoofingOutcome.UNCERTAIN:
                        return self._handle_step_failure(
                            session,
                            "UNCERTAIN_LIVENESS",
                            f"Không thể xác định chắc chắn thực thể sống trong khung hình thứ {idx + 1}. Vui lòng tránh ánh sáng chói, bóng phản chiếu màn hình và thử lại.",
                            metrics=pad_result.details,
                            anti_spoofing_outcome=pad_result.outcome.value,
                        )

                yaw, pitch = self._compute_pose_metrics(landmarks)
                yaw_ratios.append(yaw)
                pitch_ratios.append(pitch)

                center_x = (box[0] + bw / 2.0) / img_w
                center_y = (box[1] + bh / 2.0) / img_h
                box_centers.append((center_x, center_y))

            # 3. Check for Static Image / Photo Presentation Attack
            # A genuine live person executing a head turn must exhibit variance across frames
            yaw_min = min(yaw_ratios)
            yaw_max = max(yaw_ratios)
            motion_delta = round(yaw_max - yaw_min, 4)

            if motion_delta < settings.LIVENESS_MIN_MOTION_VARIANCE:
                return self._handle_step_failure(
                    session,
                    "STATIC_IMAGE_DETECTED",
                    f"Phát hiện khung hình tĩnh (biến thiên chuyển động {motion_delta} < {settings.LIVENESS_MIN_MOTION_VARIANCE}). "
                    "Vui lòng thực hiện chuyển động thật trước camera thay vì dùng ảnh tĩnh.",
                    metrics={"motion_delta": motion_delta, "yaw_ratios": yaw_ratios},
                )

            # 4. Check for Discontinuous Jumping / Teleportation
            for i in range(1, len(box_centers)):
                dx = abs(box_centers[i][0] - box_centers[i - 1][0])
                dy = abs(box_centers[i][1] - box_centers[i - 1][1])
                if dx > 0.40 or dy > 0.40:
                    return self._handle_step_failure(
                        session,
                        "DISCONTINUOUS_MOTION",
                        "Phát hiện chuyển động gián đoạn bất thường hoặc đổi ảnh đột ngột.",
                    )

            # 5. Evaluate the Specific Active Challenge Action
            start_yaw = yaw_ratios[0]
            step_passed = False
            fail_reason = ""

            if expected_action == ChallengeAction.TURN_HEAD_LEFT:
                # Turning head to subject's LEFT -> nose moves towards subject's left eye (higher x)
                # Yaw ratio increases towards >= LIVENESS_YAW_TURN_LEFT_MIN (0.65)
                # Starting position should be reasonably centered (< 0.62)
                reached_target = yaw_max >= settings.LIVENESS_YAW_TURN_LEFT_MIN
                directed_motion = (yaw_max - start_yaw) >= 0.06

                if reached_target and directed_motion:
                    step_passed = True
                else:
                    # Check if user turned right instead
                    if yaw_min <= settings.LIVENESS_YAW_TURN_RIGHT_MAX:
                        fail_reason = "Bạn đã quay đầu sang bên PHẢI trong khi yêu cầu là quay sang bên TRÁI."
                    else:
                        fail_reason = f"Chưa quay đầu đủ biên độ sang trái (đỉnh quay {yaw_max} < {settings.LIVENESS_YAW_TURN_LEFT_MIN})."

            elif expected_action == ChallengeAction.TURN_HEAD_RIGHT:
                # Turning head to subject's RIGHT -> nose moves towards subject's right eye (lower x)
                # Yaw ratio decreases towards <= LIVENESS_YAW_TURN_RIGHT_MAX (0.35)
                reached_target = yaw_min <= settings.LIVENESS_YAW_TURN_RIGHT_MAX
                directed_motion = (start_yaw - yaw_min) >= 0.06

                if reached_target and directed_motion:
                    step_passed = True
                else:
                    if yaw_max >= settings.LIVENESS_YAW_TURN_LEFT_MIN:
                        fail_reason = "Bạn đã quay đầu sang bên TRÁI trong khi yêu cầu là quay sang bên PHẢI."
                    else:
                        fail_reason = f"Chưa quay đầu đủ biên độ sang phải (đỉnh quay {yaw_min} > {settings.LIVENESS_YAW_TURN_RIGHT_MAX})."

            elif expected_action == ChallengeAction.NOD_HEAD:
                pitch_min = min(pitch_ratios)
                pitch_max = max(pitch_ratios)
                pitch_delta = pitch_max - pitch_min
                if pitch_delta >= 0.10:
                    step_passed = True
                else:
                    fail_reason = "Chưa phát hiện cử động gật đầu rõ nét."

            elif expected_action == ChallengeAction.FACE_CENTER:
                if 0.42 <= start_yaw <= 0.58 and motion_delta < 0.15:
                    step_passed = True
                else:
                    fail_reason = "Vui lòng giữ khuôn mặt nhìn thẳng vào giữa camera."

            metrics = {
                "motion_delta": motion_delta,
                "yaw_ratios": yaw_ratios,
                "pitch_ratios": pitch_ratios,
                "start_yaw": start_yaw,
                "yaw_min": yaw_min,
                "yaw_max": yaw_max,
            }

            if not step_passed:
                return self._handle_step_failure(
                    session,
                    "CHALLENGE_MISMATCH",
                    fail_reason or "Cử động không khớp với thách thức yêu cầu.",
                    metrics=metrics,
                )

            # --- STEP PASSED ---
            session.history.append({
                "step": current_step_num,
                "action": expected_action.value,
                "passed": True,
                "metrics": metrics,
                "timestamp": time.time(),
            })
            session.current_step_index += 1
            session.step_attempts = 0  # reset attempts for next step

            # Check if all steps in the session have been completed
            if session.current_step_index >= session.total_steps:
                session.status = "PASSED"
                return LivenessStepVerifyResponse(
                    session_id=session.session_id,
                    step_passed=True,
                    current_step=current_step_num,
                    total_steps=session.total_steps,
                    all_passed=True,
                    status="PASSED",
                    next_challenge=None,
                    reason_code="LIVENESS_VERIFIED",
                    message="Xác minh thực thể sống thành công! Người dùng đã hoàn thành toàn bộ chuỗi thử thách.",
                    metrics=metrics,
                    anti_spoofing_outcome=AntiSpoofingOutcome.LIVE.value,
                )
            else:
                next_challenge = session.get_current_challenge()
                return LivenessStepVerifyResponse(
                    session_id=session.session_id,
                    step_passed=True,
                    current_step=current_step_num,
                    total_steps=session.total_steps,
                    all_passed=False,
                    status="IN_PROGRESS",
                    next_challenge=next_challenge,
                    reason_code="STEP_COMPLETED",
                    message=f"Hoàn thành bước {current_step_num}/{session.total_steps}. Vui lòng thực hiện bước tiếp theo.",
                    metrics=metrics,
                    anti_spoofing_outcome=AntiSpoofingOutcome.LIVE.value,
                )

    def _handle_step_failure(
        self,
        session: LivenessSessionState,
        reason_code: str,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        anti_spoofing_outcome: Optional[str] = None,
    ) -> LivenessStepVerifyResponse:
        """Records failure attempt and determines whether session should transition to FAILED."""
        session.step_attempts += 1
        current_step_num = session.current_step_index + 1

        session.history.append({
            "step": current_step_num,
            "action": session.challenges[session.current_step_index].value,
            "passed": False,
            "reason_code": reason_code,
            "message": message,
            "metrics": metrics or {},
            "attempt": session.step_attempts,
            "timestamp": time.time(),
            "anti_spoofing_outcome": anti_spoofing_outcome,
        })

        if session.step_attempts >= settings.LIVENESS_MAX_STEP_ATTEMPTS:
            session.status = "FAILED"
            return LivenessStepVerifyResponse(
                session_id=session.session_id,
                step_passed=False,
                current_step=current_step_num,
                total_steps=session.total_steps,
                all_passed=False,
                status="FAILED",
                next_challenge=None,
                reason_code="MAX_ATTEMPTS_EXCEEDED",
                message=f"Đã vượt quá số lần thử tối đa ({settings.LIVENESS_MAX_STEP_ATTEMPTS} lần) cho bước này. Phiên bị hủy.",
                metrics=metrics,
                anti_spoofing_outcome=anti_spoofing_outcome,
            )

        return LivenessStepVerifyResponse(
            session_id=session.session_id,
            step_passed=False,
            current_step=current_step_num,
            total_steps=session.total_steps,
            all_passed=False,
            status="IN_PROGRESS",
            next_challenge=session.get_current_challenge(),
            reason_code=reason_code,
            message=f"{message} (Lần thử {session.step_attempts}/{settings.LIVENESS_MAX_STEP_ATTEMPTS}).",
            metrics=metrics,
            anti_spoofing_outcome=anti_spoofing_outcome,
        )


# Global singleton instance
_liveness_service: Optional[LivenessService] = None


def get_liveness_service() -> LivenessService:
    global _liveness_service
    if _liveness_service is None:
        _liveness_service = LivenessService()
    return _liveness_service


def set_liveness_service(service: Optional[LivenessService]):
    """Testing utility to swap or reset liveness service."""
    global _liveness_service
    _liveness_service = service
