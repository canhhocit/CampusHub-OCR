from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChallengeAction(str, Enum):
    TURN_HEAD_LEFT = "TURN_HEAD_LEFT"
    TURN_HEAD_RIGHT = "TURN_HEAD_RIGHT"
    NOD_HEAD = "NOD_HEAD"
    FACE_CENTER = "FACE_CENTER"


class ChallengeInfo(BaseModel):
    action: ChallengeAction = Field(..., description="Mã hành động yêu cầu (TURN_HEAD_LEFT, TURN_HEAD_RIGHT, NOD_HEAD)")
    instruction: str = Field(..., description="Hướng dẫn chi tiết hiển thị cho người dùng")
    step: int = Field(..., description="Thứ tự bước hiện tại (1-indexed)")
    total_steps: int = Field(..., description="Tổng số bước trong phiên kiểm tra")
    timeout_seconds: int = Field(default=30, description="Thời gian tối đa để hoàn thành bước này")


class LivenessSessionCreateResponse(BaseModel):
    session_id: str = Field(..., description="Mã phiên kiểm tra ngẫu nhiên và bảo mật")
    created_at: float = Field(..., description="Timestamp khởi tạo phiên (Unix timestamp)")
    expires_in_seconds: int = Field(..., description="Thời gian hiệu lực còn lại của phiên (giây)")
    total_steps: int = Field(..., description="Tổng số bước thử thách")
    current_step: int = Field(..., description="Bước thử thách hiện tại (1-indexed)")
    challenge: ChallengeInfo = Field(..., description="Thông tin thử thách hiện tại cần thực hiện")


class AntiSpoofingOutcome(str, Enum):
    LIVE = "LIVE"
    SPOOF = "SPOOF"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"


class FrameMetric(BaseModel):
    frame_index: int
    face_detected: bool
    face_count: int
    yaw_ratio: Optional[float] = None
    pitch_ratio: Optional[float] = None
    motion_delta: Optional[float] = None
    framing_valid: Optional[bool] = None
    framing_reason: Optional[str] = None
    anti_spoofing_outcome: Optional[str] = None
    anti_spoofing_score: Optional[float] = None


class LivenessStepVerifyResponse(BaseModel):
    session_id: str = Field(..., description="Mã phiên kiểm tra")
    step_passed: bool = Field(..., description="Trạng thái bước thử thách hiện tại có đạt hay không")
    current_step: int = Field(..., description="Bước hiện tại vừa kiểm tra")
    total_steps: int = Field(..., description="Tổng số bước trong phiên")
    all_passed: bool = Field(..., description="Toàn bộ quy trình kiểm tra đã hoàn thành thành công chưa")
    status: str = Field(..., description="Trạng thái phiên: IN_PROGRESS, PASSED, FAILED, EXPIRED")
    next_challenge: Optional[ChallengeInfo] = Field(None, description="Thử thách tiếp theo (nếu còn bước)")
    reason_code: Optional[str] = Field(None, description="Mã nguyên nhân nếu không đạt hoặc đạt bước")
    message: str = Field(..., description="Thông báo kết quả chi tiết bằng tiếng Việt")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Thông số kỹ thuật phân tích chuyển động")
    anti_spoofing_outcome: Optional[str] = Field(None, description="Kết quả phân tích chống giả mạo tĩnh: LIVE, SPOOF, UNCERTAIN, ERROR")


class LivenessSessionStatusResponse(BaseModel):
    session_id: str = Field(..., description="Mã phiên kiểm tra")
    status: str = Field(..., description="Trạng thái phiên (IN_PROGRESS, PASSED, FAILED, EXPIRED)")
    current_step: int = Field(..., description="Bước hiện tại")
    total_steps: int = Field(..., description="Tổng số bước")
    passed: bool = Field(..., description="Kết quả tổng thể của phiên")
    created_at: float = Field(..., description="Thời gian tạo")
    expires_at: float = Field(..., description="Thời gian hết hạn")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Lịch sử các bước thử thách")
    disclaimer: str = Field(..., description="Cảnh báo pháp lý và bản chất prototype thử nghiệm")


class LivenessHealthResponse(BaseModel):
    status: str = Field(..., description="Trạng thái dịch vụ: ok, degraded, unavailable")
    version: str = Field(..., description="Phiên bản API")
    active_liveness_ready: bool = Field(..., description="Mô hình YuNet phát hiện khuôn mặt và điểm mốc đã sẵn sàng")
    passive_anti_spoofing_ready: bool = Field(..., description="Cụm mô hình MiniFASNet chống giả mạo đã sẵn sàng")
    anti_spoofing_enabled: bool = Field(..., description="Cấu hình bật/tắt chống giả mạo thụ động")
    fail_closed: bool = Field(..., description="Cấu hình fail-closed khi thiếu model hoặc lỗi suy luận")
    thresholds: Dict[str, float] = Field(..., description="Các ngưỡng phán quyết phân loại và cử động")
    models: Dict[str, Any] = Field(..., description="Chi tiết trạng thái từng tệp mô hình ONNX")
    error_message: Optional[str] = Field(None, description="Chi tiết lỗi nếu dịch vụ chưa sẵn sàng")

