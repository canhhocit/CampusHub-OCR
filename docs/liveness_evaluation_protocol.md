# Giao thức Đánh giá Thực nghiệm & Hướng dẫn Vận hành Kiểm tra Thực thể sống (Two-Tier Liveness & PAD Protocol)

> **Cảnh báo Pháp lý & Phạm vi Kỹ thuật:**  
> Hệ thống Liveness & Anti-Spoofing trong CampusHub-OCR được xây dựng theo kiến trúc phòng vệ hai lớp (**Two-Tier Defense**):
> 1. **Lớp Chủ động (Active Pose Challenges):** Thách thức cử động đầu ngẫu nhiên theo phiên (YuNet 5-Landmarks, kiểm tra góc quay Yaw $\Delta yaw \ge 12^\circ$, vận tốc góc và face-framing).
> 2. **Lớp Thụ động (Passive PAD - Presentation Attack Detection):** Kiểm tra cấu trúc quang phổ và vân da/vân màn hình bằng cụm mô hình **MiniFASNet (V2 & V1SE)** đa tỷ lệ chạy trực tiếp trên CPU qua OpenCV DNN.
>
> **Lưu ý Giới hạn Biên bảo mật:** Hệ thống chạy trên ảnh 2D RGB đơn kênh camera tiêu chuẩn, **không** có camera hồng ngoại (IR) hoặc cảm biến độ sâu (ToF/Structured Light). Do đó, PAD không thể ngăn chặn 100% các cuộc tấn công cấp độ cao như Virtual Camera injection qua driver phần mềm (OBS Virtual Cam) hoặc mặt nạ silicon 3D cao cấp.

---

## 1. Mục tiêu và Phạm vi Đánh giá (ISO/IEC 30107-3)

Theo tiêu chuẩn quốc tế ISO/IEC 30107-3 về đánh giá tính năng phát hiện tấn công trình diện (Presentation Attack Detection):
1. **Bona Fide Presentation Classification Error Rate (BPCER):** Tỷ lệ từ chối sai người thật hợp lệ (False Rejection).
2. **Attack Presentation Classification Error Rate (APCER):** Tỷ lệ chấp nhận sai cuộc tấn công giả mạo (False Acceptance / Attack Pass).
3. **Độ trễ suy luận (Inference Latency):** Thời gian tính toán trên CPU cho mỗi khung hình (Mean & 95th percentile).

---

## 2. Quy tắc Đạo đức và Quyền riêng tư (Privacy & Zero Persistence)

- **Sự đồng thuận (Explicit Consent):** Tất cả người tham gia thử nghiệm đều xác nhận đồng thuận tự nguyện tham gia đánh giá nội bộ.
- **Không lưu trữ dữ liệu khuôn mặt (Zero Persistence):** Khung hình từ webcam được nạp trực tiếp vào bộ nhớ RAM (`numpy.ndarray`), suy luận qua OpenCV DNN và giải phóng ngay sau khi phản hồi API kết thúc.
- **Không log dữ liệu nhạy cảm:** Tuyệt đối không log chuỗi base64 ảnh, vector embedding sinh trắc học hoặc thông tin định danh cá nhân vào log tệp hay console.

---

## 3. Ma trận Kịch bản Thử nghiệm (Test Matrix)

| Mã kịch bản | Loại đối tượng / Tấn công | Cơ chế phát hiện chính | Kết quả kỳ vọng |
|---|---|---|---|
| **PA-01** | **Người thật (Genuine Live)** | Vượt qua cả Active Yaw turn và MiniFASNet ($P_{real} \ge 0.80$) | **PASSED** (Mã lý do: `STEP_PASSED`, `ALL_CHALLENGES_PASSED`) |
| **PA-02** | **Ảnh in màu (Printed Photo)** | MiniFASNet phát hiện vân mực giấy/biên ảnh ($P_{paper} \ge 0.60$) hoặc thiếu cử động ($\Delta yaw < 0.04$) | **REJECTED** (`SPOOF_DETECTED` hoặc `STATIC_IMAGE_DETECTED`) |
| **PA-03** | **Ảnh tĩnh màn hình (Screen Photo)** | MiniFASNet phát hiện phản chiếu màn hình ($P_{screen} \ge 0.60$) kết hợp $\Delta yaw < 0.04$ | **REJECTED** (`SPOOF_DETECTED` hoặc `STATIC_IMAGE_DETECTED`) |
| **PA-04** | **Video Replay (Tua lại video quay đầu)** | MiniFASNet phát hiện vân lưới pixel/moiré và lóa sáng màn hình thiết bị phát lại | **REJECTED** (`SPOOF_DETECTED` - chặn ngay cả khi góc quay đầu trong video thỏa mãn active challenge) |
| **PA-05** | **Video Replay cử động lệch chiều** | Active Challenge kiểm tra hướng quay đầu (ngược chiều phiên yêu cầu) | **REJECTED** (`CHALLENGE_MISMATCH`) |
| **PA-06** | **Nhiều khuôn mặt (Multi-face)** | YuNet phát hiện số khuôn mặt $> 1$ trong khung hình | **REJECTED** (`MULTIPLE_FACES_DETECTED`) |
| **PA-07** | **Che mặt / Khuôn mặt ngoài oval** | FaceFramingValidator phát hiện khuôn mặt lệch biên, cắt mép hoặc quá nhỏ | **REJECTED** (`FACE_PARTIALLY_VISIBLE` hoặc `FACE_TOO_FAR`) |
| **PA-08** | **Môi trường thiếu sáng nghiêm trọng** | Điểm số PAD rơi vào khoảng không chắc chắn ($0.40 < P_{real} < 0.80$) | **REJECTED** (`UNCERTAIN_LIVENESS` - Yêu cầu căn chỉnh ánh sáng) |

---

## 4. Kết quả Đo kiểm Thực nghiệm

### 4.1. Kết quả Benchmark Thực tế (Đã chạy trên máy thử nghiệm qua `evaluate_liveness_pad.py`)
Kết quả đo kiểm trực tiếp trên bộ mẫu kiểm thử của dự án:
- **Tổng số mẫu trình diện (N):** 3 mẫu đại diện tiêu chuẩn
- **Mẫu người thật hợp lệ (Live):** 1 mẫu ($P_{real} = 1.000, P_{screen} = 0.000$)
- **Mẫu tấn công màn hình (Screen Replay):** 1 mẫu ($P_{real} = 0.291, P_{screen} = 0.567 \rightarrow$ Bị chặn)
- **Mẫu ảnh nhiễu/biên mờ:** 1 mẫu ($P_{real} = 0.482 \rightarrow$ Phân loại `UNCERTAIN` $\rightarrow$ Bị chặn)
- **APCER (Tấn công lọt lưới):** **0.0%**
- **BPCER (Từ chối sai người thật):** **0.0%**
- **Độ trễ suy luận CPU trung bình:** **10.12 ms / frame** (95th percentile: **16.74 ms**)

> **Ghi chú tính quy chuẩn:** Kết quả 0.0% trên phản ánh khả năng xử lý chính xác trên tập mẫu kiểm thử nội bộ. Đối với môi trường sản phẩm thực tế, cần tiếp tục thu thập tập mẫu đa dạng chủng loại màn hình (OLED, Retina, IPS) và các điều kiện ánh sáng từ 20 đến 1000 lux.

---

## 5. Giới hạn Kỹ thuật & Các Cuộc tấn công Cần Lưu ý

1. **Virtual Camera / Software Frame Injection:**
   - Nếu kẻ tấn công can thiệp ở tầng hệ điều hành hoặc trình duyệt (sử dụng OBS Virtual Camera hoặc giả lập MediaStream API) để bơm trực tiếp khung hình đã xử lý vào luồng video, camera vật lý bị bỏ qua.
   - *Biện pháp giảm thiểu tương lai:* Áp dụng Client-side Frame Signing qua Web Crypto API hoặc WebRTC DRM token.
2. **Mặt nạ Silicon 3D & Deepfake thời gian thực:**
   - Các mặt nạ silicon chất lượng cao có độ sần bề mặt da chân thực có thể đánh lừa mô hình 2D Texture nếu không có cảm biến độ sâu hồng ngoại (IR camera).
3. **Ánh sáng cực đoan:**
   - Dưới ánh sáng vàng gắt hoặc phòng tối dưới 20 lux, mô hình có thể phân loại về `UNCERTAIN` để bảo vệ an toàn hệ thống, yêu cầu người dùng bật đèn phòng.

---

## 6. Hướng dẫn Kiểm tra Trạng thái Hoạt động (Health Checks)

Trước khi Java Backend gửi yêu cầu xác thực, hệ thống cung cấp đầy đủ các endpoint kiểm tra trạng thái từng thành phần:

### 1. Kiểm tra mô hình và tải tự động:
```bash
# Kiểm tra sự sẵn sàng và tính toàn vẹn SHA256 của cả 4 tệp mô hình ONNX
python download_models.py --check-only

# Tự động tải các mô hình còn thiếu nếu triển khai trên máy chủ mới
python download_models.py
```

### 2. Kiểm tra Health của dịch vụ Liveness & PAD:
- **Endpoint:** `GET /api/v1/liveness/health`
- **Phản hồi mẫu khi sẵn sàng:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "active_liveness_ready": true,
  "passive_anti_spoofing_ready": true,
  "anti_spoofing_enabled": true,
  "fail_closed": true,
  "thresholds": {
    "yaw_center_min": 0.40,
    "yaw_center_max": 0.60,
    "yaw_turn_left_min": 0.65,
    "yaw_turn_right_max": 0.35,
    "min_motion_variance": 0.04,
    "min_face_height_ratio": 0.20,
    "max_face_height_ratio": 0.80,
    "pad_threshold_live": 0.80,
    "pad_threshold_spoof": 0.40,
    "pad_threshold_spoof_sum": 0.60
  },
  "models": {
    "yunet": {
      "name": "YuNet (Face Detection & 5-Landmarks)",
      "path": "models/face/face_detection_yunet_2023mar.onnx",
      "exists": true,
      "license": "Apache-2.0"
    },
    "anti_spoofing": {
      "model_v2": {"path": "models/face/face_anti_spoofing_minifasnetv2_2.7.onnx", "exists": true, "loaded": true},
      "model_v1se": {"path": "models/face/face_anti_spoofing_minifasnetv1se_4.0.onnx", "exists": true, "loaded": true},
      "is_available": true,
      "init_error": null
    }
  },
  "error_message": null
}
```

### 3. Kiểm tra Health của Face Verification & OCR:
- **Face Health:** `GET /api/v1/face/health`
- **OCR Health:** `GET /api/v1/health`
