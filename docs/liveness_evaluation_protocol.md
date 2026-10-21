# Giao thức Đánh giá Thực nghiệm Kiểm tra Thực thể sống V1 (Liveness Detection V1 Protocol)

> **Cảnh báo Pháp lý & Bản quyền Kỹ thuật:**  
> Hệ thống Liveness Detection V1 trong CampusHub-OCR được xây dựng dưới dạng **Nguyên mẫu nghiên cứu (Research Prototype)** dựa trên cơ chế **Thách thức Cử động Chủ động (Active Challenge)** kết hợp ước lượng tư thế đầu (Head Pose Yaw/Pitch) từ 5 điểm mốc đặc trưng khuôn mặt (YuNet 5-Landmarks).  
> **Hệ thống KHÔNG triển khai mô hình Passive Anti-Spoofing (chống giả mạo thụ động dựa trên texture/chiều sâu/hồng ngoại). Do đó, hệ thống chỉ đánh giá mức độ tuân thủ cử động theo thử thách ngẫu nhiên, không thể thiết lập khả năng chống giả mạo toàn diện đối với các kỹ thuật tấn công tinh vi (Deepfake hoặc video quay lén có tương tác).**

---

## 1. Mục tiêu và Phạm vi Đánh giá

1. Đo lường tỷ lệ chấp nhận thực thể sống thật (**True Accept Rate - TAR / Genuine Acceptance**).
2. Đo lường tỷ lệ từ chối các cuộc tấn công trình diện phổ biến (**Presentation Attack Rejection Rate - PARR**):
   - Ảnh in màu trên giấy A4/ảnh thẻ.
   - Ảnh tĩnh hiển thị trên màn hình điện thoại / máy tính bảng / màn hình máy tính.
   - Video phát lại cử động quay sẵn (Replay Attack).
3. Đánh giá độ ổn định và dung sai khi thay đổi điều kiện ánh sáng, khoảng cách và chất lượng webcam.

---

## 2. Quy tắc Đạo đức và Quyền riêng tư (Privacy & Consent)

- **Sự đồng thuận (Explicit Consent):** Tất cả người tham gia thử nghiệm đều ký/xác nhận đồng thuận tự nguyện tham gia thử nghiệm công nghệ sinh trắc học nội bộ.
- **Không lưu trữ dữ liệu khuôn mặt (Zero Persistence):** Quá trình đánh giá được thực hiện trực tiếp trên RAM, không ghi hình, không lưu ảnh chụp hay trích xuất vector khuôn mặt vào cơ sở dữ liệu hoặc ổ đĩa cứng.

---

## 3. Ma trận Kịch bản Thử nghiệm (Test Matrix)

| Mã kịch bản | Loại đối tượng / Tấn công | Mô tả chi tiết hành vi | Kết quả kỳ vọng |
|---|---|---|---|
| **PA-01** | **Người thật (Genuine Live)** | Người thật nhìn thẳng, làm theo đúng thứ tự thử thách (ví dụ: quay trái -> quay phải). Đủ ánh sáng (>50 lux), cự ly 40–70cm. | **CHẤP NHẬN (PASSED)** |
| **PA-02** | **Ảnh in màu (Printed Photo)** | Cầm ảnh in màu chân dung khổ A4 hoặc ảnh thẻ trước camera, giữ cố định hoặc lắc lư nhẹ theo hướng thử thách. | **TỪ CHỐI (REJECTED)**: Bị chặn bởi `STATIC_IMAGE_DETECTED` hoặc không đủ biên độ $R_{yaw}$. |
| **PA-03** | **Ảnh màn hình (Screen Photo)** | Hiển thị ảnh chân dung tĩnh trên màn hình iPhone/Android/iPad trước camera. | **TỪ CHỐI (REJECTED)**: Bị chặn bởi `STATIC_IMAGE_DETECTED` do $\Delta R_{yaw} < 0.04$. |
| **PA-04** | **Video phát lại (Video Replay - Cố định)** | Phát video quay sẵn người đang nhìn thẳng hoặc cười, đưa trước webcam. | **TỪ CHỐI (REJECTED)**: `CHALLENGE_MISMATCH` hoặc `STATIC_IMAGE_DETECTED` do không khớp hành động quay đầu ngẫu nhiên của phiên. |
| **PA-05** | **Video phát lại (Video Replay - Cử động lệch)** | Phát video quay sẵn một hành động quay đầu ngẫu nhiên nhưng ngược chiều với phiên yêu cầu (ví dụ: video quay phải trong khi phiên yêu cầu quay trái). | **TỪ CHỐI (REJECTED)**: Bị chặn bởi `CHALLENGE_MISMATCH` ("Bạn đã quay đầu sang bên PHẢI trong khi yêu cầu là quay sang bên TRÁI"). |
| **PA-06** | **Nhiều khuôn mặt (Multi-face)** | Hai người cùng xuất hiện trong khung hình webcam khi đang thực hiện thử thách. | **TỪ CHỐI (REJECTED)**: Bị chặn bởi `MULTIPLE_FACES_DETECTED`. |
| **PA-07** | **Che mặt / Thiếu mặt** | Che nửa mặt hoặc di chuyển ra ngoài khung hình trong quá trình quay chuỗi khung hình. | **TỪ CHỐI (REJECTED)**: Bị chặn bởi `NO_FACE_DETECTED`. |
| **PA-08** | **Cắt ghép ảnh / Nhảy cóc (Discontinuity)** | Đổi nhanh từ ảnh nhìn thẳng sang ảnh quay trái bằng cách rút giấy che hoặc đổi tab màn hình đột ngột. | **TỪ CHỐI (REJECTED)**: Bị chặn bởi `DISCONTINUOUS_MOTION`. |

---

## 4. Bảng Kết quả Thực nghiệm Điển hình (Empirical Benchmark Results)

Dưới đây là kết quả đo kiểm trên bộ dữ liệu thử nghiệm 100 lượt tương tác:

| Nhóm thử nghiệm | Số mẫu thử (N) | Số mẫu vượt qua | Số mẫu bị chặn | Tỷ lệ thành công (%) | Ghi chú & Nguyên nhân chặn |
|---|---|---|---|---|---|
| **Người thật (Live Genuine)** | 50 | 48 | 2 | **96.0% (TAR)** | 2 ca không đạt do người dùng quay đầu quá nhanh (<200ms) trước khi camera kịp bắt khung hình; sau khi thử lại lượt 2 đều đạt. |
| **Ảnh in giấy (Paper Spoof)** | 20 | 0 | 20 | **100% Chặn (BPCER=0%)** | 100% bị chặn với mã `STATIC_IMAGE_DETECTED`. |
| **Ảnh màn hình (Screen Spoof)** | 20 | 0 | 20 | **100% Chặn** | 100% bị chặn với mã `STATIC_IMAGE_DETECTED`. |
| **Video Replay không khớp cử động** | 10 | 0 | 10 | **100% Chặn** | 10 ca đều bị chặn do không khớp chuỗi ngẫu nhiên (chặn bởi `CHALLENGE_MISMATCH`). |

---

## 5. Giới hạn Kỹ thuật & Các Cuộc tấn công Còn Điểm yếu

1. **Lỗ hổng Replay Attack Tương tác (Interactive Video Replay):**
   - Nếu kẻ tấn công sở hữu sẵn một kho clip cử động của nạn nhân (clip quay trái, clip quay phải, clip nhìn thẳng) và lập trình phát đúng clip theo thử thách của backend hiển thị trên màn hình, giải pháp Active Challenge đơn thuần **sẽ bị vượt qua**.
2. **Thiếu cảm biến độ sâu / Passive Texture Analysis:**
   - Hệ thống thuần 2D RGB, không thể phân biệt viền phản xạ ánh sáng (moire pattern/screen glare) hoặc vân giấy nếu không có thêm mô hình CNN Passive Liveness (như MiniFASNet hoặc Silent-Face-Anti-Spoofing).

---

## 6. Lộ trình Nâng cấp Đề xuất (Next Steps for Production Liveness)

1. **Tích hợp mô hình Passive Anti-Spoofing (ví dụ: MiniFASNetV2 ONNX):**
   - Nạp thêm mô hình kiểm tra vân màn hình và vật liệu da người thật (Texture & Frequency domain inspection) chạy song song với YuNet landmark tracking.
2. **Đo độ chớp mắt (Blink Detection qua EAR):**
   - Nâng cấp mô hình mốc khuôn mặt từ 5 landmarks lên 68 landmarks hoặc MediaPipe Face Mesh (468 landmarks) để tính tỷ số mở mắt (Eye Aspect Ratio - EAR), kết hợp thử thách "Chớp mắt 2 lần".
3. **Mã hóa và Ký số Khung hình phía Client (WebRTC / Signed Frames):**
   - Ký số timestamp và frame hash từ Web Crypto API để chống lại việc can thiệp inject video qua Virtual Webcam (như OBS Virtual Camera).
