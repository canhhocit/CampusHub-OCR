# CampusHub-OCR — Vietnamese CCCD OCR Testing & Benchmark Service

Hệ thống dịch vụ độc lập phát triển, thử nghiệm và đánh giá chất lượng nhận dạng ký tự quang học (OCR) cho Căn cước công dân (CCCD) Việt Nam.

> **Lưu ý quan trọng**: Dự án này là công cụ thử nghiệm và đánh giá độc lập (Testing Benchmark). Dự án **KHÔNG** tích hợp trực tiếp vào CampusHub Backend hay Frontend trong giai đoạn này, và **KHÔNG** chứng minh tính xác thực hay giá trị pháp lý của giấy tờ nhân thân.

---

## 1. Công nghệ & Kiến trúc

- **Ngôn ngữ**: Python 3.11+
- **Web API & Testing Server**: FastAPI, Uvicorn, Jinja2, Pydantic v2
- **Mô hình OCR**: PaddleOCR 2.8.1 (chạy trên nền tảng PaddlePaddle 2.6.2 ổn định, hỗ trợ tiếng Việt)
- **Xử lý hình ảnh**: OpenCV (cv2), Pillow, NumPy
- **Testing & Benchmark**: Pytest, TestClient, Levenshtein Distance & CER (Character Error Rate)
- **Giao diện người dùng**: HTML5, Vanilla CSS, Vanilla JavaScript (nhẹ, không dùng framework frontend nặng)

---

## 2. Cấu trúc thư mục

```
CampusHub-OCR/
├── .gitignore                      # Loại trừ .venv, __pycache__, ảnh upload tạm thời
├── requirements.txt                # Danh sách thư viện và phiên bản chuẩn hóa
├── README.md                       # Tài liệu hướng dẫn cài đặt và vận hành
├── evaluate.py                     # Công cụ benchmark đo lường CER, Exact-Match và độ trễ
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app, static mount, middleware và CORS
│   ├── config.py                   # Cấu hình giới hạn file, kích thước, cổng và bảo mật
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py               # Endpoints: /api/v1/health, /api/v1/ocr, /
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── ocr.py                  # Pydantic schemas (OCRResponse, ExtractedField, RawDetection)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_processor.py      # Kiểm tra định dạng (Magic bytes), kích thước và tiền xử lý
│   │   ├── ocr_engine.py           # Bộ suy luận PaddleOCR & MockOCREngine phục vụ test
│   │   └── cccd_extractor.py       # Trích xuất trường dữ liệu CCCD có cấu trúc (mặt trước / sau)
│   ├── static/
│   │   ├── css/style.css           # Giao diện kiểm thử tối giản, trực quan
│   │   └── js/app.js               # Logic upload, preview, gọi API và hiển thị kết quả
│   └── templates/
│       └── index.html              # Màn hình thử nghiệm OCR đơn trang
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Cấu hình Pytest, TestClient và mock fixtures
    ├── test_api.py                 # Kiểm thử các endpoint API
    ├── test_validation.py          # Kiểm thử ràng buộc hình ảnh, kích thước, file hỏng
    ├── test_extraction.py          # Kiểm thử logic phân tích trường CCCD
    ├── test_evaluation.py          # Kiểm thử thuật toán CER và runner benchmark
    └── fixtures/
        └── ground_truth_sample.json # Bộ dữ liệu ground-truth giả lập mẫu (synthetic)
```

---

## 3. Cài đặt và Chuẩn bị môi trường

### Yêu cầu hệ thống:
- Python 3.11.x (đã cài đặt trên máy)
- Windows / Linux / macOS (64-bit)

### Bước 1: Khởi tạo Virtual Environment
Mở terminal tại thư mục `CampusHub-OCR`:
```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

### Bước 2: Cài đặt các gói phụ thuộc
```bash
pip install -r requirements.txt
```

---

## 4. Khởi chạy Ứng dụng Kiểm thử

Khởi chạy máy chủ FastAPI cục bộ (mặc định lắng nghe tại `127.0.0.1:8001`):
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Mở trình duyệt web và truy cập:
👉 **`http://127.0.0.1:8001/`**

### Các tính năng trên giao diện kiểm thử:
1. **Kéo thả / Chọn ảnh CCCD**: Hỗ trợ JPG, JPEG, PNG (dung lượng tối đa 10 MB, kích thước tối thiểu 100x100px).
2. **Xem trước (Preview)**: Hiển thị ngay ảnh đã chọn và thông số dung lượng.
3. **Lựa chọn mặt thẻ**: Tự động nhận diện (`AUTO`), Mặt trước (`FRONT`), hoặc Mặt sau (`BACK`).
4. **Tiền xử lý ảnh**: Tùy chọn cân bằng tương phản và khử nhiễu tự động.
5. **Bảng dữ liệu có cấu trúc**:
   - Số CCCD (12 số).
   - Họ và tên.
   - Ngày sinh (DD/MM/YYYY).
   - Giới tính (Nam/Nữ).
   - Quốc tịch (Việt Nam).
   - Quê quán.
   - Nơi thường trú.
   - Có giá trị đến.
   - Mặt sau: Đặc điểm nhân dạng, ngày cấp, cơ quan cấp, mã MRZ.
6. **Ký tự thô & Độ tin cậy**: Xem toàn bộ các text bounding box nhận diện được cùng điểm confidence score (0 - 100%).
7. **Đo lường thời gian**: Hiển thị thời gian trích xuất chính xác theo mili-giây (ms).

---

## 5. API Documentation

### 1. Kiểm tra trạng thái máy chủ (Health Check)
- **Method**: `GET`
- **URL**: `/api/v1/health`
- **Response**:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "ocr_engine": "PaddleOCR",
  "device": "CPU",
  "supported_formats": [".jpg", ".jpeg", ".png"]
}
```

### 2. Trích xuất OCR Căn cước công dân
- **Method**: `POST`
- **URL**: `/api/v1/ocr`
- **Content-Type**: `multipart/form-data`
- **Body Form**:
  - `file`: Tệp hình ảnh (bắt buộc)
  - `side`: `AUTO` | `FRONT` | `BACK` (mặc định `AUTO`)
  - `preprocess`: `true` | `false` (mặc định `true`)
- **Response**:
```json
{
  "success": true,
  "side": "FRONT",
  "processing_time_ms": 68.5,
  "fields": {
    "id_number": {
      "name": "id_number",
      "label": "Số định danh cá nhân / CCCD",
      "value": "001200000001",
      "status": "VALID",
      "confidence": 0.992,
      "raw_text": "Số / No.: 001200000001"
    },
    "full_name": {
      "name": "full_name",
      "label": "Họ và tên",
      "value": "NGUYỄN VĂN AN",
      "status": "VALID",
      "confidence": 0.985,
      "raw_text": "NGUYỄN VĂN AN"
    }
  },
  "raw_detections": [
    {"text": "CĂN CƯỚC CÔNG DÂN", "confidence": 0.99, "box": [[10, 20], [200, 20], [200, 50], [10, 50]]}
  ],
  "image_info": {
    "width": 800,
    "height": 500,
    "channels": 3,
    "size_bytes": 125430,
    "format": "JPEG"
  },
  "warnings": [],
  "disclaimer": "Kết quả OCR mang tính chất thử nghiệm và trích xuất dữ liệu tự động..."
}
```

---

## 6. Kiểm thử tự động (Automated Testing)

Chạy toàn bộ 20 kịch bản kiểm thử tự động với Pytest:
```bash
pytest -v
```

### Các nhóm bài test được bao phủ:
- **`test_validation.py`**:
  - Tệp rỗng (0 bytes) -> Báo lỗi 400.
  - Định dạng không hỗ trợ (.pdf, .txt) -> Báo lỗi 415.
  - Tệp giả mạo phần mở rộng (fake header magic bytes) -> Báo lỗi 400.
  - Ảnh hỏng / giải mã thất bại -> Báo lỗi 400.
  - Kích thước quá nhỏ (<100px) hoặc quá lớn (>6000px) -> Báo lỗi 400.
  - Kiểm tra tiền xử lý ảnh (upscale ảnh nhỏ, giữ tỷ lệ).
- **`test_extraction.py`**:
  - Trích xuất đầy đủ 8 trường mặt trước CCCD.
  - Trích xuất 4 trường mặt sau (nhân dạng, ngày cấp dạng 'ngày DD tháng MM năm YYYY', cơ quan cấp, mã MRZ).
  - Tự động phân biệt mặt trước và mặt sau dựa trên từ khóa.
  - Bảo đảm không bịa đặt dữ liệu (các trường bị thiếu trả về `status: NOT_DETECTED` và `value: None`).
- **`test_api.py`**:
  - Endpoint `/health` và trang giao diện `/`.
  - Endpoint `/ocr` với Mock OCR engine.
- **`test_evaluation.py`**:
  - Tính toán khoảng cách Levenshtein và CER.
  - Chạy runner đánh giá trên dataset mẫu.

---

## 7. Quy trình Đánh giá Benchmark (Evaluation Procedure)

Để đánh giá định lượng độ chính xác OCR và khả năng trích xuất trường dữ liệu một cách lặp lại (repeatable):

```bash
# 1. Chạy đánh giá nhanh cho toàn bộ dữ liệu mẫu giả lập (Synthetic heuristic benchmark, < 1s)
python evaluate.py --mode synthetic

# 2. Chạy đánh giá end-to-end trên ảnh thẻ mẫu (Image mode với PaddleOCR)
python evaluate.py --mode image

# 3. So sánh hiệu năng và độ chính xác khi BẬT vs TẮT tiền xử lý ảnh (Preprocessing)
python evaluate.py --mode image --compare-preprocessing

# 4. Chạy toàn bộ và xuất báo cáo kết quả ra tệp JSON
python evaluate.py --mode all --output-json tests/fixtures/sample_report.json
```

### Các chỉ số định lượng đo lường:
- **Field-level Exact-Match (EM)**: Tỷ lệ trường trích xuất chính xác 100% so với ground-truth chuẩn.
- **Overall Exact-Match**: Tỷ lệ chính xác tổng thể trên toàn bộ các trường đánh giá.
- **Missing-Field Rate**: Tỷ lệ trường dữ liệu bị bỏ sót hoặc không trích xuất được (`MISSING`).
- **Invalid-Field Rate**: Tỷ lệ trường dữ liệu trích xuất sai quy chuẩn định dạng hoặc lịch (`INVALID`).
- **Character Error Rate (CER)**: Tỷ lệ sai số ký tự theo khoảng cách Levenshtein (tính theo từng trường và theo toàn bộ nội dung văn bản thẻ).
- **Latency Breakdown**:
  - Thời gian khởi tạo nạp trọng số mô hình OCR (Model Initialization / Cold Start).
  - Độ trễ suy luận OCR trên từng ảnh (PaddleOCR Inference Time: Mean, Median, Min, Max).
  - Độ trễ trích xuất trường dữ liệu theo quy tắc không gian (Heuristic Extraction Time: Mean, Median).
- **Phân loại nguyên nhân lỗi (Error Cause Categorization)**: Tự động phân loại sai số thành `MISSING_IN_OCR`, `OCR_RECOGNITION_ERROR`, `EXTRACTION_LABEL_MISMATCH`, `VALIDATION_FAILURE`, hoặc `NORMALIZATION_MISMATCH`.
- **Tài liệu Schema Dataset**: Tham khảo quy cách chi tiết tại [`tests/fixtures/SCHEMA.md`](file:///d:/code/DA2/sourcecode/CampusHub-OCR/tests/fixtures/SCHEMA.md).

> **Quy định đạo đức dữ liệu**: Bộ benchmark trong repository sử dụng 100% dữ liệu nhân thân giả lập (synthetic) và thẻ thử nghiệm mẫu (specimen). Tuyệt đối không đưa ảnh CCCD thật hoặc dữ liệu cá nhân của người dùng vào mã nguồn Git.

---

---

## 8. Module So khớp Khuôn mặt Cô lập (Isolated 1:1 Face Verification)

Dự án tích hợp một module PoC độc lập phục vụ đối sánh sinh trắc học 1:1 giữa ảnh chân dung in trên mặt trước CCCD và ảnh selfie từ webcam:

### 8.1. Mô hình và Bản quyền (Models & Licensing)
- **Face Detector**: **YuNet** (`face_detection_yunet_2023mar.onnx`, ~232 KB)
  - Nguồn tải tin cậy: [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  - Bản quyền: **Apache-2.0**
- **Face Recognizer**: **SFace** (`face_recognition_sface_2021dec.onnx`, ~38.6 MB)
  - Nguồn tải tin cậy: [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
  - Bản quyền: **Apache-2.0**
- **Thư mục lưu trữ**: `models/face/` (đã được cấu hình trong `.gitignore` để không commit weights nhị phân lên repository).

### 8.2. Hướng dẫn Cài đặt trên Windows (PowerShell Setup)
Chạy các lệnh PowerShell sau để chuẩn bị thư mục và tải mô hình ONNX chính thức:

```powershell
# 1. Tạo thư mục chứa mô hình nếu chưa có
New-Item -ItemType Directory -Force -Path "models/face"

# 2. Tải YuNet Face Detector (~230 KB)
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models/face/face_detection_yunet_2023mar.onnx"

# 3. Tải SFace Face Recognizer (~37 MB)
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" -OutFile "models/face/face_recognition_sface_2021dec.onnx"
```

### 8.3. Kiểm tra Trạng thái Hoạt động (Preflight / Health Check)
Kiểm tra trạng thái nạp mô hình thực tế qua endpoint `/api/v1/face/health`:
- **Khi mô hình ONNX đã sẵn sàng**: Trả về `status="ok"`, `models_loaded=true`, `engine_type="real"`.
- **Khi thiếu file mô hình**: Trả về `status="unavailable"`, `models_loaded=false`, `engine_type="unavailable"`. Hệ thống **tuyệt đối không âm thầm chuyển sang Mock** trong môi trường runtime. Gọi endpoint `/api/v1/face/verify` lúc này sẽ trả về lỗi an toàn **HTTP 503 Service Unavailable**.

### 8.4. Ngữ nghĩa Điểm số & Chất lượng Ảnh
- **`similarity_score`**: Giá trị Cosine Similarity thô giữa 2 vector đặc trưng 128 chiều trích xuất từ SFace, nằm trong khoảng $[-1.0, 1.0]$. Ngưỡng quyết định mặc định là `0.363` (theo khuyến nghị baseline LFW của OpenCV Zoo, có thể tùy chỉnh qua tham số `threshold`).
- **`confidence_score`**: Điểm hiển thị trực quan (heuristic display score) được chuẩn hóa từ cosine similarity quanh ngưỡng quyết định $[0.0 - 1.0]$ phục vụ giao diện người dùng. **Lưu ý**: Đây không phải là xác suất thống kê hậu nghiệm đã chuẩn hóa (calibrated posterior probability).
- **Quy tắc Kiểm tra Chất lượng (Quality Checks)**:
  - *Từ chối ngay (Rejection - `LOW_QUALITY_FACE`)*: Kích thước khuôn mặt $< 50$ px, độ mờ nét nghiêm trọng (Laplacian $< 15.0$), quá tối ($< 20/255$) hoặc cháy sáng ($> 240/255$).
  - *Cảnh báo nhẹ (Warning)*: Kích thước $50 - 60$ px, hơi mờ nét ($15 - 35$), thiếu sáng nhẹ ($20 - 40$), lóa sáng nhẹ ($225 - 240$), tỷ lệ khung hình bất thường ($< 0.5$ hoặc $> 1.5$).

### 8.5. Đánh giá & Benchmark
- **Chế độ Smoke Test Giả lập (`--synthetic`)**:
  ```bash
  python evaluate_face.py --synthetic
  ```
  *Lưu ý rõ ràng*: Chế độ này sinh dữ liệu giả lập để kiểm tra thông suốt luồng kỹ thuật, tính toán ma trận và xử lý ngưỡng; **không đại diện cho độ chính xác sinh trắc học thực tế**.
- **Chế độ Đánh giá Ảnh Thật (`--manifest`)**:
  ```bash
  python evaluate_face.py --manifest path/to/manifest.json --out face_benchmark.json
  ```
  Yêu cầu bắt buộc mô hình YuNet + SFace thực tế đang hoạt động và kiểm tra tính hợp lệ của đường dẫn ảnh, nhãn `is_same_person` để báo cáo chỉ số FAR (False Acceptance Rate) và FRR (False Rejection Rate) theo từng ngưỡng.

---

## 9. Module Chống Giả Mạo & Kiểm Tra Thực Thể (Liveness & Passive Anti-Spoofing / PAD)

Dự án tích hợp cơ chế phát hiện thực thể kết hợp 2 lớp độc lập (Two-Tier Liveness & Presentation Attack Detection):

### 9.1. Kiến trúc Hai Lớp (Two-Tier Defense)
1. **Lớp Chủ động (Active Pose Challenges)**:
   - Sinh phiên ngẫu nhiên (`/api/v1/liveness/session/create`) yêu cầu quay đầu trái / phải (`TURN_LEFT`, `TURN_RIGHT`) hoặc chớp mắt.
   - Sử dụng YuNet trích xuất 5 điểm mốc đặc trưng (Landmarks) để tính toán góc lệch Yaw ($\Delta yaw \ge 12^\circ$) và kiểm tra tỷ lệ khung hình khuôn mặt nằm gọn trong hình oval hướng dẫn.
2. **Lớp Thụ động (Passive PAD - Presentation Attack Detection)**:
   - Ngăn chặn triệt để tấn công phát lại video (video replay attack), in ảnh màu (print attack), và chiếu qua màn hình thiết bị (screen replay).
   - Tích hợp cụm mô hình **MiniFASNet (V2 & V1SE)** đa tỷ lệ (Multi-scale crop: scale 2.7 và scale 4.0 context).
   - Bản quyền mô hình: **Apache-2.0** ([minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)).
   - Trọng số ONNX lưu trữ tại:
     - `models/face/face_anti_spoofing_minifasnetv2_2.7.onnx` (~1.74 MB)
     - `models/face/face_anti_spoofing_minifasnetv1se_4.0.onnx` (~1.74 MB)
   - Chạy trực tiếp qua bộ suy luận **OpenCV DNN** trên CPU (~2-10ms / frame), không yêu cầu GPU hay runtime cồng kềnh.

### 9.2. Nguyên tắc An toàn & Fail-Closed
- Kết quả phân loại trả về tường minh: `LIVE`, `SPOOF`, `UNCERTAIN`, hoặc `ERROR`.
- Thiết lập cơ chế **Fail-Closed**: Khi thiếu file mô hình hoặc lỗi suy luận, hệ thống từ chối xác thực với mã lỗi `ANTI_SPOOFING_ERROR`, tuyệt đối không âm thầm cho qua (`fail-open`).
- Quay đầu thành công **không thể** vượt qua nếu bộ phát hiện thụ động đánh giá là `SPOOF`.

### 9.3. Benchmark Đánh giá PAD (ISO/IEC 30107-3)
Chạy công cụ benchmark đo lường APCER (Attack Presentation Classification Error Rate) và BPCER (Bona Fide Presentation Classification Error Rate):
```bash
python evaluate_liveness_pad.py
```

---

## 10. Nguyên tắc Bảo mật & Giới hạn Phạm vi

1. **Bảo mật dữ liệu (Data Privacy)**:
   - Dịch vụ chỉ mở trên `127.0.0.1` (localhost).
   - Hình ảnh tải lên được đọc và xử lý hoàn toàn trong bộ nhớ RAM, **không ghi tệp vĩnh viễn** xuống đĩa máy chủ.
   - Không log base64 hình ảnh, không log số CCCD, không log vector embedding khuôn mặt hoặc thông tin cá nhân ra console.
   - Không gọi ra bất kỳ dịch vụ đám mây bên thứ ba nào.

2. **Giới hạn phạm vi (Explicit Scope Boundaries)**:
   - Dự án **CHƯA** đọc mã QR mặt trước thẻ CCCD.
   - Chưa tích hợp vào luồng xác thực danh tính sinh viên của CampusHub Backend/Frontend.
   - Điểm so khớp sinh trắc học khuôn mặt đạt ngưỡng **không chứng minh** thẻ CCCD là thẻ thật và không thay thế việc kiểm tra pháp lý.

---

## 11. Bản quyền & Tác giả

- **Dự án**: CampusHub-OCR (Dự án nghiên cứu & thử nghiệm OCR và Xác minh khuôn mặt độc lập)
- **Tác giả / Phát triển**: Phạm Hữu Cảnh

