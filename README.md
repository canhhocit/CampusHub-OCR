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

Để đánh giá định lượng độ chính xác OCR và khả năng trích xuất:
```bash
python evaluate.py --dataset tests/fixtures/ground_truth_sample.json
```

Chương trình sẽ tự động phân tích và xuất bảng báo cáo định lượng:
- **Exact-Match (EM)**: Tỷ lệ trường trích xuất trùng khớp 100% với dữ liệu chuẩn.
- **Detection Rate**: Tỷ lệ trường được phát hiện trong văn bản.
- **Character Error Rate (CER)**: Tỷ lệ sai số ký tự (Substitutions + Deletions + Insertions / Total characters).
- **Latency**: Thời gian trích xuất trung bình (mili-giây).
- **Phân loại chất lượng ảnh**: Đo lường tỷ lệ thành công theo nhóm ảnh phòng lab (`high_quality_studio`) và ảnh điện thoại thực tế (`natural_mobile`).

> **Quy định đạo đức dữ liệu**: Bộ benchmark mẫu trong repository sử dụng 100% dữ liệu nhân thân giả lập (synthetic data). Tuyệt đối không đưa ảnh CCCD thật hoặc dữ liệu cá nhân của người dùng vào mã nguồn Git.

---

## 8. Nguyên tắc Bảo mật & Giới hạn Phạm vi

1. **Bảo mật dữ liệu (Data Privacy)**:
   - Dịch vụ chỉ mở trên `127.0.0.1` (localhost).
   - Hình ảnh tải lên được đọc và xử lý hoàn toàn trong bộ nhớ RAM, **không ghi tệp vĩnh viễn** xuống đĩa máy chủ.
   - Không log base64 hình ảnh, không log số CCCD hoặc thông tin cá nhân ra console.
   - Không gọi ra bất kỳ dịch vụ OCR đám mây bên thứ ba nào.

2. **Giới hạn phạm vi (Explicit Scope Boundaries)**:
   - Dự án **CHƯA** bao gồm so khớp khuôn mặt (face verification / face matching).
   - Dự án **CHƯA** bao gồm kiểm tra liveness hoặc chống giả mạo khuôn mặt (anti-spoofing).
   - Chưa tích hợp vào luồng xác thực danh tính sinh viên của CampusHub Backend/Frontend.
   - Các trường hợp ảnh bị nghiêng > 45 độ, chói lóa mất góc hoặc bị ngón tay che số CCCD có thể khiến trường trả về `NOT_DETECTED` hoặc `UNREADABLE`.

---

## 9. Bản quyền & Tác giả

- **Dự án**: CampusHub-OCR (Dự án nghiên cứu & thử nghiệm OCR độc lập)
- **Tác giả / Phát triển**: Phạm Hữu Cảnh
