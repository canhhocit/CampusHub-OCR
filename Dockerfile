# Sử dụng Python 3.11-slim ổn định và tối ưu dung lượng
FROM python:3.11-slim

# Thiết lập biến môi trường chuẩn
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Cài đặt các gói hệ thống cần thiết cho OpenCV và PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tạo người dùng non-root (theo tiêu chuẩn bảo mật UID 1000 của Hugging Face Spaces)
RUN useradd -m -u 1000 user

USER user
WORKDIR $HOME/app

# Cài đặt Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Sao chép toàn bộ mã nguồn
COPY --chown=user:user . .

# Đảm bảo 2 tệp mô hình ONNX luôn có sẵn trong image Docker
RUN mkdir -p models/face && \
    if [ ! -s models/face/face_detection_yunet_2023mar.onnx ]; then \
        echo "Downloading YuNet ONNX model..." && \
        curl -fsSL -o models/face/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx; \
    fi && \
    if [ ! -s models/face/face_recognition_sface_2021dec.onnx ]; then \
        echo "Downloading SFace ONNX model..." && \
        curl -fsSL -o models/face/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx; \
    fi

# Khai báo cổng
EXPOSE 7860 8000 8001 10000

# Khởi chạy FastAPI qua Uvicorn (nhận cổng tự động từ Render hoặc 7860 mặc định)
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
