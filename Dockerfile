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

# Sao chép toàn bộ mã nguồn và mô hình (ONNX models trong models/face)
COPY --chown=user:user . .

# Khai báo cổng 7860 (cổng mặc định của Hugging Face Spaces)
EXPOSE 7860

# Khởi chạy FastAPI qua Uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
