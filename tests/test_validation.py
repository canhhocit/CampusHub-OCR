import pytest
from app.services.image_processor import ImageProcessor, ImageValidationError
from tests.conftest import create_valid_image_bytes, create_corrupt_image_bytes

def test_empty_image_bytes():
    with pytest.raises(ImageValidationError) as exc_info:
        ImageProcessor.validate_image_bytes(b"", "empty.jpg")
    assert exc_info.value.status_code == 400
    assert "rỗng" in exc_info.value.message

def test_unsupported_extension():
    data = create_valid_image_bytes()
    with pytest.raises(ImageValidationError) as exc_info:
        ImageProcessor.validate_image_bytes(data, "card.pdf")
    assert exc_info.value.status_code == 415
    assert "Định dạng tệp không được hỗ trợ" in exc_info.value.message

def test_invalid_magic_bytes():
    fake_jpg = b"Not really an image header text here"
    with pytest.raises(ImageValidationError) as exc_info:
        ImageProcessor.validate_image_bytes(fake_jpg, "card.jpg")
    assert exc_info.value.status_code == 400
    assert "không phải là hình ảnh" in exc_info.value.message

def test_corrupted_image_decode():
    corrupted = create_corrupt_image_bytes()
    with pytest.raises(ImageValidationError) as exc_info:
        ImageProcessor.validate_image_bytes(corrupted, "corrupt.jpg")
    assert exc_info.value.status_code == 400
    assert "Không thể giải mã" in exc_info.value.message

def test_image_dimensions_too_small():
    tiny = create_valid_image_bytes(width=50, height=50)
    with pytest.raises(ImageValidationError) as exc_info:
        ImageProcessor.validate_image_bytes(tiny, "tiny.jpg")
    assert exc_info.value.status_code == 400
    assert "quá nhỏ" in exc_info.value.message

def test_valid_jpeg_and_png():
    jpeg_bytes = create_valid_image_bytes(width=800, height=500, format="JPEG")
    img, meta = ImageProcessor.validate_image_bytes(jpeg_bytes, "test.jpg")
    assert img is not None
    assert meta.width == 800
    assert meta.height == 500
    assert meta.format == "JPEG"

    png_bytes = create_valid_image_bytes(width=600, height=400, format="PNG")
    img_png, meta_png = ImageProcessor.validate_image_bytes(png_bytes, "test.png")
    assert img_png is not None
    assert meta_png.width == 600
    assert meta_png.format == "PNG"

def test_preprocessing():
    jpeg_bytes = create_valid_image_bytes(width=400, height=300)
    img, _ = ImageProcessor.validate_image_bytes(jpeg_bytes)
    processed = ImageProcessor.preprocess_for_ocr(img)
    assert processed is not None
    # Verify small image was upscaled for better text recognition
    assert processed.shape[1] >= 800
