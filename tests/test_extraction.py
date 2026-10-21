from app.schemas.ocr import CardSide, FieldStatus, RawDetection
from app.services.cccd_extractor import CCCDExtractor

def test_detect_side_front():
    detections = [
        RawDetection(text="CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", confidence=0.99),
        RawDetection(text="CĂN CƯỚC CÔNG DÂN", confidence=0.98),
        RawDetection(text="Số / No.: 001200000001", confidence=0.99),
        RawDetection(text="Họ và tên: NGUYỄN VĂN AN", confidence=0.97),
    ]
    side = CCCDExtractor.detect_side(detections)
    assert side == CardSide.FRONT

def test_detect_side_back():
    detections = [
        RawDetection(text="ĐẶC ĐIỂM NHÂN DẠNG", confidence=0.98),
        RawDetection(text="Nốt ruồi đuôi lông mày", confidence=0.95),
        RawDetection(text="Ngày 10 tháng 05 năm 2021", confidence=0.98),
        RawDetection(text="CỤC TRƯỞNG CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH", confidence=0.99),
    ]
    side = CCCDExtractor.detect_side(detections)
    assert side == CardSide.BACK

def test_extract_front_fields():
    detections = [
        RawDetection(text="CĂN CƯỚC CÔNG DÂN", confidence=0.99),
        RawDetection(text="Số / No.: 012345678912", confidence=0.99),
        RawDetection(text="Họ và tên / Full name:", confidence=0.98),
        RawDetection(text="TRẦN VĂN BÌNH", confidence=0.97),
        RawDetection(text="Ngày sinh / Date of birth: 12/04/2001", confidence=0.98),
        RawDetection(text="Giới tính: Nam", confidence=0.99),
        RawDetection(text="Quốc tịch: Việt Nam", confidence=0.99),
        RawDetection(text="Quê quán: Xã Tân Lập, Huyện Đan Phượng, Hà Nội", confidence=0.95),
        RawDetection(text="Nơi thường trú: Số 10 Tràng Tiền, Hoàn Kiếm, Hà Nội", confidence=0.94),
        RawDetection(text="Có giá trị đến / Date of expiry: 12/04/2026", confidence=0.98),
    ]

    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert side == CardSide.FRONT
    assert fields["id_number"].value == "012345678912"
    assert fields["id_number"].status == FieldStatus.VALID
    assert fields["full_name"].value == "TRẦN VĂN BÌNH"
    assert fields["full_name"].status == FieldStatus.VALID
    assert fields["date_of_birth"].value == "12/04/2001"
    assert fields["gender"].value == "Nam"
    assert fields["nationality"].value == "Việt Nam"
    assert "Đan Phượng" in fields["place_of_origin"].value
    assert "Tràng Tiền" in fields["place_of_residence"].value
    assert fields["expiry_date"].value == "12/04/2026"

def test_extract_back_fields():
    detections = [
        RawDetection(text="Đặc điểm nhân dạng / Personal identification:", confidence=0.96),
        RawDetection(text="Sẹo chấm 0.5cm trên sau cánh mũi trái", confidence=0.95),
        RawDetection(text="Hà Nội, ngày 15 tháng 09 năm 2022", confidence=0.98),
        RawDetection(text="CỤC TRƯỞNG CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI", confidence=0.99),
        RawDetection(text="I<VNM012345678912<<<<<<<<<<<<<", confidence=0.97),
        RawDetection(text="0104128M2604125VNM<<<<<<<<<<<2", confidence=0.96),
        RawDetection(text="TRAN<<VAN<BINH<<<<<<<<<<<<<<<<", confidence=0.98),
    ]

    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.BACK)
    assert side == CardSide.BACK
    assert "Sẹo chấm" in fields["personal_features"].value
    assert fields["issue_date"].value == "15/09/2022"
    assert "CỤC TRƯỞNG" in fields["issue_authority"].value
    assert "I<VNM" in fields["mrz"].value

def test_missing_fields_not_fabricated():
    # Only ID is present, other fields missing
    detections = [
        RawDetection(text="CĂN CƯỚC CÔNG DÂN", confidence=0.99),
        RawDetection(text="001200000001", confidence=0.99),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert fields["id_number"].value == "001200000001"
    assert fields["full_name"].status == FieldStatus.NOT_DETECTED
    assert fields["full_name"].value is None
    assert fields["date_of_birth"].status == FieldStatus.NOT_DETECTED
    assert fields["date_of_birth"].value is None
