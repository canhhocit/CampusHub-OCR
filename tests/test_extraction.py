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

def test_1_id_number_extracted_from_s6_no_concatenation():
    """1. ID number extracted from s6/No036205013225."""
    detections = [
        RawDetection(text="CAN CU'O'C CONG DAN", confidence=0.99),
        RawDetection(text="s6/No036205013225", confidence=0.98),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert fields["idNumber"].value == "036205013225"
    assert fields["idNumber"].status in (FieldStatus.EXTRACTED, FieldStatus.VALID)

def test_2_name_extracted_from_separate_line():
    """2. Name extracted from a separate line."""
    detections = [
        RawDetection(text="Hova ten/ Full name.", confidence=0.98),
        RawDetection(text="PHAM HU'U CANH", confidence=0.99),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert fields["fullName"].value == "PHAM HU'U CANH"
    assert fields["fullName"].status in (FieldStatus.EXTRACTED, FieldStatus.VALID)

def test_3_dob_extracted_from_concatenated_label():
    """3. Date of birth extracted from a concatenated label and value."""
    detections = [
        RawDetection(text="Ngay sinh/Date of birth26/06/2005", confidence=0.97),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert fields["dateOfBirth"].value == "26/06/2005"
    assert fields["dateOfBirth"].status in (FieldStatus.EXTRACTED, FieldStatus.VALID)

def test_4_gender_and_nationality_on_one_line():
    """4. Gender and nationality extracted when both appear on one line."""
    detections = [
        RawDetection(text="Gioi tinh/SexNamQuoc tich/NationalityViet Nam", confidence=0.96),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert fields["gender"].value == "Nam"
    assert fields["nationality"].value == "Việt Nam"

def test_5_place_of_origin_from_following_line():
    """5. Place of origin extracted from the following line."""
    detections = [
        RawDetection(text="Quequan / Place of onigin", confidence=0.95),
        RawDetection(text="Yen Tho,Y YenNam Dinh", confidence=0.94),
        RawDetection(text="Noi thung trul Place ofresidenceThön ThanhBinh", confidence=0.95),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert "Yen Tho" in fields["placeOfOrigin"].value
    assert "Place of onigin" not in fields["placeOfOrigin"].value

def test_6_multiline_residence_without_absorbing_expiry():
    """6. Multiline place of residence extracted without absorbing the expiry date."""
    detections = [
        RawDetection(text="Noi thung trul Place ofresidenceThön ThanhBinh", confidence=0.95),
        RawDetection(text="Co giatrden26/06/2030", confidence=0.97),
        RawDetection(text="Yen ThoY YenNam Dinh", confidence=0.94),
        RawDetection(text="Dateofexpiry", confidence=0.90),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    res_val = fields["placeOfResidence"].value
    assert "Thön ThanhBinh" in res_val
    assert "Yen Tho" in res_val
    # Must NOT absorb the expiry date into residence!
    assert "26/06/2030" not in res_val
    assert "Co giatrden" not in res_val
    assert "Dateofexpiry" not in res_val

def test_7_expiry_date_extracted_correctly():
    """7. Expiry date extracted correctly."""
    detections = [
        RawDetection(text="Co giatrden26/06/2030", confidence=0.98),
        RawDetection(text="Dateofexpiry", confidence=0.92),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert fields["expiryDate"].value == "26/06/2030"
    assert fields["expiryDate"].status in (FieldStatus.EXTRACTED, FieldStatus.VALID)

def test_8_missing_id_number():
    """8. Missing ID number."""
    detections = [
        RawDetection(text="CAN CU'O'C CONG DAN", confidence=0.99),
        RawDetection(text="Hova ten: PHAM HUU CANH", confidence=0.95),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert fields["idNumber"].status in (FieldStatus.MISSING, FieldStatus.NOT_DETECTED)
    assert fields["idNumber"].value is None

def test_9_invalid_dates_and_malformed_id():
    """9. Invalid dates and malformed ID numbers."""
    detections = [
        RawDetection(text="Số / No.: 12345", confidence=0.99), # only 5 digits
        RawDetection(text="Ngày sinh: 31/02/2005", confidence=0.95), # Feb 31 invalid calendar
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    # 5 digits -> missing 12-digit or invalid
    assert fields["idNumber"].status in (FieldStatus.MISSING, FieldStatus.INVALID)
    # Feb 31 calendar invalid
    assert fields["dateOfBirth"].status == FieldStatus.INVALID
    assert "không hợp lệ trong lịch" in fields["dateOfBirth"].validation_message

def test_10_empty_ocr_results_and_low_confidence():
    """10. Empty OCR results and low-confidence detections."""
    # Empty
    side, fields_empty = CCCDExtractor.extract_fields([], requested_side=CardSide.FRONT)
    assert fields_empty["idNumber"].status in (FieldStatus.MISSING, FieldStatus.NOT_DETECTED)
    assert fields_empty["fullName"].status in (FieldStatus.MISSING, FieldStatus.NOT_DETECTED)

    # Low confidence detection (< 0.70)
    low_conf_detections = [
        RawDetection(text="Số: 036205013225", confidence=0.45),
        RawDetection(text="Họ và tên: PHAM HUU CANH", confidence=0.55),
    ]
    _, fields_low = CCCDExtractor.extract_fields(low_conf_detections, requested_side=CardSide.FRONT)
    assert fields_low["idNumber"].status == FieldStatus.NEEDS_REVIEW
    assert fields_low["idNumber"].value == "036205013225"
    assert fields_low["fullName"].status == FieldStatus.NEEDS_REVIEW

def test_end_to_end_user_sample():
    """Full reproduction and resolution of user's sample raw OCR output."""
    raw_lines = [
        "CONG HOA XAHOI CHU NGHiAVIET NAM",
        "SOCIALISTREPUBLICOFVIETNAM",
        "CAN CU'O'C CONG DAN",
        "Citizen ldentity Card",
        "s6/No036205013225",
        "Hova ten/ Full name.",
        "PHAM HU'U CANH",
        "Ngay sinh/Date of birth26/06/2005",
        "Gioi tinh/SexNamQuoc tich/NationalityViet Nam",
        "Quequan / Place of onigin",
        "Yen Tho,Y YenNam Dinh",
        "Noi thung trul Place ofresidenceThön ThanhBinh",
        "Co giatrden26/06/2030",
        "Yen ThoY YenNam Dinh",
        "Dateofexpiry",
    ]
    detections = [RawDetection(text=t, confidence=0.95) for t in raw_lines]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)

    assert side == CardSide.FRONT
    assert fields["idNumber"].value == "036205013225"
    assert fields["idNumber"].status in (FieldStatus.EXTRACTED, FieldStatus.VALID)
    assert fields["fullName"].value == "PHAM HU'U CANH"
    assert fields["dateOfBirth"].value == "26/06/2005"
    assert fields["gender"].value == "Nam"
    assert fields["nationality"].value == "Việt Nam"
    assert "Yen Tho" in fields["placeOfOrigin"].value
    assert "Thön ThanhBinh" in fields["placeOfResidence"].value
    assert fields["expiryDate"].value == "26/06/2030"

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
    assert "Sẹo chấm" in fields["personalFeatures"].value
    assert fields["issueDate"].value == "15/09/2022"
    assert "CỤC TRƯỞNG" in fields["issueAuthority"].value
    assert "I<VNM" in fields["mrz"].value

def test_fix_ocr_digit_errors_utility():
    from app.services.cccd_extractor import fix_ocr_digit_errors
    assert fix_ocr_digit_errors("OO1200000001") == "001200000001"
    assert fix_ocr_digit_errors("0793O100000Z") == "079301000002"
    assert fix_ocr_digit_errors("26/O6/2OO5") == "26/06/2005"
    assert fix_ocr_digit_errors("l5/O8/2O2O") == "15/08/2020"

def test_extract_id_number_with_ocr_confusion():
    detections = [
        RawDetection(text="CAN CU'O'C CONG DAN", confidence=0.98),
        RawDetection(text="So / No.: OO12OO0OO0O1", confidence=0.95),  # 'O' instead of '0'
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert fields["idNumber"].value == "001200000001"
    assert fields["idNumber"].status in (FieldStatus.EXTRACTED, FieldStatus.VALID)

def test_extract_id_number_with_spaced_digits():
    detections = [
        RawDetection(text="CAN CU'O'C CONG DAN", confidence=0.98),
        RawDetection(text="Số: 036 205 013 225", confidence=0.96),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert fields["idNumber"].value == "036205013225"
    assert fields["idNumber"].status in (FieldStatus.EXTRACTED, FieldStatus.VALID)

def test_extract_dates_with_ocr_confusion():
    detections = [
        RawDetection(text="Ngay sinh: 26/O6/2OO5", confidence=0.97),
        RawDetection(text="Co gia tri den: 26/O6/2O3O", confidence=0.96),
    ]
    side, fields = CCCDExtractor.extract_fields(detections, requested_side=CardSide.FRONT)
    assert fields["dateOfBirth"].value == "26/06/2005"
    assert fields["expiryDate"].value == "26/06/2030"
