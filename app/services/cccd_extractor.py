import re
from typing import Optional, Tuple
from app.schemas.ocr import (
    CardSide,
    FieldStatus,
    ExtractedField,
    RawDetection,
)

class CCCDExtractor:
    @staticmethod
    def detect_side(detections: list[RawDetection]) -> CardSide:
        """Determines if the document is FRONT or BACK side based on landmark keywords."""
        all_text = " ".join(d.text.upper() for d in detections)
        
        front_score = 0
        back_score = 0
        
        # Front landmarks
        if "CĂN CƯỚC CÔNG DÂN" in all_text or "CAN CUOC CONG DAN" in all_text:
            front_score += 3
        if "HỌ VÀ TÊN" in all_text or "HO VA TEN" in all_text or "FULL NAME" in all_text:
            front_score += 2
        if "NGÀY SINH" in all_text or "DATE OF BIRTH" in all_text:
            front_score += 2
        if "QUỐC TỊCH" in all_text or "NATIONALITY" in all_text:
            front_score += 2
        if "QUÊ QUÁN" in all_text or "PLACE OF ORIGIN" in all_text:
            front_score += 2
        if "NƠI THƯỜNG TRÚ" in all_text or "PLACE OF RESIDENCE" in all_text:
            front_score += 2
        if re.search(r"\b\d{12}\b", all_text):
            front_score += 3

        # Back landmarks
        if "ĐẶC ĐIỂM NHÂN DẠNG" in all_text or "DAC DIEM NHAN DANG" in all_text:
            back_score += 3
        if "NGÀY CẤP" in all_text or "NGAY CAP" in all_text:
            back_score += 2
        if "CỤC TRƯỞNG" in all_text or "CUC TRUONG" in all_text or "GIÁM ĐỐC" in all_text:
            back_score += 2
        if "CẢNH SÁT QUẢN LÝ" in all_text or "CANH SAT QUAN LY" in all_text:
            back_score += 2
        if "I<VNM" in all_text or "<<" in all_text:
            back_score += 3
            
        if back_score > front_score:
            return CardSide.BACK
        return CardSide.FRONT

    @classmethod
    def extract_fields(
        cls, detections: list[RawDetection], requested_side: CardSide = CardSide.AUTO
    ) -> Tuple[CardSide, dict[str, ExtractedField]]:
        """Extracts structured fields from raw OCR detections."""
        effective_side = requested_side
        if effective_side == CardSide.AUTO:
            effective_side = cls.detect_side(detections)

        if effective_side == CardSide.FRONT:
            fields = cls._extract_front(detections)
        else:
            fields = cls._extract_back(detections)

        return effective_side, fields

    @staticmethod
    def _extract_front(detections: list[RawDetection]) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}
        lines = [d.text.strip() for d in detections]
        full_text = "\n".join(lines)

        # 1. ID Number (12 digits)
        id_field = ExtractedField(
            name="id_number",
            label="Số định danh cá nhân / CCCD",
            status=FieldStatus.NOT_DETECTED,
        )
        for d in detections:
            # Look for 12 digits directly or after 'Số / No'
            m = re.search(r"\b(\d{3}[\s.-]?\d{3}[\s.-]?\d{6}|\d{12})\b", d.text)
            if m:
                clean_id = re.sub(r"[\s.-]", "", m.group(1))
                if len(clean_id) == 12 and clean_id.isdigit():
                    id_field.value = clean_id
                    id_field.status = FieldStatus.VALID
                    id_field.confidence = d.confidence
                    id_field.raw_text = d.text
                    break
        # Fallback check across whole text
        if id_field.status == FieldStatus.NOT_DETECTED:
            m = re.search(r"\b(\d{12})\b", full_text)
            if m:
                id_field.value = m.group(1)
                id_field.status = FieldStatus.VALID
                id_field.raw_text = m.group(0)

        fields["id_number"] = id_field

        # 2. Full Name
        name_field = ExtractedField(
            name="full_name",
            label="Họ và tên",
            status=FieldStatus.NOT_DETECTED,
        )
        for i, d in enumerate(detections):
            upper = d.text.upper()
            if "HỌ VÀ TÊN" in upper or "HO VA TEN" in upper or "FULL NAME" in upper:
                # Value may be on the same line after colon, or on the next line
                same_line = re.sub(r"^(.*?)(HỌ VÀ TÊN|HO VA TEN|FULL NAME)[:\s/]*", "", d.text, flags=re.IGNORECASE).strip()
                if same_line and len(same_line) >= 3 and same_line == same_line.upper():
                    name_field.value = same_line
                    name_field.status = FieldStatus.VALID
                    name_field.confidence = d.confidence
                    name_field.raw_text = d.text
                    break
                elif i + 1 < len(detections):
                    next_text = detections[i + 1].text.strip()
                    # Check if next line is uppercase Vietnamese name and doesn't contain field label
                    if next_text and next_text == next_text.upper() and not any(kw in next_text for kw in ["NGÀY", "SINH", "GIỚI", "TÍNH", "SỐ"]):
                        name_field.value = next_text
                        name_field.status = FieldStatus.VALID
                        name_field.confidence = detections[i + 1].confidence
                        name_field.raw_text = next_text
                        break
        fields["full_name"] = name_field

        # 3. Date of birth
        dob_field = ExtractedField(
            name="date_of_birth",
            label="Ngày sinh",
            status=FieldStatus.NOT_DETECTED,
        )
        for i, d in enumerate(detections):
            if any(kw in d.text.upper() for kw in ["NGÀY SINH", "NGAY SINH", "BIRTH"]):
                m = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", d.text)
                if m:
                    dob_field.value = m.group(1).replace("-", "/")
                    dob_field.status = FieldStatus.VALID
                    dob_field.confidence = d.confidence
                    dob_field.raw_text = d.text
                    break
                elif i + 1 < len(detections):
                    m_next = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", detections[i + 1].text)
                    if m_next:
                        dob_field.value = m_next.group(1).replace("-", "/")
                        dob_field.status = FieldStatus.VALID
                        dob_field.confidence = detections[i + 1].confidence
                        dob_field.raw_text = detections[i + 1].text
                        break
        fields["date_of_birth"] = dob_field

        # 4. Gender
        gender_field = ExtractedField(
            name="gender",
            label="Giới tính",
            status=FieldStatus.NOT_DETECTED,
        )
        for d in detections:
            if "GIỚI TÍNH" in d.text.upper() or "GIOI TINH" in d.text.upper() or "SEX" in d.text.upper():
                upper = d.text.upper()
                if "NỮ" in upper or "FEMALE" in upper:
                    gender_field.value = "Nữ"
                    gender_field.status = FieldStatus.VALID
                    gender_field.confidence = d.confidence
                    gender_field.raw_text = d.text
                    break
                elif "NAM" in upper or "MALE" in upper:
                    gender_field.value = "Nam"
                    gender_field.status = FieldStatus.VALID
                    gender_field.confidence = d.confidence
                    gender_field.raw_text = d.text
                    break
        fields["gender"] = gender_field

        # 5. Nationality
        nat_field = ExtractedField(
            name="nationality",
            label="Quốc tịch",
            status=FieldStatus.NOT_DETECTED,
        )
        for d in detections:
            upper = d.text.upper()
            if "QUỐC TỊCH" in upper or "QUOC TICH" in upper or "NATIONALITY" in upper:
                if "VIỆT NAM" in upper or "VIET NAM" in upper:
                    nat_field.value = "Việt Nam"
                    nat_field.status = FieldStatus.VALID
                    nat_field.confidence = d.confidence
                    nat_field.raw_text = d.text
                    break
        fields["nationality"] = nat_field

        # 6. Place of origin
        origin_field = ExtractedField(
            name="place_of_origin",
            label="Quê quán",
            status=FieldStatus.NOT_DETECTED,
        )
        for i, d in enumerate(detections):
            upper = d.text.upper()
            if "QUÊ QUÁN" in upper or "QUE QUAN" in upper or "ORIGIN" in upper:
                val = re.sub(r"^(.*?)(QUÊ QUÁN|QUE QUAN|ORIGIN)[:\s/]*", "", d.text, flags=re.IGNORECASE).strip()
                collected = [val] if val else []
                # Check next line if it continues the address and is not the next field header
                if i + 1 < len(detections):
                    next_text = detections[i + 1].text.strip()
                    if not any(kw in next_text.upper() for kw in ["NƠI THƯỜNG TRÚ", "NOI THUONG TRU", "RESIDENCE", "GIÁ TRỊ"]):
                        collected.append(next_text)
                combined = ", ".join(filter(None, collected))
                if combined:
                    origin_field.value = combined
                    origin_field.status = FieldStatus.VALID
                    origin_field.confidence = d.confidence
                    origin_field.raw_text = combined
                    break
        fields["place_of_origin"] = origin_field

        # 7. Place of residence
        residence_field = ExtractedField(
            name="place_of_residence",
            label="Nơi thường trú",
            status=FieldStatus.NOT_DETECTED,
        )
        for i, d in enumerate(detections):
            upper = d.text.upper()
            if "NƠI THƯỜNG TRÚ" in upper or "NOI THUONG TRU" in upper or "RESIDENCE" in upper:
                val = re.sub(r"^(.*?)(NƠI THƯỜNG TRÚ|NOI THUONG TRU|RESIDENCE)[:\s/]*", "", d.text, flags=re.IGNORECASE).strip()
                collected = [val] if val else []
                # Capture following 1-2 lines until expiry or end
                for offset in (1, 2):
                    if i + offset < len(detections):
                        next_line = detections[i + offset].text.strip()
                        if not any(kw in next_line.upper() for kw in ["CÓ GIÁ TRỊ", "CO GIA TRI", "EXPIRY", "VAL"]):
                            collected.append(next_line)
                        else:
                            break
                combined = ", ".join(filter(None, collected))
                if combined:
                    residence_field.value = combined
                    residence_field.status = FieldStatus.VALID
                    residence_field.confidence = d.confidence
                    residence_field.raw_text = combined
                    break
        fields["place_of_residence"] = residence_field

        # 8. Expiry date
        expiry_field = ExtractedField(
            name="expiry_date",
            label="Có giá trị đến",
            status=FieldStatus.NOT_DETECTED,
        )
        for d in detections:
            upper = d.text.upper()
            if any(kw in upper for kw in ["CÓ GIÁ TRỊ", "CO GIA TRI", "EXPIRY", "VAL"]):
                if "KHÔNG THỜI HẠN" in upper or "KHONG THOI HAN" in upper:
                    expiry_field.value = "Không thời hạn"
                    expiry_field.status = FieldStatus.VALID
                    expiry_field.confidence = d.confidence
                    expiry_field.raw_text = d.text
                    break
                m = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", d.text)
                if m:
                    expiry_field.value = m.group(1).replace("-", "/")
                    expiry_field.status = FieldStatus.VALID
                    expiry_field.confidence = d.confidence
                    expiry_field.raw_text = d.text
                    break
        fields["expiry_date"] = expiry_field

        return fields

    @staticmethod
    def _extract_back(detections: list[RawDetection]) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}
        lines = [d.text.strip() for d in detections]

        # 1. Personal characteristics
        features_field = ExtractedField(
            name="personal_features",
            label="Đặc điểm nhân dạng",
            status=FieldStatus.NOT_DETECTED,
        )
        for i, d in enumerate(detections):
            upper = d.text.upper()
            if "ĐẶC ĐIỂM NHÂN DẠNG" in upper or "DAC DIEM NHAN DANG" in upper or "IDENTIFICATION" in upper:
                val = re.sub(r"^(.*?)(ĐẶC ĐIỂM NHÂN DẠNG|DAC DIEM NHAN DANG|IDENTIFICATION)[:\s/]*", "", d.text, flags=re.IGNORECASE).strip()
                collected = [val] if val else []
                if i + 1 < len(detections):
                    next_text = detections[i + 1].text.strip()
                    if not any(kw in next_text.upper() for kw in ["NGÀY", "THÁNG", "NĂM", "CỤC TRƯỞNG", "GIÁM ĐỐC", "I<VNM"]):
                        collected.append(next_text)
                combined = " ".join(filter(None, collected))
                if combined:
                    features_field.value = combined
                    features_field.status = FieldStatus.VALID
                    features_field.confidence = d.confidence
                    features_field.raw_text = combined
                    break
        fields["personal_features"] = features_field

        # 2. Date of Issue
        issue_date_field = ExtractedField(
            name="issue_date",
            label="Ngày cấp",
            status=FieldStatus.NOT_DETECTED,
        )
        for d in detections:
            # Pattern: ngày DD tháng MM năm YYYY
            m = re.search(r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", d.text, re.IGNORECASE)
            if m:
                day, month, year = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
                issue_date_field.value = f"{day}/{month}/{year}"
                issue_date_field.status = FieldStatus.VALID
                issue_date_field.confidence = d.confidence
                issue_date_field.raw_text = d.text
                break
            # Standard DD/MM/YYYY after 'ngày' or 'date'
            m2 = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", d.text)
            if m2 and any(kw in d.text.upper() for kw in ["NGÀY", "CẤP", "DATE"]):
                issue_date_field.value = m2.group(1).replace("-", "/")
                issue_date_field.status = FieldStatus.VALID
                issue_date_field.confidence = d.confidence
                issue_date_field.raw_text = d.text
                break
        fields["issue_date"] = issue_date_field

        # 3. Authority
        auth_field = ExtractedField(
            name="issue_authority",
            label="Cơ quan cấp",
            status=FieldStatus.NOT_DETECTED,
        )
        for d in detections:
            upper = d.text.upper()
            if any(kw in upper for kw in ["CỤC TRƯỞNG", "CUC TRUONG", "CẢNH SÁT QUẢN LÝ", "GIÁM ĐỐC CÔNG AN"]):
                auth_field.value = d.text.strip()
                auth_field.status = FieldStatus.VALID
                auth_field.confidence = d.confidence
                auth_field.raw_text = d.text
                break
        fields["issue_authority"] = auth_field

        # 4. MRZ (Machine Readable Zone - 3 lines on chip card)
        mrz_field = ExtractedField(
            name="mrz",
            label="Mã vùng đọc máy (MRZ)",
            status=FieldStatus.NOT_DETECTED,
        )
        mrz_lines = []
        for d in detections:
            cleaned = d.text.replace(" ", "").upper()
            # MRZ line on Vietnamese ID cards has 30 characters, contains << or starts with I<VNM
            if "<<" in cleaned or cleaned.startswith("I<VNM") or (len(cleaned) >= 28 and "<" in cleaned):
                mrz_lines.append(cleaned)
        if mrz_lines:
            mrz_field.value = "\n".join(mrz_lines)
            mrz_field.status = FieldStatus.VALID
            mrz_field.confidence = round(sum(d.confidence for d in detections if d.text in mrz_lines) / max(len(mrz_lines), 1), 4)
            mrz_field.raw_text = "\n".join(mrz_lines)
        fields["mrz"] = mrz_field

        return fields
