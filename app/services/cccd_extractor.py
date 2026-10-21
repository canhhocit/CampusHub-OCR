import re
import unicodedata
from datetime import datetime
from typing import Optional, Tuple, Set, List
from app.schemas.ocr import (
    CardSide,
    FieldStatus,
    ExtractedField,
    RawDetection,
)

# Vietnamese province codes on CCCD (001 - 096)
VALID_PROVINCE_CODES = {
    "001": "Hà Nội", "002": "Hà Giang", "004": "Cao Bằng", "006": "Bắc Kạn",
    "008": "Tuyên Quang", "010": "Lào Cai", "011": "Điện Biên", "012": "Lai Châu",
    "014": "Sơn La", "015": "Yên Bái", "017": "Hòa Bình", "019": "Thái Nguyên",
    "020": "Lạng Sơn", "022": "Quảng Ninh", "024": "Bắc Giang", "025": "Phú Thọ",
    "026": "Vĩnh Phúc", "027": "Bắc Ninh", "030": "Hải Dương", "031": "Hải Phòng",
    "033": "Hưng Yên", "034": "Thái Bình", "035": "Hà Nam", "036": "Nam Định",
    "037": "Ninh Bình", "038": "Thanh Hóa", "040": "Nghệ An", "042": "Hà Tĩnh",
    "044": "Quảng Bình", "045": "Quảng Trị", "046": "Thừa Thiên Huế", "048": "Đà Nẵng",
    "049": "Quảng Nam", "051": "Quảng Ngãi", "052": "Bình Định", "054": "Phú Yên",
    "056": "Khánh Hòa", "058": "Ninh Thuận", "060": "Bình Thuận", "062": "Kon Tum",
    "064": "Gia Lai", "066": "Đắk Lắk", "067": "Đắk Nông", "068": "Lâm Đồng",
    "070": "Bình Phước", "072": "Tây Ninh", "074": "Bình Dương", "075": "Đồng Nai",
    "077": "Bà Rịa - Vũng Tàu", "079": "TP. Hồ Chí Minh", "080": "Long An",
    "082": "Tiền Giang", "083": "Bến Tre", "084": "Trà Vinh", "086": "Vĩnh Long",
    "087": "Đồng Tháp", "089": "An Giang", "091": "Kiên Giang", "092": "Cần Thơ",
    "093": "Hậu Giang", "094": "Sóc Trăng", "095": "Bạc Liêu", "096": "Cà Mau"
}

def remove_accents(input_str: str) -> str:
    """Removes Vietnamese tone marks and accents for robust OCR label matching."""
    if not input_str:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).replace("đ", "d").replace("Đ", "D")

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""

class CCCDExtractor:
    @staticmethod
    def detect_side(detections: list[RawDetection]) -> CardSide:
        """Determines if the card is FRONT or BACK side based on landmark keywords."""
        all_text = " ".join(d.text.upper() for d in detections)
        all_norm = remove_accents(all_text)
        
        front_score = 0
        back_score = 0
        
        # Front landmarks
        if "CAN CUOC CONG DAN" in all_norm or "CITIZEN" in all_norm:
            front_score += 4
        if "HO VA TEN" in all_norm or "HOVA TEN" in all_norm or "FULL NAME" in all_norm:
            front_score += 3
        if "NGAY SINH" in all_norm or "DATE OF BIRTH" in all_norm or "BIRTH" in all_norm:
            front_score += 2
        if "QUOC TICH" in all_norm or "NATIONALITY" in all_norm:
            front_score += 2
        if "QUE QUAN" in all_norm or "QUEQUAN" in all_norm or "ORIGIN" in all_norm or "ONIGIN" in all_norm:
            front_score += 2
        if "NOI THUONG TRU" in all_norm or "RESIDENCE" in all_norm:
            front_score += 2
        if re.search(r"\d{12}", all_text):
            front_score += 4

        # Back landmarks
        if "DAC DIEM NHAN DANG" in all_norm or "IDENTIFICATION" in all_norm:
            back_score += 4
        if "NGAY CAP" in all_norm or "DATE OF ISSUE" in all_norm:
            back_score += 3
        if "CUC TRUONG" in all_norm or "GIAM DOC" in all_norm or "CANH SAT QUAN LY" in all_norm:
            back_score += 3
        if "I<VNM" in all_text or "<<" in all_text:
            back_score += 5
            
        return CardSide.BACK if back_score > front_score else CardSide.FRONT

    @classmethod
    def extract_fields(
        cls, detections: list[RawDetection], requested_side: CardSide = CardSide.AUTO
    ) -> Tuple[CardSide, dict[str, ExtractedField]]:
        """Extracts and validates structured CCCD fields from OCR detections."""
        effective_side = requested_side
        if effective_side == CardSide.AUTO:
            effective_side = cls.detect_side(detections)

        if effective_side == CardSide.FRONT:
            fields = cls._extract_front(detections)
        else:
            fields = cls._extract_back(detections)

        return effective_side, fields

    @classmethod
    def _extract_front(cls, detections: list[RawDetection]) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}
        
        # 1. ID Number (12 digits)
        id_field = cls._parse_id_number(detections)
        fields["idNumber"] = id_field
        fields["id_number"] = id_field

        # 2. Expiry date & identification of expiry detection indices
        expiry_field, expiry_indices = cls._parse_expiry_date(detections)
        fields["expiryDate"] = expiry_field
        fields["expiry_date"] = expiry_field

        # 3. Full Name
        name_field = cls._parse_full_name(detections)
        fields["fullName"] = name_field
        fields["full_name"] = name_field

        # 4. Date of Birth
        dob_field = cls._parse_date_of_birth(detections)
        fields["dateOfBirth"] = dob_field
        fields["date_of_birth"] = dob_field

        # 5. Gender
        gender_field = cls._parse_gender(detections)
        fields["gender"] = gender_field

        # 6. Nationality
        nationality_field = cls._parse_nationality(detections)
        fields["nationality"] = nationality_field

        # 7. Place of Origin (Quê quán)
        origin_field, origin_indices = cls._parse_place_of_origin(detections, expiry_indices)
        fields["placeOfOrigin"] = origin_field
        fields["place_of_origin"] = origin_field

        # 8. Place of Residence (Nơi thường trú)
        residence_field = cls._parse_place_of_residence(detections, expiry_indices, origin_indices)
        fields["placeOfResidence"] = residence_field
        fields["place_of_residence"] = residence_field

        return fields

    # -------------------------------------------------------------------------
    # FIELD PARSERS (FRONT SIDE)
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_id_number(detections: list[RawDetection]) -> ExtractedField:
        field = ExtractedField(
            name="idNumber",
            label="Số định danh cá nhân / CCCD",
            status=FieldStatus.MISSING,
        )

        candidate_id: Optional[str] = None
        candidate_conf: float = 0.0
        candidate_raw: str = ""

        for d in detections:
            text = d.text.strip()
            # 1. Match label-attached pattern (e.g. s6/No036205013225, No: 036205013225)
            m_label = re.search(r"(?:s[oóô60][\s/]*[nñ][o0][.:\s]*|[sS][oốô60][.:\s]*|[nN][o0][.:\s]*)(\d{12})", text, re.IGNORECASE)
            if m_label:
                candidate_id = m_label.group(1)
                candidate_conf = d.confidence
                candidate_raw = text
                break

            # 2. Standalone or embedded 12 digits
            m_digits = re.search(r"(?<!\d)(\d{12})(?!\d)", text)
            if m_digits:
                candidate_id = m_digits.group(1)
                candidate_conf = d.confidence
                candidate_raw = text
                break

            # 3. Spaced digits (e.g. 036 205 013 225)
            m_spaced = re.search(r"(?<!\d)(\d{3}[\s.-]\d{3}[\s.-]\d{6})(?!\d)", text)
            if m_spaced:
                clean = re.sub(r"[\s.-]", "", m_spaced.group(1))
                if len(clean) == 12:
                    candidate_id = clean
                    candidate_conf = d.confidence
                    candidate_raw = text
                    break

        if not candidate_id:
            field.status = FieldStatus.MISSING
            field.validation_message = "Không tìm thấy số CCCD 12 chữ số trong ảnh."
            return field

        field.value = candidate_id
        field.raw_text = candidate_raw
        field.confidence = candidate_conf
        field.extraction_confidence = 1.0

        # Validate CCCD 12 digits format and structure
        if len(candidate_id) != 12 or not candidate_id.isdigit():
            field.status = FieldStatus.INVALID
            field.validation_message = f"Độ dài số CCCD không hợp lệ ({len(candidate_id)} ký tự, yêu cầu đúng 12 số)."
            return field

        prov_code = candidate_id[:3]
        if prov_code not in VALID_PROVINCE_CODES:
            field.status = FieldStatus.NEEDS_REVIEW
            field.validation_message = f"Mã tỉnh/thành phố ({prov_code}) chưa nằm trong danh mục mã hành chính CCCD chuẩn."
            return field

        field.status = FieldStatus.EXTRACTED if candidate_conf >= 0.70 else FieldStatus.NEEDS_REVIEW
        return field

    @staticmethod
    def _parse_full_name(detections: list[RawDetection]) -> ExtractedField:
        field = ExtractedField(
            name="fullName",
            label="Họ và tên",
            status=FieldStatus.MISSING,
        )

        for i, d in enumerate(detections):
            norm = remove_accents(d.text.upper())
            if any(k in norm for k in ["HO VA TEN", "HOVA TEN", "FULL NAME"]):
                # Case 1: Name is on the same line after label
                cleaned = re.sub(
                    r"^(.*?)(?:h[oọ]\s*(?:v[aà]\s*)?t[eê]n|hova\s*ten|full\s*name)[:\s./-]*",
                    "",
                    d.text,
                    flags=re.IGNORECASE
                ).strip()
                cleaned = re.sub(r"[.·•*]$", "", cleaned).strip()

                if cleaned and len(cleaned) >= 3 and cleaned == cleaned.upper():
                    field.value = cleaned
                    field.status = FieldStatus.EXTRACTED if d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                    field.confidence = d.confidence
                    field.extraction_confidence = 0.95
                    field.raw_text = d.text
                    return field

                # Case 2: Name is on the immediate next line
                if i + 1 < len(detections):
                    nxt = detections[i + 1]
                    nxt_text = nxt.text.strip()
                    nxt_norm = remove_accents(nxt_text.upper())
                    
                    # Ensure next line is uppercase name and not another header
                    is_not_header = not any(k in nxt_norm for k in [
                        "NGAY SINH", "DATE OF BIRTH", "GIOI TINH", "SEX", "QUOC TICH", "SO ", "NO "
                    ])
                    # Must be mostly uppercase alphabetic
                    if is_not_header and nxt_text == nxt_text.upper() and len(nxt_text) >= 3:
                        field.value = nxt_text
                        field.status = FieldStatus.EXTRACTED if nxt.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                        field.confidence = nxt.confidence
                        field.extraction_confidence = 0.95
                        field.raw_text = nxt_text
                        return field

        return field

    @staticmethod
    def _parse_date_of_birth(detections: list[RawDetection]) -> ExtractedField:
        field = ExtractedField(
            name="dateOfBirth",
            label="Ngày sinh",
            status=FieldStatus.MISSING,
        )

        for i, d in enumerate(detections):
            norm = remove_accents(d.text.upper())
            if any(k in norm for k in ["NGAY SINH", "NGAYSINH", "DATE OF BIRTH", "DOB", "BIRTH"]):
                # Look for date pattern on same line (e.g. birth26/06/2005)
                m = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", d.text)
                target_d = d
                if not m and i + 1 < len(detections):
                    m = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", detections[i + 1].text)
                    target_d = detections[i + 1]

                if m:
                    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    formatted_date = f"{day:02d}/{month:02d}/{year:04d}"
                    field.raw_text = target_d.text
                    field.confidence = target_d.confidence
                    field.extraction_confidence = 0.95

                    # Calendar validity check
                    try:
                        parsed_dt = datetime(year, month, day)
                        cur_year = datetime.now().year
                        if 1900 <= year <= cur_year:
                            field.value = formatted_date
                            field.status = FieldStatus.EXTRACTED if target_d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                        else:
                            field.value = formatted_date
                            field.status = FieldStatus.INVALID
                            field.validation_message = f"Năm sinh ({year}) nằm ngoài phạm vi thông lệ (1900 - {cur_year})."
                    except ValueError as e:
                        field.value = formatted_date
                        field.status = FieldStatus.INVALID
                        field.validation_message = f"Ngày sinh không hợp lệ trong lịch: {str(e)}."
                    return field

        return field

    @staticmethod
    def _parse_gender(detections: list[RawDetection]) -> ExtractedField:
        field = ExtractedField(
            name="gender",
            label="Giới tính",
            status=FieldStatus.MISSING,
        )

        for d in detections:
            norm = remove_accents(d.text.upper())
            if "GIOI TINH" in norm or "GIOITINH" in norm or "SEX" in norm:
                field.raw_text = d.text
                field.confidence = d.confidence
                field.extraction_confidence = 0.95

                # Handles SexNam, Gioi tinh: Nam, Sex: Male, etc.
                m = re.search(r"(?:gi[oớ]i\s*t[ií]nh|sex)[\s/:]*(nam|n[ữu]|nu|female|male)", d.text, re.IGNORECASE)
                if m:
                    matched = m.group(1).lower()
                    field.value = "Nữ" if matched in ("nữ", "nu", "female") else "Nam"
                    field.status = FieldStatus.EXTRACTED if d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                    return field

                # Fallback search for standalone gender keywords in the line
                if re.search(r"\b(n[ữu]|female)\b", d.text, re.IGNORECASE):
                    field.value = "Nữ"
                    field.status = FieldStatus.EXTRACTED
                    return field
                elif re.search(r"\b(nam|male)\b", d.text, re.IGNORECASE):
                    field.value = "Nam"
                    field.status = FieldStatus.EXTRACTED
                    return field

        return field

    @staticmethod
    def _parse_nationality(detections: list[RawDetection]) -> ExtractedField:
        field = ExtractedField(
            name="nationality",
            label="Quốc tịch",
            status=FieldStatus.MISSING,
        )

        for d in detections:
            norm = remove_accents(d.text.upper())
            if "QUOC TICH" in norm or "QUOCTICH" in norm or "NATIONALITY" in norm:
                field.raw_text = d.text
                field.confidence = d.confidence
                field.extraction_confidence = 0.95

                # Handles NationalityViet Nam, Quoc tich: Viet Nam
                if re.search(r"vi[eệ]t\s*nam|vietnam|vnm", d.text, re.IGNORECASE):
                    field.value = "Việt Nam"
                    field.status = FieldStatus.EXTRACTED if d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                    return field

        return field

    @staticmethod
    def _parse_expiry_date(detections: list[RawDetection]) -> Tuple[ExtractedField, Set[int]]:
        field = ExtractedField(
            name="expiryDate",
            label="Có giá trị đến",
            status=FieldStatus.MISSING,
        )
        expiry_indices: Set[int] = set()

        for idx, d in enumerate(detections):
            norm = remove_accents(d.text.upper())
            # Match expiry keywords: Co giatrden, Có giá trị đến, Date of expiry, Dateofexpiry
            is_expiry_line = any(k in norm for k in [
                "CO GIA TRI", "GIA TRI DEN", "GIATRDEN", "DATE OF EXPIRY", "DATEOFEXPIRY", "EXPIRY", "VAL"
            ])
            if is_expiry_line:
                expiry_indices.add(idx)

                # Indefinite check
                if "KHONG THOI HAN" in norm or "INDEFINITE" in norm:
                    field.value = "Không thời hạn"
                    field.status = FieldStatus.EXTRACTED if d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                    field.confidence = d.confidence
                    field.extraction_confidence = 0.95
                    field.raw_text = d.text
                    continue

                # Date regex
                m = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", d.text)
                target_d = d
                if not m and idx + 1 < len(detections):
                    nxt = detections[idx + 1]
                    m_nxt = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", nxt.text)
                    if m_nxt:
                        m = m_nxt
                        target_d = nxt
                        expiry_indices.add(idx + 1)

                if m:
                    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    formatted_date = f"{day:02d}/{month:02d}/{year:04d}"
                    field.raw_text = target_d.text
                    field.confidence = target_d.confidence
                    field.extraction_confidence = 0.95

                    try:
                        parsed_dt = datetime(year, month, day)
                        if 1990 <= year <= 2150:
                            field.value = formatted_date
                            field.status = FieldStatus.EXTRACTED if target_d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                        else:
                            field.value = formatted_date
                            field.status = FieldStatus.INVALID
                            field.validation_message = f"Năm hết hạn ({year}) nằm ngoài phạm vi thông lệ."
                    except ValueError as e:
                        field.value = formatted_date
                        field.status = FieldStatus.INVALID
                        field.validation_message = f"Ngày hết hạn không hợp lệ trong lịch: {str(e)}."

        return field, expiry_indices

    @staticmethod
    def _parse_place_of_origin(detections: list[RawDetection], expiry_indices: Set[int]) -> Tuple[ExtractedField, Set[int]]:
        field = ExtractedField(
            name="placeOfOrigin",
            label="Quê quán",
            status=FieldStatus.MISSING,
        )
        origin_indices: Set[int] = set()

        for idx, d in enumerate(detections):
            if idx in expiry_indices:
                continue
            norm = remove_accents(d.text.upper())
            # Match Que quan, Quequan, Place of origin, onigin
            if any(k in norm for k in ["QUE QUAN", "QUEQUAN", "PLACE OF ORIGIN", "ONIGIN", "ORIGIN"]):
                origin_indices.add(idx)
                
                # Strip bilingual labels: Quequan / Place of onigin:
                cleaned = re.sub(
                    r"^(.*?)(?:qu[eêéèẻẽẹ\w]{1,5}\s*qu[aáàảãạ\w]{1,5}|quequan)[\s/]*(?:place\s*of\s*(?:origin|onigin|birth|binth))?[:\s./-]*",
                    "",
                    d.text,
                    flags=re.IGNORECASE
                ).strip()
                cleaned = re.sub(r"^(?:place\s*of\s*(?:origin|onigin|birth|binth))[:\s./-]*", "", cleaned, flags=re.IGNORECASE).strip()

                # Check if this line is purely a label
                norm_clean = remove_accents(cleaned.upper())
                clean_words = set(re.findall(r"\b\w+\b", norm_clean))
                label_words = {"QUE", "QUAN", "QUEQUAN", "PLACE", "OF", "ORIGIN", "ONIGIN", "BIRTH", "BINTH"}
                if not clean_words or clean_words.issubset(label_words):
                    cleaned = ""

                collected_parts = [cleaned] if cleaned else []

                # Look at next line if address continues and is not the residence header
                if idx + 1 < len(detections) and (idx + 1) not in expiry_indices:
                    nxt = detections[idx + 1]
                    nxt_norm = remove_accents(nxt.text.upper())
                    is_res_header = any(k in nxt_norm for k in [
                        "NOI THUONG TRU", "NOI THUNG TRU", "RESIDENCE", "CO GIA TRI", "GIATRDEN", "EXPIRY"
                    ])
                    if not is_res_header and len(nxt.text.strip()) >= 3:
                        collected_parts.append(nxt.text.strip())
                        origin_indices.add(idx + 1)

                val = ", ".join(filter(None, collected_parts))
                if val:
                    field.value = normalize_whitespace(val)
                    field.status = FieldStatus.EXTRACTED if d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                    field.confidence = d.confidence
                    field.extraction_confidence = 0.90
                    field.raw_text = val
                    break

        return field, origin_indices

    @staticmethod
    def _parse_place_of_residence(
        detections: list[RawDetection], expiry_indices: Set[int], origin_indices: Set[int]
    ) -> ExtractedField:
        field = ExtractedField(
            name="placeOfResidence",
            label="Nơi thường trú",
            status=FieldStatus.MISSING,
        )

        for idx, d in enumerate(detections):
            if idx in expiry_indices or idx in origin_indices:
                continue
            norm = remove_accents(d.text.upper())
            # Match Noi thuong tru, Noi thung trul, Place of residence
            if any(k in norm for k in ["NOI THUONG TRU", "THUONG TRU", "THUNG TRU", "PLACE OF RESIDENCE", "RESIDENCE"]):
                # Strip bilingual label prefix: Noi thung trul Place ofresidenceThön ThanhBinh
                cleaned = re.sub(
                    r"^(.*?)(?:n[oơöôóò\w]{1,4}i?\s*th[uưüúù\w]{1,4}[oơöôóò\w]{1,4}ng\s*tr[uúüù\w]{1,4}|noi\s*thung\s*trul?)[\s/]*(?:place\s*of\s*residence|residence)?[:\s./-]*",
                    "",
                    d.text,
                    flags=re.IGNORECASE
                ).strip()
                cleaned = re.sub(r"^(?:place\s*of\s*residence|residence)[:\s./-]*", "", cleaned, flags=re.IGNORECASE).strip()

                # Check if this line is purely a label
                norm_clean = remove_accents(cleaned.upper())
                clean_words = set(re.findall(r"[a-zA-Z0-9]+", norm_clean.replace("'", "")))
                label_words = {"NOI", "THUONG", "TRU", "THUNG", "TRUL", "PLACE", "OF", "RESIDENCE"}
                if not clean_words or clean_words.issubset(label_words):
                    cleaned = ""

                collected_parts = [cleaned] if cleaned else []

                # Scan following lines while ignoring expiry lines and headers
                for offset in range(1, 4):
                    next_idx = idx + offset
                    if next_idx >= len(detections) or next_idx in expiry_indices or next_idx in origin_indices:
                        continue
                    nxt = detections[next_idx]
                    nxt_norm = remove_accents(nxt.text.upper())

                    # Stop if encounter another header or date
                    if any(k in nxt_norm for k in ["CO GIA TRI", "GIA TRI DEN", "GIATRDEN", "EXPIRY", "CAN CUOC", "CITIZEN"]):
                        continue
                    if re.search(r"\b\d{2}/\d{2}/\d{4}\b", nxt.text):
                        continue

                    # Valid address continuation
                    if len(nxt.text.strip()) >= 3 and not re.search(r"^\d{12}$", nxt.text.strip()):
                        collected_parts.append(nxt.text.strip())

                val = ", ".join(filter(None, collected_parts))
                if val:
                    field.value = normalize_whitespace(val)
                    field.status = FieldStatus.EXTRACTED if d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                    field.confidence = d.confidence
                    field.extraction_confidence = 0.90
                    field.raw_text = val
                    break

        return field

    # -------------------------------------------------------------------------
    # FIELD PARSERS (BACK SIDE)
    # -------------------------------------------------------------------------

    @classmethod
    def _extract_back(cls, detections: list[RawDetection]) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}

        # 1. Personal characteristics
        features_field = ExtractedField(
            name="personalFeatures",
            label="Đặc điểm nhân dạng",
            status=FieldStatus.MISSING,
        )
        for i, d in enumerate(detections):
            norm = remove_accents(d.text.upper())
            if any(k in norm for k in ["DAC DIEM NHAN DANG", "NHAN DANG", "IDENTIFICATION"]):
                val = re.sub(
                    r"^(.*?)(?:đ[aặ]c\s*đi[eể]m\s*nh[aâ]n\s*d[aạ]ng|dac\s*diem\s*nhan\s*dang)[\s/]*(?:personal\s*identification|identification)?[:\s./-]*",
                    "",
                    d.text,
                    flags=re.IGNORECASE
                ).strip()
                val = re.sub(r"^(?:personal\s*identification|identification)[:\s./-]*", "", val, flags=re.IGNORECASE).strip()
                collected = [val] if val else []
                if i + 1 < len(detections):
                    nxt = detections[i + 1]
                    nxt_norm = remove_accents(nxt.text.upper())
                    if not any(k in nxt_norm for k in ["NGAY", "THANG", "NAM", "CUC TRUONG", "GIAM DOC", "I<VNM"]):
                        collected.append(nxt.text.strip())
                combined = " ".join(filter(None, collected))
                if combined:
                    features_field.value = normalize_whitespace(combined)
                    features_field.status = FieldStatus.EXTRACTED if d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                    features_field.confidence = d.confidence
                    features_field.raw_text = combined
                    break
        fields["personalFeatures"] = features_field
        fields["personal_features"] = features_field

        # 2. Date of Issue
        issue_date_field = ExtractedField(
            name="issueDate",
            label="Ngày cấp",
            status=FieldStatus.MISSING,
        )
        for d in detections:
            # Pattern: ngày DD tháng MM năm YYYY
            m = re.search(r"ng[aà]y\s*(\d{1,2})\s*th[aá]ng\s*(\d{1,2})\s*n[aă]m\s*(\d{4})", d.text, re.IGNORECASE)
            if m:
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                formatted = f"{day:02d}/{month:02d}/{year:04d}"
                issue_date_field.value = formatted
                issue_date_field.status = FieldStatus.EXTRACTED if d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                issue_date_field.confidence = d.confidence
                issue_date_field.raw_text = d.text
                break
            # Fallback DD/MM/YYYY
            m2 = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", d.text)
            norm = remove_accents(d.text.upper())
            if m2 and any(k in norm for k in ["NGAY", "CAP", "DATE", "ISSUE"]):
                day, month, year = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                formatted = f"{day:02d}/{month:02d}/{year:04d}"
                issue_date_field.value = formatted
                issue_date_field.status = FieldStatus.EXTRACTED if d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                issue_date_field.confidence = d.confidence
                issue_date_field.raw_text = d.text
                break
        fields["issueDate"] = issue_date_field
        fields["issue_date"] = issue_date_field

        # 3. Authority
        auth_field = ExtractedField(
            name="issueAuthority",
            label="Cơ quan cấp",
            status=FieldStatus.MISSING,
        )
        for d in detections:
            norm = remove_accents(d.text.upper())
            if any(k in norm for k in ["CUC TRUONG", "CANH SAT QUAN LY", "GIAM DOC CONG AN"]):
                auth_field.value = d.text.strip()
                auth_field.status = FieldStatus.EXTRACTED if d.confidence >= 0.70 else FieldStatus.NEEDS_REVIEW
                auth_field.confidence = d.confidence
                auth_field.raw_text = d.text
                break
        fields["issueAuthority"] = auth_field
        fields["issue_authority"] = auth_field

        # 4. MRZ
        mrz_field = ExtractedField(
            name="mrz",
            label="Mã vùng đọc máy (MRZ)",
            status=FieldStatus.MISSING,
        )
        mrz_lines = []
        for d in detections:
            cleaned = d.text.replace(" ", "").upper()
            if "<<" in cleaned or cleaned.startswith("I<VNM") or (len(cleaned) >= 28 and "<" in cleaned):
                mrz_lines.append(cleaned)
        if mrz_lines:
            mrz_field.value = "\n".join(mrz_lines)
            mrz_field.status = FieldStatus.EXTRACTED
            mrz_field.confidence = round(
                sum(d.confidence for d in detections if d.text in mrz_lines) / max(len(mrz_lines), 1),
                4
            )
            mrz_field.raw_text = "\n".join(mrz_lines)
        fields["mrz"] = mrz_field

        return fields
