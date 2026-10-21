# Ground-Truth Dataset Schema & Documentation for CampusHub-OCR

## 1. Overview

This document specifies the format, schema, and guidelines for benchmark and evaluation datasets used in `CampusHub-OCR`.

The evaluation framework supports two modes:
1. **Synthetic OCR Detections (`data_type: "synthetic"`)**: Fast, 100% deterministic evaluation of field extraction heuristics, boundary parsing, regex rules, and edge-case validation.
2. **End-to-End Image Evaluation (`data_type: "image"`)**: End-to-end evaluation executing the full pipeline: image validation -> preprocessing (optional) -> PaddleOCR detection/recognition -> CCCD field extraction.

---

## 2. Privacy & Compliance Rules

- **STRICT PROHIBITION**: Never commit, upload, log, or persist real Vietnamese CCCD images or genuine citizen personal data.
- All evaluation samples must use either:
  1. Fully synthetic mock detections with fictitious test data.
  2. Legally safe specimen cards (*Thẻ Căn cước mẫu / Specimen*) published officially by authorities or synthetically rendered cards clearly stamped `SPECIMEN / THẺ THỬ NGHIỆM`.

---

## 3. Dataset Schema (JSON)

An evaluation dataset is a JSON array of sample objects. Each sample conforms to the following schema:

```json
[
  {
    "id": "SYNTH_CCCD_FRONT_01",
    "data_type": "synthetic",
    "quality_category": "high_quality_studio",
    "side": "FRONT",
    "image_path": null,
    "ground_truth": {
      "id_number": "001200000001",
      "full_name": "NGUYỄN VĂN AN",
      "date_of_birth": "15/08/2002",
      "gender": "Nam",
      "nationality": "Việt Nam",
      "place_of_origin": "Xã Hải Tân, Huyện Hải Hậu, Tỉnh Nam Định",
      "place_of_residence": "Phòng 302 KTX Đại học Bách Khoa, Hai Bà Trưng, Hà Nội",
      "expiry_date": "15/08/2027"
    },
    "expected_statuses": {
      "id_number": "EXTRACTED",
      "full_name": "EXTRACTED"
    },
    "ground_truth_text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\nCĂN CƯỚC CÔNG DÂN\nSố / No.: 001200000001\n...",
    "ocr_detections": [
      {
        "text": "Số / No.: 001200000001",
        "confidence": 0.99,
        "box": [[100.0, 150.0], [300.0, 150.0], [300.0, 180.0], [100.0, 180.0]]
      }
    ]
  }
]
```

### Field Definitions

| Field Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | `string` | **Yes** | Unique identifier for the sample (e.g. `SYNTH_CCCD_FRONT_01`, `IMG_SPECIMEN_01`). |
| `data_type` | `string` | **Yes** | Either `"synthetic"` (uses `ocr_detections`) or `"image"` (reads `image_path`). |
| `quality_category` | `string` | **Yes** | Visual / textual quality category (see Section 4). |
| `side` | `string` | **Yes** | Card side: `"FRONT"` or `"BACK"`. |
| `image_path` | `string` or `null` | Optional | Relative path from project root to the test image file (required if `data_type="image"`). |
| `ground_truth` | `object` | **Yes** | Key-value mapping of expected structured fields (see Section 5). |
| `expected_statuses` | `object` | Optional | Map of field name -> expected `FieldStatus` (`"EXTRACTED"`, `"NEEDS_REVIEW"`, `"MISSING"`, `"INVALID"`). |
| `ground_truth_text` | `string` | Optional | Complete reference transcription of the card text, used for Character Error Rate (CER). |
| `ocr_detections` | `array` | Optional | List of pre-computed OCR detections (required if `data_type="synthetic"`). |

---

## 4. Quality Categories

Samples are tagged with a `quality_category` to enable fine-grained performance breakdown:

- `high_quality_studio`: Clear, flat scans or well-lit front-facing photographs with no glare or perspective distortion.
- `natural_mobile`: Typical mobile camera captures with mild rotation, shadows, or natural lighting variations.
- `challenging_concatenated`: OCR output where labels and values are merged without delimiters (e.g. `s6/No036...`, `birth26/06/...`, `SexNam`).
- `multiline_dense`: Cards with multi-line origin or residence fields that test line-wrapping heuristics.
- `malformed_boundary`: Negative test cases containing missing fields, invalid calendar dates (e.g. 31/02), or truncated ID numbers.

---

## 5. Supported Field Names

### Front Side (`side: "FRONT"`)
- `id_number` (or `idNumber`): 12-digit citizen identification number.
- `full_name` (or `fullName`): Full name in uppercase Vietnamese.
- `date_of_birth` (or `dateOfBirth`): Date in `DD/MM/YYYY` format.
- `gender`: `"Nam"` or `"Nữ"`.
- `nationality`: Standardized to `"Việt Nam"`.
- `place_of_origin` (or `placeOfOrigin`): Native place / Quê quán.
- `place_of_residence` (or `placeOfResidence`): Registered residence / Nơi thường trú.
- `expiry_date` (or `expiryDate`): Date in `DD/MM/YYYY` format or `"Không thời hạn"`.

### Back Side (`side: "BACK"`)
- `personal_features` (or `personalFeatures`): Personal identification marks.
- `issue_date` (or `issueDate`): Issue date in `DD/MM/YYYY` format.
- `issue_authority` (or `issueAuthority`): Issuing authority title.
- `mrz`: Machine Readable Zone text (2-3 lines).

---

## 6. Normalization Rules for Evaluation

To ensure objective and reproducible comparisons, ground truth and extracted values undergo standardized evaluation normalization before metric computation:

1. **Unicode NFC Canonicalization**: Normalize all text to Unicode NFC (`unicodedata.normalize("NFC", text)`), resolving precomposed vs decomposed Vietnamese diacritic discrepancies.
2. **Whitespace Normalization**: Trim leading and trailing whitespace; collapse consecutive internal spaces, tabs, and newlines into a single space.
3. **Punctuation Trimming**: Strip trailing noise punctuation (such as trailing `.`, `,`, `:`, `;`, `-`) commonly produced by OCR bounding-box edge noise.
4. **Case Folding for Free Text**: Exact match comparisons for name, address, and place of origin perform case-insensitive comparison (`s.lower()`) to avoid penalizing minor font-case quirks while strictly checking character content.
5. **Preservation of Digits and Dates**: Digits, slashes (`/`), and substantive characters are strictly preserved. The evaluation pipeline does **not** silently remove or swap digits to mask extraction errors.
