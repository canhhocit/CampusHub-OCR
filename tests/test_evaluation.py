import os
from evaluate import (
    calculate_levenshtein_distance,
    calculate_cer,
    normalize_for_evaluation,
    BenchmarkEvaluator,
)

def test_levenshtein_distance():
    assert calculate_levenshtein_distance("kitten", "sitting") == 3
    assert calculate_levenshtein_distance("012345678912", "012345678912") == 0
    assert calculate_levenshtein_distance("", "test") == 4
    assert calculate_levenshtein_distance("abc", "") == 3

def test_cer_calculation():
    # Exact match -> CER = 0.0
    assert calculate_cer("NGUYỄN VĂN AN", "NGUYỄN VĂN AN") == 0.0
    # 1 character error in 10-char string -> CER = 0.1
    assert calculate_cer("1234567890", "123456789X") == 0.1
    # Empty reference
    assert calculate_cer("", "") == 0.0

def test_normalize_for_evaluation():
    # 1. Trims cosmetic trailing punctuation
    assert normalize_for_evaluation("Hà Nội,") == "hà nội"
    assert normalize_for_evaluation("Nam Định.") == "nam định"
    assert normalize_for_evaluation("26/06/2005:") == "26/06/2005"
    
    # 2. Collapses excessive whitespace
    assert normalize_for_evaluation("NGUYỄN    VĂN   AN") == "nguyễn văn an"
    
    # 3. Preserves digits, slashes, and Vietnamese diacritics
    assert normalize_for_evaluation("036205013225") == "036205013225"
    assert normalize_for_evaluation("26/06/2030") == "26/06/2030"
    assert normalize_for_evaluation("Phường Bến Nghé") == "phường bến nghé"

def test_benchmark_evaluator_synthetic_mode():
    evaluator = BenchmarkEvaluator("tests/fixtures/ground_truth_sample.json", mode="synthetic")
    results = evaluator.run_evaluation(preprocess=True)

    assert results["total_samples"] == 7
    assert results["synthetic_count"] == 7
    assert results["image_count"] == 0
    assert results["overall"]["exact_match_rate"] == 1.0  # 100% on synthetic test suite
    assert results["overall"]["missing_rate"] == 0.0
    assert results["overall"]["invalid_rate"] == 0.0

    # Field metrics checks
    assert "id_number" in results["field_metrics"]
    assert results["field_metrics"]["id_number"]["exact_match_rate"] == 1.0
    assert results["field_metrics"]["full_name"]["exact_match_rate"] == 1.0
    assert results["field_metrics"]["date_of_birth"]["exact_match_rate"] == 1.0
    assert results["field_metrics"]["expiry_date"]["exact_match_rate"] == 1.0

    # Latency tracking
    assert "latency_stats" in results
    assert results["latency_stats"]["extraction_mean_ms"] >= 0.0

def test_benchmark_evaluator_error_analysis():
    # Test error categorization on an artificial error
    sample = {
        "id": "TEST_ERR_SAMPLE",
        "data_type": "synthetic",
        "side": "FRONT",
        "ground_truth": {
            "id_number": "001200000001"
        },
        "expected_statuses": {
            "id_number": "EXTRACTED"
        },
        "ocr_detections": [
            {"text": "CAN CUOC CONG DAN", "confidence": 0.99},
            {"text": "So: 999999999999", "confidence": 0.99}  # Mismatched ID
        ]
    }
    
    evaluator = BenchmarkEvaluator("tests/fixtures/ground_truth_sample.json", mode="synthetic")
    res = evaluator.evaluate_sample(sample)
    
    assert res["sample_passed"] is False
    assert res["fields"]["id_number"]["exact_match"] is False
    assert res["fields"]["id_number"]["error_cause"] == "EXTRACTION_VALUE_MISMATCH"

def test_benchmark_evaluator_image_mode():
    img_path = "tests/fixtures/images/specimen_cccd_front.png"
    if not os.path.exists(img_path):
        from tests.generate_specimen_card import generate_specimen_front_card
        generate_specimen_front_card(img_path)

    evaluator = BenchmarkEvaluator("tests/fixtures/ground_truth_sample.json", mode="image")
    results = evaluator.run_evaluation(preprocess=True)

    assert results["image_count"] == 1
    assert results["overall"]["exact_match_fields"] >= 5
    assert results["field_metrics"]["id_number"]["exact_match_rate"] == 1.0
    assert results["field_metrics"]["date_of_birth"]["exact_match_rate"] == 1.0
    assert results["field_metrics"]["gender"]["exact_match_rate"] == 1.0
    assert results["field_metrics"]["nationality"]["exact_match_rate"] == 1.0
    assert results["field_metrics"]["expiry_date"]["exact_match_rate"] == 1.0
    assert results["latency_stats"]["inference_mean_ms"] > 0
    assert results["model_init_time_ms"] >= 0
