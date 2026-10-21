from evaluate import calculate_levenshtein_distance, calculate_cer, BenchmarkEvaluator

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

def test_benchmark_evaluator_run():
    evaluator = BenchmarkEvaluator("tests/fixtures/ground_truth_sample.json")
    results = evaluator.run_evaluation()

    assert results["total_samples"] == 3
    assert "id_number" in results["field_metrics"]
    assert results["field_metrics"]["id_number"]["exact_match_rate"] == 1.0
    assert results["field_metrics"]["full_name"]["exact_match_rate"] == 1.0
    assert "high_quality_studio" in results["quality_categories"]
