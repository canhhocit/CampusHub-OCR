#!/usr/bin/env python3
"""
CampusHub-OCR Benchmark & Evaluation Suite.
Measures Character Error Rate (CER), Field-Level Exact Match (EM), 
Processing Latency, and Categorical Quality Success Rates.
"""

import json
import time
import argparse
import sys
from typing import Dict, Any, List, Tuple
from app.schemas.ocr import RawDetection, CardSide, FieldStatus
from app.services.cccd_extractor import CCCDExtractor

# Ensure clean UTF-8 printing on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Computes Character Error Rate (CER):
    CER = Levenshtein_Distance(reference, hypothesis) / len(reference)
    """
    ref_clean = reference.strip()
    hyp_clean = hypothesis.strip()
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    dist = calculate_levenshtein_distance(ref_clean, hyp_clean)
    return min(dist / len(ref_clean), 1.0)

class BenchmarkEvaluator:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def run_evaluation(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "total_samples": len(self.data),
            "field_metrics": {},
            "quality_categories": {},
            "latencies_ms": [],
        }

        field_stats: Dict[str, Dict[str, Any]] = {}
        category_stats: Dict[str, Dict[str, int]] = {}

        for sample in self.data:
            sample_id = sample.get("id", "unknown")
            category = sample.get("quality_category", "standard")
            ground_truth = sample.get("ground_truth", {})
            synthetic_detections_data = sample.get("ocr_detections", [])
            side = CardSide(sample.get("side", "FRONT"))

            # Convert to RawDetection objects
            detections = [
                RawDetection(
                    text=item["text"],
                    confidence=item.get("confidence", 0.95),
                    box=item.get("box", None),
                )
                for item in synthetic_detections_data
            ]

            # Measure extraction latency
            t0 = time.perf_counter()
            _, extracted_fields = CCCDExtractor.extract_fields(detections, requested_side=side)
            t_ms = (time.perf_counter() - t0) * 1000
            results["latencies_ms"].append(t_ms)

            # Evaluate each expected ground-truth field
            sample_success = True
            for field_name, expected_val in ground_truth.items():
                if field_name not in field_stats:
                    field_stats[field_name] = {
                        "total": 0,
                        "exact_match": 0,
                        "total_cer": 0.0,
                        "detected_count": 0,
                    }

                field_stats[field_name]["total"] += 1
                extracted = extracted_fields.get(field_name)
                actual_val = extracted.value if (extracted and extracted.status == FieldStatus.VALID) else ""

                if actual_val:
                    field_stats[field_name]["detected_count"] += 1

                # Exact Match check (normalized case and spacing)
                is_em = actual_val.strip().lower() == expected_val.strip().lower()
                if is_em:
                    field_stats[field_name]["exact_match"] += 1
                else:
                    sample_success = False

                # CER
                cer = calculate_cer(expected_val, actual_val)
                field_stats[field_name]["total_cer"] += cer

            # Track Category Success
            if category not in category_stats:
                category_stats[category] = {"total": 0, "passed": 0}
            category_stats[category]["total"] += 1
            if sample_success:
                category_stats[category]["passed"] += 1

        # Summarize Field Metrics
        for f_name, stats in field_stats.items():
            tot = stats["total"]
            results["field_metrics"][f_name] = {
                "samples": tot,
                "exact_match_rate": round(stats["exact_match"] / tot, 4) if tot > 0 else 0.0,
                "detection_rate": round(stats["detected_count"] / tot, 4) if tot > 0 else 0.0,
                "average_cer": round(stats["total_cer"] / tot, 4) if tot > 0 else 0.0,
            }

        # Summarize Category Metrics
        for cat, stats in category_stats.items():
            tot = stats["total"]
            results["quality_categories"][cat] = {
                "total": tot,
                "passed": stats["passed"],
                "success_rate": round(stats["passed"] / tot, 4) if tot > 0 else 0.0,
            }

        return results

    def print_report(self, results: Dict[str, Any]):
        print("\n" + "=" * 70)
        print("          CAMPUSHUB-OCR EVALUATION & BENCHMARK REPORT          ")
        print("=" * 70)
        print(f"Tổng số mẫu đánh giá: {results['total_samples']}")
        if results['latencies_ms']:
            avg_lat = sum(results['latencies_ms']) / len(results['latencies_ms'])
            print(f"Độ trễ trung bình trích xuất: {avg_lat:.2f} ms")
        print("-" * 70)

        print(f"{'Trường thông tin':<22} | {'Số mẫu':<8} | {'Exact Match':<12} | {'Detection':<10} | {'Avg CER':<8}")
        print("-" * 70)
        for field, m in results["field_metrics"].items():
            em_pct = f"{m['exact_match_rate'] * 100:.1f}%"
            det_pct = f"{m['detection_rate'] * 100:.1f}%"
            cer_pct = f"{m['average_cer'] * 100:.1f}%"
            print(f"{field:<22} | {m['samples']:<8} | {em_pct:<12} | {det_pct:<10} | {cer_pct:<8}")
        print("-" * 70)

        print("\nHiệu suất theo nhóm chất lượng hình ảnh:")
        print(f"{'Nhóm chất lượng':<25} | {'Số mẫu':<8} | {'Thành công':<12} | {'Tỷ lệ':<8}")
        print("-" * 70)
        for cat, m in results["quality_categories"].items():
            rate_pct = f"{m['success_rate'] * 100:.1f}%"
            print(f"{cat:<25} | {m['total']:<8} | {m['passed']:<12} | {rate_pct:<8}")
        print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CampusHub-OCR Benchmark Runner")
    parser.add_argument(
        "--dataset",
        default="tests/fixtures/ground_truth_sample.json",
        help="Đường dẫn đến tệp JSON ground-truth benchmark"
    )
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(args.dataset)
    metrics = evaluator.run_evaluation()
    evaluator.print_report(metrics)
