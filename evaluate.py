#!/usr/bin/env python3
"""
CampusHub-OCR Benchmark & Evaluation Suite.
Measures:
- Field-level & overall Exact-Match (EM) accuracy
- Missing-field rate and Invalid-field rate
- Character Error Rate (CER) for field text & card transcriptions
- Processing latencies (mean, median, min, max), isolating model init vs per-image inference
- Preprocessing comparison (ENABLED vs DISABLED)
- Detailed error analysis with root-cause categorization
"""

import os
import sys
import json
import time
import re
import unicodedata
import statistics
import argparse
from typing import Dict, Any, List, Optional, Tuple, Set

from app.schemas.ocr import RawDetection, CardSide, FieldStatus, ExtractedField
from app.services.cccd_extractor import CCCDExtractor
from app.services.image_processor import ImageProcessor
from app.services.ocr_engine import get_ocr_engine, BaseOCREngine

# Ensure clean UTF-8 printing on Windows console
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
    CER = Levenshtein_Distance(reference, hypothesis) / max(len(reference), 1)
    Capped at 1.0 (100%).
    """
    ref_clean = reference.strip()
    hyp_clean = hypothesis.strip()
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    dist = calculate_levenshtein_distance(ref_clean, hyp_clean)
    return min(dist / len(ref_clean), 1.0)

def normalize_for_evaluation(value: Optional[str]) -> str:
    """
    Standardized normalization rules for objective evaluation:
    1. Unicode NFC canonicalization
    2. Collapse all whitespaces/tabs/newlines
    3. Strip cosmetic trailing edge punctuation (. , : ; -)
    4. Lowercase for case-insensitive content match
    Preserves all numbers, letters, slashes, and diacritics.
    """
    if not value:
        return ""
    nfc = unicodedata.normalize("NFC", str(value))
    collapsed = re.sub(r"\s+", " ", nfc).strip()
    cleaned = re.sub(r"[\s,;.:-]+$", "", collapsed)
    return cleaned.lower()

class BenchmarkEvaluator:
    def __init__(self, dataset_path: str, mode: str = "all", compare_preprocessing: bool = False):
        self.dataset_path = dataset_path
        self.mode = mode.lower()  # "synthetic", "image", or "all"
        self.compare_preprocessing = compare_preprocessing
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
            
        # Filter samples based on requested mode
        if self.mode == "synthetic":
            self.samples = [s for s in self.raw_data if s.get("data_type", "synthetic") == "synthetic"]
        elif self.mode == "image":
            self.samples = [s for s in self.raw_data if s.get("data_type") == "image"]
        else:
            self.samples = self.raw_data

        self.ocr_engine: Optional[BaseOCREngine] = None
        self.model_init_time_ms: float = 0.0

    def _ensure_engine_loaded(self):
        """Loads and measures OCR engine cold start initialization latency."""
        if self.ocr_engine is None:
            t0 = time.perf_counter()
            self.ocr_engine = get_ocr_engine()
            # Force lazy weights load if PaddleOCR
            if hasattr(self.ocr_engine, "_get_ocr"):
                self.ocr_engine._get_ocr()
            self.model_init_time_ms = (time.perf_counter() - t0) * 1000

    def evaluate_sample(
        self, sample: Dict[str, Any], preprocess: bool = True
    ) -> Dict[str, Any]:
        """Evaluates a single sample through the extraction pipeline."""
        sample_id = sample.get("id", "unknown")
        data_type = sample.get("data_type", "synthetic")
        side = CardSide(sample.get("side", "FRONT"))
        ground_truth = sample.get("ground_truth", {})
        expected_statuses = sample.get("expected_statuses", {})
        ref_text = sample.get("ground_truth_text", "")
        
        inf_ms = 0.0
        detections: List[RawDetection] = []

        if data_type == "image":
            img_rel_path = sample.get("image_path")
            if not img_rel_path or not os.path.exists(img_rel_path):
                raise FileNotFoundError(f"Image not found at path: {img_rel_path} for sample {sample_id}")
            
            with open(img_rel_path, "rb") as f:
                img_bytes = f.read()

            self._ensure_engine_loaded()
            raw_image, _ = ImageProcessor.validate_image_bytes(img_bytes, filename=os.path.basename(img_rel_path))
            
            processed_img = ImageProcessor.preprocess_for_ocr(raw_image) if preprocess else raw_image
            
            t_inf0 = time.perf_counter()
            detections = self.ocr_engine.detect_and_recognize(processed_img)
            inf_ms = (time.perf_counter() - t_inf0) * 1000
        else:
            # Synthetic pre-computed detections
            detections = [
                RawDetection(
                    text=item["text"],
                    confidence=item.get("confidence", 0.95),
                    box=item.get("box", None),
                )
                for item in sample.get("ocr_detections", [])
            ]

        # Field extraction
        t_ext0 = time.perf_counter()
        effective_side, extracted_fields = CCCDExtractor.extract_fields(detections, requested_side=side)
        ext_ms = (time.perf_counter() - t_ext0) * 1000

        # Card-level text CER (if reference text provided)
        hyp_text = "\n".join(d.text.strip() for d in detections)
        card_cer = calculate_cer(ref_text, hyp_text) if ref_text else 0.0

        # Field evaluation
        field_evaluations: Dict[str, Any] = {}
        sample_all_matched = True

        # Evaluate expected fields from ground_truth
        for field_key, expected_val in ground_truth.items():
            extracted: Optional[ExtractedField] = (
                extracted_fields.get(field_key) or 
                extracted_fields.get(self._to_camel(field_key)) or
                extracted_fields.get(self._to_snake(field_key))
            )

            actual_val = extracted.value if extracted else None
            actual_status = extracted.status if extracted else FieldStatus.MISSING
            
            expected_status_str = expected_statuses.get(field_key, "EXTRACTED")
            
            norm_exp = normalize_for_evaluation(expected_val)
            norm_act = normalize_for_evaluation(actual_val)

            # Exact match check
            is_exact_match = (norm_exp == norm_act) and (actual_status not in (FieldStatus.MISSING, FieldStatus.INVALID))
            if expected_val is None:
                # If ground truth explicitly expects missing/null value
                is_exact_match = actual_val is None or actual_status == FieldStatus.MISSING

            if not is_exact_match:
                sample_all_matched = False

            # Status match check
            status_matched = (actual_status.value == expected_status_str)

            # Field-level CER
            field_cer = calculate_cer(norm_exp, norm_act) if norm_exp else (0.0 if not norm_act else 1.0)

            # Error cause categorization
            error_cause = None
            if not is_exact_match:
                if actual_status == FieldStatus.INVALID:
                    error_cause = "VALIDATION_FAILURE"
                elif not actual_val or actual_status == FieldStatus.MISSING:
                    # Check if expected text existed in raw OCR detections
                    found_in_ocr = any(norm_exp in normalize_for_evaluation(d.text) for d in detections)
                    if not found_in_ocr:
                        has_partial = any(calculate_cer(norm_exp, normalize_for_evaluation(d.text)) < 0.40 for d in detections)
                        error_cause = "OCR_RECOGNITION_ERROR" if has_partial else "MISSING_IN_OCR"
                    else:
                        error_cause = "EXTRACTION_LABEL_MISMATCH"
                else:
                    if field_cer < 0.20:
                        error_cause = "NORMALIZATION_MISMATCH"
                    else:
                        error_cause = "EXTRACTION_VALUE_MISMATCH"

            field_evaluations[field_key] = {
                "expected": expected_val,
                "actual": actual_val,
                "status": actual_status.value,
                "expected_status": expected_status_str,
                "status_matched": status_matched,
                "exact_match": is_exact_match,
                "cer": field_cer,
                "error_cause": error_cause,
            }

        return {
            "id": sample_id,
            "data_type": data_type,
            "quality_category": sample.get("quality_category", "standard"),
            "inference_ms": inf_ms,
            "extraction_ms": ext_ms,
            "total_latency_ms": inf_ms + ext_ms,
            "card_cer": card_cer,
            "sample_passed": sample_all_matched,
            "fields": field_evaluations,
            "raw_detections_count": len(detections),
        }

    def run_evaluation(self, preprocess: bool = True) -> Dict[str, Any]:
        """Runs evaluation over all selected samples."""
        results: Dict[str, Any] = {
            "dataset_path": self.dataset_path,
            "mode": self.mode,
            "preprocess_enabled": preprocess,
            "total_samples": len(self.samples),
            "synthetic_count": sum(1 for s in self.samples if s.get("data_type", "synthetic") == "synthetic"),
            "image_count": sum(1 for s in self.samples if s.get("data_type") == "image"),
            "model_init_time_ms": 0.0,
            "field_metrics": {},
            "quality_categories": {},
            "error_analysis": [],
            "latency_stats": {},
            "card_cer_list": [],
            "overall": {
                "total_fields": 0,
                "exact_match_fields": 0,
                "exact_match_rate": 0.0,
                "missing_fields": 0,
                "missing_rate": 0.0,
                "invalid_fields": 0,
                "invalid_rate": 0.0,
                "average_cer": 0.0,
            }
        }

        field_accum: Dict[str, Dict[str, Any]] = {}
        cat_accum: Dict[str, Dict[str, int]] = {}
        latencies_inf: List[float] = []
        latencies_ext: List[float] = []
        latencies_tot: List[float] = []
        cer_sum = 0.0

        for sample in self.samples:
            eval_res = self.evaluate_sample(sample, preprocess=preprocess)
            
            # Latencies
            if eval_res["inference_ms"] > 0:
                latencies_inf.append(eval_res["inference_ms"])
            latencies_ext.append(eval_res["extraction_ms"])
            latencies_tot.append(eval_res["total_latency_ms"])
            
            if eval_res["card_cer"] > 0:
                results["card_cer_list"].append(eval_res["card_cer"])

            cat = eval_res["quality_category"]
            if cat not in cat_accum:
                cat_accum[cat] = {"total": 0, "passed": 0}
            cat_accum[cat]["total"] += 1
            if eval_res["sample_passed"]:
                cat_accum[cat]["passed"] += 1

            for f_key, f_eval in eval_res["fields"].items():
                if f_key not in field_accum:
                    field_accum[f_key] = {
                        "total": 0,
                        "exact_match": 0,
                        "missing": 0,
                        "invalid": 0,
                        "cer_sum": 0.0,
                    }

                acc = field_accum[f_key]
                acc["total"] += 1
                results["overall"]["total_fields"] += 1
                cer_sum += f_eval["cer"]

                if f_eval["exact_match"]:
                    acc["exact_match"] += 1
                    results["overall"]["exact_match_fields"] += 1
                else:
                    results["error_analysis"].append({
                        "sample_id": eval_res["id"],
                        "field": f_key,
                        "expected": f_eval["expected"],
                        "actual": f_eval["actual"],
                        "status": f_eval["status"],
                        "cer": round(f_eval["cer"], 4),
                        "cause": f_eval["error_cause"],
                    })

                if f_eval["status"] == FieldStatus.MISSING.value or f_eval["actual"] is None:
                    acc["missing"] += 1
                    results["overall"]["missing_fields"] += 1
                elif f_eval["status"] == FieldStatus.INVALID.value:
                    acc["invalid"] += 1
                    results["overall"]["invalid_fields"] += 1

                acc["cer_sum"] += f_eval["cer"]

        # Aggregate Overall Metrics
        tot_f = results["overall"]["total_fields"]
        if tot_f > 0:
            results["overall"]["exact_match_rate"] = round(results["overall"]["exact_match_fields"] / tot_f, 4)
            results["overall"]["missing_rate"] = round(results["overall"]["missing_fields"] / tot_f, 4)
            results["overall"]["invalid_rate"] = round(results["overall"]["invalid_fields"] / tot_f, 4)
            results["overall"]["average_cer"] = round(cer_sum / tot_f, 4)

        # Aggregate Field Metrics
        for f_name, acc in field_accum.items():
            tot = acc["total"]
            results["field_metrics"][f_name] = {
                "samples": tot,
                "exact_match_count": acc["exact_match"],
                "exact_match_rate": round(acc["exact_match"] / tot, 4) if tot > 0 else 0.0,
                "missing_rate": round(acc["missing"] / tot, 4) if tot > 0 else 0.0,
                "invalid_rate": round(acc["invalid"] / tot, 4) if tot > 0 else 0.0,
                "average_cer": round(acc["cer_sum"] / tot, 4) if tot > 0 else 0.0,
            }

        # Aggregate Quality Categories
        for cat, acc in cat_accum.items():
            tot = acc["total"]
            results["quality_categories"][cat] = {
                "total": tot,
                "passed": acc["passed"],
                "success_rate": round(acc["passed"] / tot, 4) if tot > 0 else 0.0,
            }

        # Aggregate Latencies
        results["model_init_time_ms"] = round(self.model_init_time_ms, 2)
        results["latency_stats"] = {
            "extraction_mean_ms": round(statistics.mean(latencies_ext), 2) if latencies_ext else 0.0,
            "extraction_median_ms": round(statistics.median(latencies_ext), 2) if latencies_ext else 0.0,
            "inference_mean_ms": round(statistics.mean(latencies_inf), 2) if latencies_inf else 0.0,
            "inference_median_ms": round(statistics.median(latencies_inf), 2) if latencies_inf else 0.0,
            "total_mean_ms": round(statistics.mean(latencies_tot), 2) if latencies_tot else 0.0,
            "total_median_ms": round(statistics.median(latencies_tot), 2) if latencies_tot else 0.0,
            "total_min_ms": round(min(latencies_tot), 2) if latencies_tot else 0.0,
            "total_max_ms": round(max(latencies_tot), 2) if latencies_tot else 0.0,
        }

        return results

    def print_report(self, res: Dict[str, Any], title_suffix: str = ""):
        print("\n" + "=" * 80)
        print(f"        CAMPUSHUB-OCR BENCHMARK & EVALUATION REPORT {title_suffix}".center(80))
        print("=" * 80)
        print(f"Dataset File        : {res['dataset_path']}")
        print(f"Evaluation Mode     : {res['mode'].upper()} (Synthetic: {res['synthetic_count']}, Real/Image: {res['image_count']})")
        print(f"Image Preprocessing : {'ENABLED' if res['preprocess_enabled'] else 'DISABLED'}")
        print(f"Model Init Time     : {res['model_init_time_ms']:.2f} ms")
        print("-" * 80)

        # Latency Summary
        l = res["latency_stats"]
        print(f"Processing Latency:")
        print(f"  - Extraction (Heuristics) : Mean: {l['extraction_mean_ms']} ms | Median: {l['extraction_median_ms']} ms")
        if res["image_count"] > 0:
            print(f"  - OCR Inference (Paddle)  : Mean: {l['inference_mean_ms']} ms | Median: {l['inference_median_ms']} ms")
        print(f"  - Total Pipeline Latency  : Mean: {l['total_mean_ms']} ms | Median: {l['total_median_ms']} ms (Min: {l['total_min_ms']}, Max: {l['total_max_ms']})")
        print("-" * 80)

        # Overall Summary Table
        o = res["overall"]
        print(f"OVERALL SUMMARY ({o['total_fields']} field instances evaluated):")
        print(f"  - Exact-Match Accuracy : {o['exact_match_rate'] * 100:.2f}% ({o['exact_match_fields']}/{o['total_fields']})")
        print(f"  - Missing-Field Rate   : {o['missing_rate'] * 100:.2f}% ({o['missing_fields']}/{o['total_fields']})")
        print(f"  - Invalid-Field Rate   : {o['invalid_rate'] * 100:.2f}% ({o['invalid_fields']}/{o['total_fields']})")
        print(f"  - Average CER          : {o['average_cer'] * 100:.2f}%")
        print("-" * 80)

        # Field-Level Breakdown
        print(f"{'Field Name':<22} | {'Samples':<8} | {'Exact Match':<12} | {'Missing':<9} | {'Invalid':<9} | {'Avg CER':<8}")
        print("-" * 80)
        for f_name, m in res["field_metrics"].items():
            em_s = f"{m['exact_match_rate'] * 100:.1f}%"
            mis_s = f"{m['missing_rate'] * 100:.1f}%"
            inv_s = f"{m['invalid_rate'] * 100:.1f}%"
            cer_s = f"{m['average_cer'] * 100:.1f}%"
            print(f"{f_name:<22} | {m['samples']:<8} | {em_s:<12} | {mis_s:<9} | {inv_s:<9} | {cer_s:<8}")
        print("-" * 80)

        # Quality Categories
        print("\nQuality Categories Breakdown:")
        print(f"{'Category':<28} | {'Total':<6} | {'Passed':<8} | {'Success Rate':<12}")
        print("-" * 80)
        for cat, c in res["quality_categories"].items():
            rate_s = f"{c['success_rate'] * 100:.1f}%"
            print(f"{cat:<28} | {c['total']:<6} | {c['passed']:<8} | {rate_s:<12}")
        print("-" * 80)

        # Error Analysis Section
        errs = res.get("error_analysis", [])
        if errs:
            print(f"\nERROR ANALYSIS & ROOT CAUSE BREAKDOWN ({len(errs)} failures):")
            print(f"{'Sample ID':<26} | {'Field':<18} | {'CER':<6} | {'Cause':<22}")
            print("-" * 80)
            for err in errs:
                print(f"{err['sample_id']:<26} | {err['field']:<18} | {err['cer'] * 100:>4.1f}% | {err['cause']:<22}")
                print(f"   -> Expected : {err['expected']}")
                print(f"   -> Actual   : {err['actual']} (Status: {err['status']})")
            print("-" * 80)
        else:
            print("\n[SUCCESS] No extraction errors detected across all evaluated fields!")
        print("=" * 80 + "\n")

    def print_preprocessing_comparison(self, res_on: Dict[str, Any], res_off: Dict[str, Any]):
        print("\n" + "=" * 80)
        print("          PREPROCESSING COMPARISON: ENABLED VS DISABLED          ".center(80))
        print("=" * 80)
        print(f"{'Metric':<32} | {'Enabled':<18} | {'Disabled':<18} | {'Delta':<12}")
        print("-" * 80)
        
        em_on = res_on["overall"]["exact_match_rate"] * 100
        em_off = res_off["overall"]["exact_match_rate"] * 100
        delta_em = em_on - em_off
        print(f"{'Overall Exact Match':<32} | {em_on:>6.2f}%            | {em_off:>6.2f}%            | {delta_em:>+6.2f}%")

        cer_on = res_on["overall"]["average_cer"] * 100
        cer_off = res_off["overall"]["average_cer"] * 100
        delta_cer = cer_on - cer_off
        print(f"{'Average CER':<32} | {cer_on:>6.2f}%            | {cer_off:>6.2f}%            | {delta_cer:>+6.2f}%")

        lat_on = res_on["latency_stats"]["total_mean_ms"]
        lat_off = res_off["latency_stats"]["total_mean_ms"]
        delta_lat = lat_on - lat_off
        print(f"{'Mean Total Latency (ms)':<32} | {lat_on:>8.2f} ms        | {lat_off:>8.2f} ms        | {delta_lat:>+8.2f} ms")
        
        inf_on = res_on["latency_stats"]["inference_mean_ms"]
        inf_off = res_off["latency_stats"]["inference_mean_ms"]
        delta_inf = inf_on - inf_off
        print(f"{'Mean Inference Latency (ms)':<32} | {inf_on:>8.2f} ms        | {inf_off:>8.2f} ms        | {delta_inf:>+8.2f} ms")
        print("=" * 80 + "\n")

    @staticmethod
    def _to_camel(s: str) -> str:
        parts = s.split("_")
        return parts[0] + "".join(p.capitalize() for p in parts[1:])

    @staticmethod
    def _to_snake(s: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CampusHub-OCR Benchmark & Evaluation Suite")
    parser.add_argument(
        "--dataset",
        default="tests/fixtures/ground_truth_sample.json",
        help="Path to the JSON ground-truth benchmark dataset",
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "image", "all"],
        default="all",
        help="Evaluation mode: 'synthetic' (fast, heuristic-only), 'image' (end-to-end PaddleOCR), or 'all'",
    )
    parser.add_argument(
        "--compare-preprocessing",
        action="store_true",
        help="Run image samples twice (preprocessing enabled vs disabled) and print comparative analysis",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to output machine-readable JSON evaluation report",
    )
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(
        dataset_path=args.dataset,
        mode=args.mode,
        compare_preprocessing=args.compare_preprocessing
    )

    if args.compare_preprocessing:
        print("[INFO] Running evaluation with Preprocessing ENABLED...")
        res_enabled = evaluator.run_evaluation(preprocess=True)
        evaluator.print_report(res_enabled, "(PREPROCESSING ENABLED)")

        print("[INFO] Running evaluation with Preprocessing DISABLED...")
        res_disabled = evaluator.run_evaluation(preprocess=False)
        evaluator.print_report(res_disabled, "(PREPROCESSING DISABLED)")

        evaluator.print_preprocessing_comparison(res_enabled, res_disabled)

        if args.output_json:
            combined = {
                "preprocessing_enabled": res_enabled,
                "preprocessing_disabled": res_disabled,
            }
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Report written to: {args.output_json}")
    else:
        res = evaluator.run_evaluation(preprocess=True)
        evaluator.print_report(res)

        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Report written to: {args.output_json}")
