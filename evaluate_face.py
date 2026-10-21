#!/usr/bin/env python3
"""
CampusHub-OCR — 1:1 Face Verification Benchmark & Evaluation Tool.
Evaluates:
- False Acceptance Rate (FAR / FMR)
- False Rejection Rate (FRR / FNMR)
- Accuracy, Precision, Recall, and F1-score across various similarity thresholds
- Face detection failure rates on CCCD cards vs selfies
- Latency benchmarks (mean, median, p95)
- Image quality warning distributions
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import numpy as np

# Ensure UTF-8 printing on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import settings
from app.services.face_verifier import (
    FaceVerifierService,
    YuNetSFaceEngine,
    MockFaceEngine,
)
from app.services.image_processor import ImageProcessor


def run_benchmark(
    pairs: List[Dict[str, Any]],
    verifier: FaceVerifierService,
    thresholds: List[float] = [0.25, 0.30, 0.363, 0.40, 0.45, 0.50],
) -> Dict[str, Any]:
    """Runs face verification evaluation over paired samples."""
    print("=" * 70)
    print("CAMPUSHUB-OCR: 1:1 FACE VERIFICATION BENCHMARK SUITE")
    print("=" * 70)
    print(f"Total test pairs: {len(pairs)}")
    print(f"Thresholds under evaluation: {thresholds}")
    print("-" * 70)

    results = []
    latencies = []
    detection_failures = {"card": 0, "selfie": 0}
    quality_warnings_count = 0

    for idx, pair in enumerate(pairs, start=1):
        pair_id = pair.get("id", f"pair_{idx}")
        is_same = bool(pair.get("is_same_person", True))
        card_img = pair.get("card_image")
        selfie_img = pair.get("selfie_image")

        t0 = time.perf_counter()
        resp = verifier.verify_faces(card_img, selfie_img, threshold=settings.FACE_MATCH_THRESHOLD)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        if not resp.card_face.detected:
            detection_failures["card"] += 1
        if not resp.selfie_face.detected:
            detection_failures["selfie"] += 1
        if resp.warnings:
            quality_warnings_count += len(resp.warnings)

        results.append({
            "pair_id": pair_id,
            "ground_truth_same": is_same,
            "similarity_score": resp.similarity_score,
            "card_detected": resp.card_face.detected,
            "selfie_detected": resp.selfie_face.detected,
            "warnings": resp.warnings,
        })

    # Evaluate metrics across threshold grid
    threshold_metrics = {}
    total_positives = sum(1 for r in results if r["ground_truth_same"])
    total_negatives = len(results) - total_positives

    for th in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        for r in results:
            predicted_same = (r["card_detected"] and r["selfie_detected"] and r["similarity_score"] >= th)
            actual_same = r["ground_truth_same"]

            if predicted_same and actual_same:
                tp += 1
            elif predicted_same and not actual_same:
                fp += 1
            elif not predicted_same and not actual_same:
                tn += 1
            else:
                fn += 1

        far = round(fp / max(1, total_negatives) * 100, 2)  # False Accept Rate
        frr = round(fn / max(1, total_positives) * 100, 2)  # False Reject Rate
        acc = round((tp + tn) / max(1, len(results)) * 100, 2)
        prec = round(tp / max(1, tp + fp) * 100, 2)
        rec = round(tp / max(1, tp + fn) * 100, 2)
        f1 = round(2 * prec * rec / max(1, prec + rec), 2)

        threshold_metrics[str(th)] = {
            "threshold": th,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "FAR_percent": far,
            "FRR_percent": frr,
            "Accuracy_percent": acc,
            "Precision_percent": prec,
            "Recall_percent": rec,
            "F1_percent": f1,
        }

    summary = {
        "total_pairs": len(pairs),
        "total_positives": total_positives,
        "total_negatives": total_negatives,
        "detection_failures": detection_failures,
        "quality_warnings_total": quality_warnings_count,
        "latencies_ms": {
            "mean": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "median": round(float(np.median(latencies)), 2) if latencies else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        },
        "threshold_metrics": threshold_metrics,
    }

    print("\nBENCHMARK RESULTS SUMMARY:")
    print(f"- Mean Latency: {summary['latencies_ms']['mean']} ms (p95: {summary['latencies_ms']['p95']} ms)")
    print(f"- Card Face Detection Failures: {detection_failures['card']}/{len(pairs)}")
    print(f"- Selfie Face Detection Failures: {detection_failures['selfie']}/{len(pairs)}")
    print(f"- Quality Warnings Logged: {quality_warnings_count}")
    print("\nMETRICS BY THRESHOLD:")
    print(f"{'Threshold':<10} | {'FAR (%)':<10} | {'FRR (%)':<10} | {'Acc (%)':<10} | {'F1 (%)':<10}")
    print("-" * 56)
    for th, m in threshold_metrics.items():
        print(f"{th:<10} | {m['FAR_percent']:<10} | {m['FRR_percent']:<10} | {m['Accuracy_percent']:<10} | {m['F1_percent']:<10}")
    print("=" * 70)

    return summary


def create_synthetic_benchmark_pairs(count: int = 10) -> List[Dict[str, Any]]:
    """Generates synthetic test pairs for offline CI pipeline testing."""
    pairs = []
    for i in range(count):
        is_same = (i % 2 == 0)
        card_img = np.full((400, 600, 3), 180, dtype=np.uint8)
        selfie_img = np.full((500, 500, 3), 180 if is_same else 80, dtype=np.uint8)
        pairs.append({
            "id": f"synthetic_{i+1}",
            "is_same_person": is_same,
            "card_image": card_img,
            "selfie_image": selfie_img,
        })
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Evaluate 1:1 Face Verification performance.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to evaluation dataset manifest JSON.")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic test pairs.")
    parser.add_argument("--out", type=str, default=None, help="Output JSON results path.")
    args = parser.parse_args()

    # Initialize Face Verifier
    try:
        verifier = FaceVerifierService(engine=YuNetSFaceEngine())
        print("[INFO] Initialized FaceVerifierService with YuNet + SFace models.")
    except Exception as e:
        print(f"[WARN] Could not load ONNX models ({e}). Falling back to MockFaceEngine.")
        verifier = FaceVerifierService(engine=MockFaceEngine())

    if args.synthetic or not args.manifest:
        print("[INFO] Running in synthetic benchmark mode (10 pairs).")
        pairs = create_synthetic_benchmark_pairs(10)
    else:
        # Load manifest
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        # Load images from manifest
        pairs = []
        base_dir = os.path.dirname(args.manifest)
        for item in manifest_data:
            c_path = os.path.join(base_dir, item["card_image"])
            s_path = os.path.join(base_dir, item["selfie_image"])
            card_img = cv2.imread(c_path)
            selfie_img = cv2.imread(s_path)
            if card_img is not None and selfie_img is not None:
                pairs.append({
                    "id": item.get("id"),
                    "is_same_person": item.get("is_same_person", True),
                    "card_image": card_img,
                    "selfie_image": selfie_img,
                })

    summary = run_benchmark(pairs, verifier)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Benchmark results saved to: {args.out}")


if __name__ == "__main__":
    main()
