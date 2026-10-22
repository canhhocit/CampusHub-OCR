"""
evaluate_liveness_pad.py
========================
Evaluation and benchmarking tool for Passive Face Anti-Spoofing (PAD) & Active Liveness.
Standardized according to ISO/IEC 30107-3 Presentation Attack Detection metrics:

- APCER (Attack Presentation Classification Error Rate):
    Proportion of spoof presentations (screen replay, printed photos) incorrectly classified as LIVE.
- BPCER (Bona Fide Presentation Classification Error Rate):
    Proportion of genuine bona fide live presentations incorrectly rejected as SPOOF or UNCERTAIN.
- Latency (ms):
    Mean and percentile inference execution time per frame on CPU.

Usage:
    python evaluate_liveness_pad.py --help
    python evaluate_liveness_pad.py --benchmark
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

from app.config import settings
from app.schemas.liveness import AntiSpoofingOutcome
from app.services.anti_spoofing_service import (
    MiniFASNetAntiSpoofingDetector,
    AntiSpoofingResult,
)


def evaluate_sample_dataset(
    detector: MiniFASNetAntiSpoofingDetector,
    samples: List[Tuple[str, str, Tuple[int, int, int, int]]],
) -> Dict[str, Any]:
    """
    Evaluates a dataset of test presentations.

    Args:
        detector: Initialized MiniFASNet detector.
        samples: List of (image_path, ground_truth, face_bbox)
                 ground_truth: 'LIVE' or 'SPOOF'

    Returns:
        Dictionary containing APCER, BPCER, accuracy, latency stats.
    """
    total_live = 0
    total_spoof = 0
    false_accept_spoof = 0  # Spoof classified as LIVE (Security failure)
    false_reject_live = 0   # Live classified as SPOOF or UNCERTAIN (Usability failure)
    latencies: List[float] = []

    results_log = []

    for img_path, truth, bbox in samples:
        if not os.path.exists(img_path):
            print(f"[WARN] Sample file not found, skipping: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        t0 = time.time()
        res: AntiSpoofingResult = detector.predict(img, bbox)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)

        is_pred_live = (res.outcome == AntiSpoofingOutcome.LIVE)

        if truth == "LIVE":
            total_live += 1
            if not is_pred_live:
                false_reject_live += 1
        elif truth == "SPOOF":
            total_spoof += 1
            if is_pred_live:
                false_accept_spoof += 1

        results_log.append({
            "path": os.path.basename(img_path),
            "ground_truth": truth,
            "predicted_outcome": res.outcome.value,
            "score_real": res.score_real,
            "score_paper": res.score_paper,
            "score_screen": res.score_screen,
            "latency_ms": round((t1 - t0) * 1000.0, 2),
        })

    apcer = (false_accept_spoof / total_spoof) * 100.0 if total_spoof > 0 else 0.0
    bpcer = (false_reject_live / total_live) * 100.0 if total_live > 0 else 0.0

    return {
        "total_samples": len(results_log),
        "total_live": total_live,
        "total_spoof": total_spoof,
        "false_accept_spoof (APCER count)": false_accept_spoof,
        "false_reject_live (BPCER count)": false_reject_live,
        "APCER (%)": round(apcer, 2),
        "BPCER (%)": round(bpcer, 2),
        "latency_mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        "samples": results_log,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Passive Face Anti-Spoofing (MiniFASNet)")
    parser.add_argument("--benchmark", action="store_true", help="Run standard synthetic and available sample benchmark")
    args = parser.parse_args()

    print("=" * 65)
    print("CampusHub-OCR: Passive Face Anti-Spoofing (PAD) Benchmark")
    print("=" * 65)

    detector = MiniFASNetAntiSpoofingDetector()
    if not detector.is_available:
        print("[ERROR] MiniFASNet detector models are not available. Check models/face/.")
        sys.exit(1)

    print("[INFO] Model V2:   ", detector.model_path_v2)
    print("[INFO] Model V1SE: ", detector.model_path_v1se)
    print(f"[INFO] Threshold LIVE:  >= {settings.ANTI_SPOOFING_THRESHOLD_LIVE}")
    print(f"[INFO] Threshold SPOOF: <= {settings.ANTI_SPOOFING_THRESHOLD_SPOOF}")
    print("-" * 65)

    # Built-in verification samples if present in scratch
    scratch_dir = r"C:\Users\Canh\.gemini\antigravity-ide\brain\d20dffd6-e21e-424c-aaf4-f22b6fb0cadb\scratch"
    samples = []

    img_t1 = os.path.join(scratch_dir, "image_T1.jpg")
    img_f1 = os.path.join(scratch_dir, "image_F1.jpg")

    if os.path.exists(img_t1):
        # image_T1 is a genuine bona fide real face
        samples.append((img_t1, "LIVE", (130, 100, 160, 200)))
    if os.path.exists(img_f1):
        # image_F1 is a screen replay presentation attack
        samples.append((img_f1, "SPOOF", (130, 100, 160, 200)))

    # Synthetic baseline checks
    synth_live = np.full((480, 640, 3), 140, dtype=np.uint8)
    cv2.circle(synth_live, (320, 240), 90, (200, 180, 160), -1)
    synth_path = os.path.join("scratch_synth_test.jpg")
    cv2.imwrite(synth_path, synth_live)
    samples.append((synth_path, "SPOOF", (220, 140, 200, 200)))

    report = evaluate_sample_dataset(detector, samples)

    print("\nBenchmark Results Summary:")
    print(f"- Total Presentations:      {report['total_samples']}")
    print(f"- Genuine Live Samples:     {report['total_live']}")
    print(f"- Attack Spoof Samples:     {report['total_spoof']}")
    print(f"- Attack Error Rate (APCER): {report['APCER (%)']}% (Lower is better)")
    print(f"- False Rejection (BPCER):   {report['BPCER (%)']}% (Lower is better)")
    print(f"- Average Latency (CPU):    {report['latency_mean_ms']} ms/frame")
    print(f"- 95th Percentile Latency:  {report['latency_p95_ms']} ms/frame")
    print("\nSample Log Details:")
    for s in report["samples"]:
        print(f"  * {s['path']:<22} | Truth: {s['ground_truth']:<5} -> Pred: {s['predicted_outcome']:<9} "
              f"| Real: {s['score_real']:.3f}, Screen: {s['score_screen']:.3f}, Paper: {s['score_paper']:.3f} "
              f"({s['latency_ms']}ms)")

    if os.path.exists(synth_path):
        os.remove(synth_path)

    print("=" * 65)


if __name__ == "__main__":
    main()
