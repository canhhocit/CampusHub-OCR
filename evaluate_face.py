#!/usr/bin/env python3
"""
CampusHub-OCR — 1:1 Face Verification Benchmark & Evaluation Tool.

MODES:
1. SYNTHETIC SMOKE TEST (--synthetic):
   Validates pipeline flow, request handling, and threshold grid calculation.
   WARNING: DOES NOT measure real biometric accuracy, FAR, or FRR.

2. REAL-IMAGE BENCHMARK (--manifest <path>):
   Evaluates genuine (same-person) and impostor (different-person) pairs
   using real consented test images and real YuNet + SFace ONNX models.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

# Ensure UTF-8 console output on Windows
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


def run_benchmark(
    pairs: List[Dict[str, Any]],
    verifier: FaceVerifierService,
    is_synthetic: bool = False,
    thresholds: List[float] = [0.25, 0.30, 0.363, 0.40, 0.45, 0.50],
) -> Dict[str, Any]:
    """Runs evaluation over paired samples and computes biometric metrics across thresholds."""
    mode_label = "SYNTHETIC PIPELINE SMOKE TEST" if is_synthetic else "REAL-IMAGE BIOMETRIC BENCHMARK"

    print("=" * 72)
    print(f"CAMPUSHUB-OCR: 1:1 FACE VERIFICATION [{mode_label}]")
    print("=" * 72)

    if is_synthetic:
        print("************************************************************************")
        print("[NOTICE: SYNTHETIC SMOKE TEST MODE]")
        print(" - This run tests pipeline mechanics, error handling, and threshold loops.")
        print(" - Synthetic uniform-color images DO NOT represent real human faces.")
        print(" - NEVER present these results as biometric accuracy or model performance.")
        print("************************************************************************\n")

    total_pairs = len(pairs)
    genuine_pairs_count = sum(1 for p in pairs if p.get("is_same_person", True))
    impostor_pairs_count = total_pairs - genuine_pairs_count

    print(f"Total valid test pairs: {total_pairs}")
    print(f" - Genuine pairs (same-person):    {genuine_pairs_count}")
    print(f" - Impostor pairs (different-person): {impostor_pairs_count}")
    print(f"Thresholds under evaluation: {thresholds}")
    print("-" * 72)

    results = []
    latencies = []
    detection_failures = {"card": 0, "selfie": 0}
    quality_rejections = {"card": 0, "selfie": 0, "total_pairs_blocked": 0}
    quality_warnings_count = 0
    matching_executed_count = 0
    genuine_scores = []
    impostor_scores = []

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
        elif resp.card_face.quality and not resp.card_face.quality.is_valid:
            quality_rejections["card"] += 1

        if not resp.selfie_face.detected:
            detection_failures["selfie"] += 1
        elif resp.selfie_face.quality and not resp.selfie_face.quality.is_valid:
            quality_rejections["selfie"] += 1

        if resp.pipeline_state == "BLOCKED_QUALITY":
            quality_rejections["total_pairs_blocked"] += 1
        elif resp.pipeline_state == "MATCHED":
            matching_executed_count += 1
            if is_same:
                genuine_scores.append(resp.similarity_score)
            else:
                impostor_scores.append(resp.similarity_score)

        if resp.warnings:
            quality_warnings_count += len(resp.warnings)

        results.append({
            "pair_id": pair_id,
            "ground_truth_same": is_same,
            "pipeline_state": resp.pipeline_state,
            "similarity_score": resp.similarity_score,
            "success": resp.success,
            "match": resp.match,
            "card_valid": resp.card_face.detected and (resp.card_face.quality.is_valid if resp.card_face.quality else False),
            "selfie_valid": resp.selfie_face.detected and (resp.selfie_face.quality.is_valid if resp.selfie_face.quality else False),
            "warnings": resp.warnings,
        })

    def calc_dist(scores: List[float]) -> Dict[str, Any]:
        if not scores:
            return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        arr = np.array(scores, dtype=np.float32)
        return {
            "count": len(scores),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "std": round(float(np.std(arr)), 4),
        }

    similarity_distribution = {
        "genuine": calc_dist(genuine_scores),
        "impostor": calc_dist(impostor_scores),
    }

    # Evaluate metrics across threshold grid
    threshold_metrics = {}

    for th in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        for r in results:
            predicted_same = (r["success"] and r["similarity_score"] >= th)
            actual_same = r["ground_truth_same"]

            if predicted_same and actual_same:
                tp += 1
            elif predicted_same and not actual_same:
                fp += 1
            elif not predicted_same and not actual_same:
                tn += 1
            else:
                fn += 1

        far = round(fp / max(1, impostor_pairs_count) * 100, 2) if impostor_pairs_count > 0 else 0.0
        frr = round(fn / max(1, genuine_pairs_count) * 100, 2) if genuine_pairs_count > 0 else 0.0
        acc = round((tp + tn) / max(1, total_pairs) * 100, 2)
        prec = round(tp / max(1, tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
        rec = round(tp / max(1, tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / max(0.001, prec + rec), 2) if (prec + rec) > 0 else 0.0

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
        "evaluation_mode": "SYNTHETIC_SMOKE_TEST" if is_synthetic else "REAL_IMAGE_BENCHMARK",
        "biometric_validity": "SMOKE_TEST_ONLY_DO_NOT_CITE" if is_synthetic else "VALID_EMPIRICAL_BENCHMARK",
        "engine_type": "MockFaceEngine" if verifier.is_mock() else "YuNetSFaceEngine (Real ONNX)",
        "total_pairs": total_pairs,
        "genuine_pairs": genuine_pairs_count,
        "impostor_pairs": impostor_pairs_count,
        "detection_failures": detection_failures,
        "quality_rejections": quality_rejections,
        "quality_warnings_total": quality_warnings_count,
        "matching_executed": matching_executed_count,
        "similarity_distribution": similarity_distribution,
        "latencies_ms": {
            "mean": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "median": round(float(np.median(latencies)), 2) if latencies else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        },
        "threshold_metrics": threshold_metrics,
        "disclaimer": (
            "Kết quả thử nghiệm giả lập (smoke test) chỉ kiểm tra chu trình kỹ thuật, "
            "không phản ánh độ chính xác sinh trắc học thực tế." if is_synthetic else
            "Kết quả đánh giá trên tập dữ liệu mẫu nội bộ. Cần tiếp tục mở rộng tập dữ liệu đa dạng."
        ),
    }

    print("\nBENCHMARK RESULTS SUMMARY:")
    print(f"- Mean Latency: {summary['latencies_ms']['mean']} ms (p95: {summary['latencies_ms']['p95']} ms)")
    print(f"- Pairs Matching Executed: {matching_executed_count}/{total_pairs}")
    print(f"- Pairs Blocked by Quality Gate: {quality_rejections['total_pairs_blocked']}/{total_pairs}")
    print(f"- Card Face Detection Failures: {detection_failures['card']}/{total_pairs} (Quality Rejections: {quality_rejections['card']})")
    print(f"- Selfie Face Detection Failures: {detection_failures['selfie']}/{total_pairs} (Quality Rejections: {quality_rejections['selfie']})")
    print(f"- Quality Warnings Logged: {quality_warnings_count}")
    print(f"- Genuine Similarity: Mean={similarity_distribution['genuine']['mean']}, Range=[{similarity_distribution['genuine']['min']}, {similarity_distribution['genuine']['max']}]")
    print(f"- Impostor Similarity: Mean={similarity_distribution['impostor']['mean']}, Range=[{similarity_distribution['impostor']['min']}, {similarity_distribution['impostor']['max']}]")
    print("\nMETRICS BY THRESHOLD:")
    print(f"{'Threshold':<10} | {'FAR (%)':<10} | {'FRR (%)':<10} | {'Acc (%)':<10} | {'F1 (%)':<10}")
    print("-" * 56)
    for th, m in threshold_metrics.items():
        print(f"{th:<10} | {m['FAR_percent']:<10} | {m['FRR_percent']:<10} | {m['Accuracy_percent']:<10} | {m['F1_percent']:<10}")
    print("=" * 72)

    return summary


def create_synthetic_smoke_test_pairs(count: int = 10) -> List[Dict[str, Any]]:
    """Generates synthetic textured images purely for pipeline verification and smoke testing."""
    pairs = []
    for i in range(count):
        is_same = (i % 2 == 0)
        np.random.seed(i)
        card_img = np.random.randint(60, 200, (400, 600, 3), dtype=np.uint8)
        selfie_img = np.random.randint(60, 200, (500, 500, 3), dtype=np.uint8)
        pairs.append({
            "id": f"synthetic_smoke_{i+1}",
            "is_same_person": is_same,
            "card_image": card_img,
            "selfie_image": selfie_img,
        })
    return pairs


def load_manifest_dataset(manifest_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Loads and validates a real dataset from a manifest JSON file."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Tệp manifest không tồn tại: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Tệp manifest không phải định dạng JSON hợp lệ: {e}")

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Tệp manifest rỗng hoặc không phải danh sách các cặp ảnh.")

    base_dir = os.path.dirname(manifest_path)
    valid_pairs = []
    stats = {"total_in_manifest": len(data), "missing_files": 0, "missing_labels": 0}

    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue

        c_rel = item.get("card_image")
        s_rel = item.get("selfie_image")
        if not c_rel or not s_rel:
            stats["missing_files"] += 1
            continue

        if "is_same_person" not in item:
            stats["missing_labels"] += 1
            continue

        c_path = os.path.join(base_dir, c_rel)
        s_path = os.path.join(base_dir, s_rel)

        if not os.path.isfile(c_path) or not os.path.isfile(s_path):
            stats["missing_files"] += 1
            continue

        card_img = cv2.imread(c_path)
        selfie_img = cv2.imread(s_path)
        if card_img is None or selfie_img is None:
            stats["missing_files"] += 1
            continue

        valid_pairs.append({
            "id": item.get("id", f"pair_{idx}"),
            "is_same_person": bool(item["is_same_person"]),
            "card_image": card_img,
            "selfie_image": selfie_img,
        })

    if len(valid_pairs) == 0:
        raise ValueError(
            f"Không có cặp ảnh hợp lệ nào được tải từ {manifest_path}. "
            f"(Tổng mục: {stats['total_in_manifest']}, Tệp thiếu/lỗi: {stats['missing_files']}, Nhãn thiếu: {stats['missing_labels']})"
        )

    return valid_pairs, stats


def main():
    parser = argparse.ArgumentParser(
        description="CampusHub-OCR 1:1 Face Verification Benchmark Suite"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Đường dẫn tệp manifest JSON chứa danh sách các cặp ảnh thực tế."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Chạy ở chế độ Smoke Test giả lập kiểm tra luồng pipeline (không đo độ chính xác thật)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Đường dẫn lưu kết quả báo cáo định dạng JSON."
    )
    args = parser.parse_args()

    # Determine execution mode
    is_synthetic = bool(args.synthetic or not args.manifest)

    if is_synthetic:
        # In synthetic mode, try real engine, or use MockFaceEngine explicitly labeled as mock
        try:
            verifier = FaceVerifierService(engine=YuNetSFaceEngine())
            print("[INFO] Khởi tạo FaceVerifierService với YuNet + SFace thực tế.")
        except Exception:
            print("[INFO] Không tìm thấy model ONNX thực tế. Khởi tạo MockFaceEngine cho Smoke Test.")
            verifier = FaceVerifierService(engine=MockFaceEngine())

        pairs = create_synthetic_smoke_test_pairs(10)
    else:
        # Real-image evaluation: STRICT REQUIREMENT for real ONNX models
        try:
            verifier = FaceVerifierService(engine=YuNetSFaceEngine())
            print("[INFO] Khởi tạo FaceVerifierService với YuNet + SFace thực tế.")
        except Exception as e:
            print(f"\n[FATAL ERROR] Đánh giá ảnh thực tế bắt buộc phải có mô hình YuNet và SFace.")
            print(f"Chi tiết lỗi: {e}")
            print(f"Vui lòng kiểm tra thư mục '{settings.FACE_MODEL_DIR}' trước khi chạy benchmark.")
            sys.exit(1)

        print(f"[INFO] Đang tải và xác thực tệp manifest: {args.manifest}")
        try:
            pairs, stats = load_manifest_dataset(args.manifest)
            print(f"[INFO] Tải thành công {len(pairs)}/{stats['total_in_manifest']} cặp ảnh hợp lệ.")
            if stats["missing_files"] > 0:
                print(f"[WARN] Bỏ qua {stats['missing_files']} cặp ảnh do không tìm thấy tệp hoặc ảnh hỏng.")
            if stats["missing_labels"] > 0:
                print(f"[WARN] Bỏ qua {stats['missing_labels']} mục do thiếu trường 'is_same_person'.")
        except Exception as e:
            print(f"[FATAL ERROR] Lỗi khi nạp dữ liệu manifest: {e}")
            sys.exit(1)

    summary = run_benchmark(pairs, verifier, is_synthetic=is_synthetic)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Kết quả báo cáo đã được lưu tại: {args.out}")


if __name__ == "__main__":
    main()
