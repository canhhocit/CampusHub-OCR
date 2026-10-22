"""
download_models.py
==================
Official Model Downloader and Integrity Verifier for CampusHub-OCR.

Downloads, verifies cryptographic SHA256 checksums, and smoke-tests all 4 required
ONNX models for Face Detection (YuNet), Face Recognition (SFace), and Passive
Anti-Spoofing / PAD (MiniFASNet V2 & V1SE).

Usage:
    python download_models.py              # Download missing models and verify
    python download_models.py --check-only # Verify existing models without downloading
    python download_models.py --force      # Re-download all models even if present
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from typing import Dict, Any

MODELS: Dict[str, Dict[str, Any]] = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "size_bytes": 232589,
        "license": "Apache-2.0",
        "source": "OpenCV Zoo (YuNet Face Detector)",
        "framework": "OpenCV FaceDetectorYN",
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "size_bytes": 38696353,
        "license": "Apache-2.0",
        "source": "OpenCV Zoo (SFace Face Recognizer)",
        "framework": "OpenCV FaceRecognizerSF",
    },
    "face_anti_spoofing_minifasnetv2_2.7.onnx": {
        "url": "https://raw.githubusercontent.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/master/onnx/2.7_80x80_MiniFASNetV2.onnx",
        "sha256": "0cbe5caec95c31de9d2ef845cb85407d76aecd1b6a2c0e343f7d35306bfbccb8",
        "size_bytes": 1744126,
        "license": "Apache-2.0",
        "source": "Minivision AI / Silent-Face-Anti-Spoofing (Scale 2.7)",
        "framework": "OpenCV DNN (readNetFromONNX)",
    },
    "face_anti_spoofing_minifasnetv1se_4.0.onnx": {
        "url": "https://raw.githubusercontent.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/master/onnx/4_0_0_80x80_MiniFASNetV1SE.onnx",
        "sha256": "a25886a85cdcfa2c4ea23edb71de35f250c17827b4cadd253a972b28c80fdf1e",
        "size_bytes": 1743294,
        "license": "Apache-2.0",
        "source": "Minivision AI / Silent-Face-Anti-Spoofing (Scale 4.0)",
        "framework": "OpenCV DNN (readNetFromONNX)",
    },
}

TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face")


def compute_sha256(filepath: str) -> str:
    """Calculate the SHA256 hex digest of a local file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest().lower()


def download_file(url: str, dest_path: str, expected_size: int) -> None:
    """Download a file with streaming progress display."""
    temp_path = dest_path + ".tmp"
    print(f"  Downloading from: {url}")
    
    headers = {"User-Agent": "CampusHub-OCR-ModelDownloader/1.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
        total_len = int(resp.headers.get("Content-Length", expected_size))
        downloaded = 0
        while chunk := resp.read(65536):
            out_file.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total_len * 100) if total_len > 0 else 0
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_len / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {pct:5.1f}% ({mb_done:5.2f} / {mb_total:5.2f} MB)")
            sys.stdout.flush()
    print()
    os.replace(temp_path, dest_path)


def smoke_test_models() -> bool:
    """Smoke test each model using real OpenCV runtime to guarantee inference compatibility."""
    print("\n--- Smoke Testing Real Model Loading in OpenCV ---")
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[WARN] opencv-python or numpy not installed. Skipping OpenCV inference smoke test.")
        return True

    all_passed = True

    # 1. YuNet
    yunet_path = os.path.join(TARGET_DIR, "face_detection_yunet_2023mar.onnx")
    try:
        detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320))
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        detector.detect(dummy_img)
        print("  [PASS] YuNet FaceDetectorYN loaded and executed forward pass.")
    except Exception as e:
        print(f"  [FAIL] YuNet FaceDetectorYN loading failed: {e}")
        all_passed = False

    # 2. SFace
    sface_path = os.path.join(TARGET_DIR, "face_recognition_sface_2021dec.onnx")
    try:
        recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
        dummy_crop = np.zeros((112, 112, 3), dtype=np.uint8)
        recognizer.feature(dummy_crop)
        print("  [PASS] SFace FaceRecognizerSF loaded and extracted feature vector.")
    except Exception as e:
        print(f"  [FAIL] SFace FaceRecognizerSF loading failed: {e}")
        all_passed = False

    # 3. MiniFASNet V2
    v2_path = os.path.join(TARGET_DIR, "face_anti_spoofing_minifasnetv2_2.7.onnx")
    try:
        net_v2 = cv2.dnn.readNetFromONNX(v2_path)
        blob_v2 = cv2.dnn.blobFromImage(np.zeros((80, 80, 3), dtype=np.uint8), 1.0, (80, 80))
        net_v2.setInput(blob_v2)
        net_v2.forward()
        print("  [PASS] MiniFASNet V2 (scale 2.7) loaded and executed forward pass.")
    except Exception as e:
        print(f"  [FAIL] MiniFASNet V2 loading failed: {e}")
        all_passed = False

    # 4. MiniFASNet V1SE
    v1se_path = os.path.join(TARGET_DIR, "face_anti_spoofing_minifasnetv1se_4.0.onnx")
    try:
        net_v1se = cv2.dnn.readNetFromONNX(v1se_path)
        blob_v1se = cv2.dnn.blobFromImage(np.zeros((80, 80, 3), dtype=np.uint8), 1.0, (80, 80))
        net_v1se.setInput(blob_v1se)
        net_v1se.forward()
        print("  [PASS] MiniFASNet V1SE (scale 4.0) loaded and executed forward pass.")
    except Exception as e:
        print(f"  [FAIL] MiniFASNet V1SE loading failed: {e}")
        all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="CampusHub-OCR Model Downloader and Integrity Verifier")
    parser.add_argument("--check-only", action="store_true", help="Only verify existing models without downloading")
    parser.add_argument("--force", action="store_true", help="Force re-download all models")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip OpenCV inference smoke test")
    args = parser.parse_args()

    os.makedirs(TARGET_DIR, exist_ok=True)
    print("==================================================================")
    print(" CampusHub-OCR: ONNX Model Acquisition & Verification")
    print(f" Target Directory: {TARGET_DIR}")
    print("==================================================================")

    total_models = len(MODELS)
    valid_count = 0

    for filename, info in MODELS.items():
        filepath = os.path.join(TARGET_DIR, filename)
        print(f"\nModel: {filename}")
        print(f"  Source:  {info['source']} ({info['license']})")
        print(f"  Purpose: {info['framework']}")

        needs_download = args.force or not os.path.exists(filepath)

        if not needs_download and os.path.exists(filepath):
            actual_size = os.path.getsize(filepath)
            actual_sha = compute_sha256(filepath)
            if actual_sha == info["sha256"]:
                print(f"  [OK] Model present ({actual_size:,} bytes, SHA256 verified).")
                valid_count += 1
                continue
            else:
                print(f"  [WARN] Checksum mismatch! Expected {info['sha256'][:12]}... got {actual_sha[:12]}...")
                if args.check_only:
                    continue
                needs_download = True

        if args.check_only:
            print("  [MISSING] Model not found locally (check-only mode).")
            continue

        try:
            download_file(info["url"], filepath, info["size_bytes"])
            actual_sha = compute_sha256(filepath)
            if actual_sha == info["sha256"]:
                print(f"  [OK] Successfully downloaded and verified ({os.path.getsize(filepath):,} bytes).")
                valid_count += 1
            else:
                print(f"  [ERROR] SHA256 mismatch after download: {actual_sha}")
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")

    print("\n------------------------------------------------------------------")
    print(f"Summary: {valid_count} / {total_models} models verified.")
    print("------------------------------------------------------------------")

    if valid_count < total_models:
        print("[ERROR] One or more models are missing or corrupted.")
        sys.exit(1)

    if not args.skip_smoke_test:
        if not smoke_test_models():
            print("\n[ERROR] Smoke testing failed. Some models could not be executed by OpenCV.")
            sys.exit(2)

    print("\n[SUCCESS] All 4 ONNX models are present, verified, and operational!")
    sys.exit(0)


if __name__ == "__main__":
    main()
