"""
Standalone script for tuning quality-check thresholds.

Run this against a folder of test images (mix of good and deliberately
bad ones) to see the ACTUAL metrics your images produce, so you can pick
sensible threshold values in app/quality.py instead of guessing.

Usage:
    python tune_quality.py path/to/test_images_folder/
"""

import sys
import os
from app.quality import check_image_quality

def main():
    if len(sys.argv) != 2:
        print("Usage: python tune_quality.py <folder_of_test_images>")
        return

    folder = sys.argv[1]
    print(f"{'FILENAME':<30} {'PASS?':<8} {'BLUR_VAR':<12} {'BRIGHTNESS':<12} {'REASON'}")
    print("-" * 80)

    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, filename)
        with open(path, "rb") as f:
            image_bytes = f.read()

        result = check_image_quality(image_bytes)
        metrics = result.get("metrics", {})
        blur = metrics.get("laplacian_variance", "N/A")
        brightness = metrics.get("mean_brightness", "N/A")
        reason = result.get("reason") or "—"

        print(f"{filename:<30} {str(result['quality_pass']):<8} {str(blur):<12} {str(brightness):<12} {reason}")

if __name__ == "__main__":
    main()