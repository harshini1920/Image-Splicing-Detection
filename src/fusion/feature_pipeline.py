"""
Feature Fusion Pipeline
---------------------------
Runs every feature-extraction module on a single image and returns one
flat, numeric feature dict, plus a "raw" dict of human-readable details
(for the eventual PDF/JSON report) and an "extras" dict of arrays/maps
(for visualization).

Usage:
    from feature_pipeline import extract_all_features
    result = extract_all_features("image.jpg")
    result["features"]   # flat dict[str, float]
    result["raw"]        # nested dict -- metadata flags, file info, etc.
    result["extras"]     # dict of numpy arrays/maps for plotting

CLI:
    python feature_pipeline.py image.jpg
"""

import os
import sys
import time

# wire up imports from sibling src/ folders
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["metadata", "pixel_rgb", "ela", "frequency"]:
    _p = os.path.join(_SRC_DIR, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from extract_exif import analyze_metadata
from pixel_analysis import pixel_features
from multi_scale_ela import multi_scale_ela_features
from cross_quality_ratios import cross_quality_ratio_features
from dct_fft_analysis import frequency_features


def _metadata_numeric_features(meta_result):
    exif = meta_result["exif"]
    file_info = meta_result["file_info"]
    flags = meta_result["flags"]

    software = (exif.get("software") or "").lower()
    editing_keywords = [
        "photoshop", "gimp", "lightroom", "affinity", "paint.net",
        "pixlr", "canva", "snapseed", "picsart",
    ]
    has_editing_software = any(kw in software for kw in editing_keywords)
    real_flags = [f for f in flags if not f.startswith("No obvious")]

    return {
        "meta_has_exif": float(exif["has_exif"]),
        "meta_gps_present": float(exif["gps_present"]),
        "meta_has_software_tag": 1.0 if exif.get("software") else 0.0,
        "meta_has_editing_software": 1.0 if has_editing_software else 0.0,
        "meta_has_camera_info": 1.0 if (exif.get("camera_make") or exif.get("camera_model")) else 0.0,
        "meta_is_tiff": 1.0 if file_info.get("format") == "TIFF" else 0.0,
        "meta_is_jpeg": 1.0 if file_info.get("format") == "JPEG" else 0.0,
        "meta_flag_count": float(len(real_flags)),
    }


def extract_all_features(image_path, max_dim=1024):
    """
    Run every feature module on one image.

    Returns dict with:
        "features"    flat dict[str, float], ~368 entries
        "raw"         nested dict with metadata flags/details
        "extras"      dict of arrays (ELA maps, noise grid, FFT spectrum)
        "timing_sec"  float, total extraction time
    """
    t0 = time.time()
    features = {}

    meta_result = analyze_metadata(image_path)
    features.update(_metadata_numeric_features(meta_result))

    pixel_feats, pixel_extras = pixel_features(image_path, max_dim=max_dim)
    features.update({f"pix_{k}": v for k, v in pixel_feats.items()})

    ela_feats, ela_maps = multi_scale_ela_features(image_path, max_dim=max_dim)
    features.update(ela_feats)

    ratio_feats = cross_quality_ratio_features(ela_maps)
    features.update(ratio_feats)

    freq_feats, freq_extras = frequency_features(image_path, max_dim=max_dim)
    features.update({f"freq_{k}": v for k, v in freq_feats.items()})

    elapsed = time.time() - t0

    return {
        "features": features,
        "raw": {
            "metadata": meta_result,
            "image_path": image_path,
        },
        "extras": {
            "ela_maps": ela_maps,
            "pixel_noise_grid": pixel_extras["noise_grid"],
            "pixel_histograms": pixel_extras["histograms"],
            "freq_blockiness_h": freq_extras["blockiness_h"],
            "freq_blockiness_v": freq_extras["blockiness_v"],
            "freq_fft_spectrum_log": freq_extras["fft_spectrum_log"],
        },
        "timing_sec": elapsed,
    }


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python feature_pipeline.py <image_path>")
        sys.exit(1)

    result = extract_all_features(sys.argv[1])
    feats = result["features"]

    print(f"Extracted {len(feats)} total features in {result['timing_sec']:.2f}s\n")

    by_module = {}
    for k in feats:
        prefix = k.split("_")[0] if not k.startswith("q") else "ela"
        by_module.setdefault(prefix, 0)
        by_module[prefix] += 1
    print("Feature count by module:")
    for mod, count in sorted(by_module.items(), key=lambda x: -x[1]):
        print(f"  {mod:>8}: {count}")

    print("\nMetadata flags:")
    for flag in result["raw"]["metadata"]["flags"]:
        print(f"  - {flag}")

    out_path = os.path.splitext(sys.argv[1])[0] + "_features.json"
    with open(out_path, "w") as f:
        json.dump(feats, f, indent=2)
    print(f"\nSaved flat feature vector to {out_path}")