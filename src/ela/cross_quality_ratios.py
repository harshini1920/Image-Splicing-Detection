"""Cross-quality ELA ratios: 4 quality pairs x (mean, std, p5, p95) = 16 features."""

import numpy as np
from multi_scale_ela import multi_scale_ela_features

QUALITY_PAIRS = [(30, 90), (50, 90), (60, 80), (30, 60)]
EPS = 1e-6


def cross_quality_ratio_features(ela_maps):
    """ela_maps: {quality: 2-D residual map}, as returned by multi_scale_ela_features."""
    feats = {}
    for qa, qb in QUALITY_PAIRS:
        ratio = ela_maps[qa] / (ela_maps[qb] + EPS)
        prefix = f"ratio_{qa}_{qb}"
        feats[f"{prefix}_mean"] = float(np.mean(ratio))
        feats[f"{prefix}_std"] = float(np.std(ratio))
        feats[f"{prefix}_p5"] = float(np.percentile(ratio, 5))
        feats[f"{prefix}_p95"] = float(np.percentile(ratio, 95))
    return feats


def cross_quality_ratio_features_from_image(path_or_image, max_dim=1024):
    _, ela_maps = multi_scale_ela_features(path_or_image, max_dim=max_dim)
    return cross_quality_ratio_features(ela_maps)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cross_quality_ratios.py <image_path>")
        sys.exit(1)
    f = cross_quality_ratio_features_from_image(sys.argv[1])
    print(f"Extracted {len(f)} features\n")
    for k, v in f.items():
        print(f"{k:>22}: {v:,.3f}")