"""
Splicing Localization (pipeline step 8)
-------------------------------------------
Produces a per-pixel "suspicion heatmap" showing WHERE in an image tampering
likely occurred, unlike every earlier module (ELA, pixel/RGB, DCT/FFT),
which each collapse their evidence into whole-image summary statistics for
a classifier.

Approach (classical, no deep learning -- consistent with the rest of this
project's CPU-only, interpretable design):

  1. Reuse the ELA residual maps already computed at all 7 quality levels.
     Each map is z-scored independently (so differing scales across quality
     levels don't bias the fusion), then averaged. Pixels that are outliers
     relative to the REST of their own image, at ANY quality level, light up.

  2. Reuse the DCT block-grid strength map from the frequency module.
     A spliced region often breaks the regular 8x8 JPEG grid pattern.

  3. Compute a fine-grained local noise-inconsistency map (finer than the
     4x4 grid used in pixel_analysis.py) and upsample it to full resolution.
     A pasted region usually carries different sensor/compression noise.

  4. Z-score each of the three evidence maps, average them into one
     combined heatmap, smooth it, then threshold + clean with morphology
     to extract candidate tampered regions as bounding boxes.

Usage:
    from splicing_localization import localize_splicing

    result = localize_splicing("image.jpg")
    result["heatmap"]   # (H, W) float array, 0-1, higher = more suspicious
    result["mask"]      # (H, W) binary array, thresholded heatmap
    result["boxes"]     # list of {"bbox": (x0,y0,x1,y1), "score": float}

CLI:
    python splicing_localization.py image.jpg [output_plot.png]
"""

import os
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["ela", "frequency"]:
    _p = os.path.join(_SRC_DIR, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from multi_scale_ela import multi_scale_ela_features, load_image_rgb, QUALITY_LEVELS
from dct_fft_analysis import to_gray, _blockiness_map

MIN_REGION_AREA = 600  # pixels; smaller connected regions are treated as noise
SMOOTH_SIGMA = 4.0
THRESHOLD_PERCENTILE = 98.5  # only the top (100-this)% of combined evidence is flagged
# NOTE: these constants are the module's main tuning knobs. A percentile
# threshold (rather than a fixed mean + k*std) is used because real photo
# texture (foliage, water, skin) has a heavy-tailed evidence distribution
# that breaks a fixed std-based cutoff -- std-based thresholding produced
# many false-positive boxes on real, texture-rich photos during testing,
# even though it looked fine on smooth synthetic test images. Percentile
# thresholding adapts to each image's own evidence distribution instead.
#
# 98.5 was chosen by sweeping 95-99 against clean vs. synthetically spliced
# test images (both smooth and texture-rich): it was the only value with
# zero false-positive regions on both clean tests while still localizing
# the tampered region (sometimes as a smaller box fully inside the true
# region rather than its exact full extent -- a partial-but-correct hit,
# not a miss). This is NOT calibrated against a labeled dataset, and a
# genuinely different trade-off exists depending on how large the tampered
# region is relative to the image (a percentile threshold inherently
# favors small/medium anomalies over ones covering a large image fraction).
# Proper calibration should happen once CASIA/Columbia ground-truth masks
# are available (step 11) by sweeping THRESHOLD_PERCENTILE against known
# tampered regions rather than against synthetic tests. Raise the
# percentile for fewer, more confident boxes; lower it for higher recall.


def _zscore(arr):
    mean, std = arr.mean(), arr.std()
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _ela_evidence_map(image_path, max_dim=1024):
    """Average z-scored ELA residual across all 7 quality levels -> (H, W)."""
    _, ela_maps = multi_scale_ela_features(image_path, max_dim=max_dim)
    z_stack = np.stack([_zscore(ela_maps[q]) for q in QUALITY_LEVELS], axis=0)
    return z_stack.mean(axis=0)


def _dct_evidence_map(gray):
    """Z-scored JPEG block-grid strength map -> (H, W)."""
    diff_h, diff_v = _blockiness_map(gray)
    combined = diff_h + diff_v
    # the raw map is mostly zero off-grid; smooth slightly so grid lines
    # spread into a usable region signal rather than staying needle-thin
    combined = ndimage.gaussian_filter(combined, sigma=2.0)
    return _zscore(combined)


def _noise_evidence_map(gray, cell=24):
    """
    Fine-grained local noise sigma, computed on a cell x cell grid and
    upsampled (nearest) back to full resolution, then z-scored.
    """
    h, w = gray.shape
    n_rows = max(1, h // cell)
    n_cols = max(1, w // cell)

    noise_grid = np.zeros((n_rows, n_cols), dtype=np.float64)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)

    for r in range(n_rows):
        y0, y1 = r * cell, (r + 1) * cell if r < n_rows - 1 else h
        for c in range(n_cols):
            x0, x1 = c * cell, (c + 1) * cell if c < n_cols - 1 else w
            block = gray[y0:y1, x0:x1]
            bh, bw = block.shape
            if bh < 3 or bw < 3:
                noise_grid[r, c] = 0.0
                continue
            conv = ndimage.convolve(block, kernel, mode="reflect")[1:-1, 1:-1]
            noise_grid[r, c] = np.sqrt(np.pi / 2.0) / (6.0 * (bw - 2) * (bh - 2)) * np.sum(np.abs(conv))

    upsampled = np.array(
        Image.fromarray(noise_grid.astype(np.float32)).resize((w, h), Image.BILINEAR)
    )
    return _zscore(upsampled)


def _extract_boxes(mask, heatmap, min_area=MIN_REGION_AREA):
    """Connected-component labeling -> list of bounding boxes with scores."""
    labeled, n = ndimage.label(mask)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_area:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        score = float(heatmap[ys, xs].mean())
        boxes.append({
            "bbox": (int(x0), int(y0), int(x1), int(y1)),
            "area_px": int(len(ys)),
            "score": score,
        })
    boxes.sort(key=lambda b: -b["score"])
    return boxes


def localize_splicing(image_path, max_dim=1024):
    """
    Produce a suspicion heatmap and candidate tampered-region boxes.

    Returns
    -------
    dict with:
        "heatmap"  (H, W) float array, min-max normalized to [0, 1]
        "mask"     (H, W) uint8 binary array (thresholded heatmap)
        "boxes"    list of {"bbox": (x0,y0,x1,y1), "area_px": int, "score": float}
                   sorted by score descending; empty if nothing crosses threshold
    """
    img = load_image_rgb(image_path, max_dim=max_dim) if False else None
    # load via the same resize convention the ELA/frequency modules use
    pil_img = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path.convert("RGB")
    w, h = pil_img.size
    scale = max_dim / max(w, h)
    if scale < 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.asarray(pil_img, dtype=np.float64)
    gray = to_gray(arr)

    ela_z = _ela_evidence_map(image_path, max_dim=max_dim)
    dct_z = _dct_evidence_map(gray)
    noise_z = _noise_evidence_map(gray)

    # all three should already share the same (H, W) since every module
    # resizes with the same max_dim convention; guard anyway
    target_shape = gray.shape
    def _match(a):
        if a.shape == target_shape:
            return a
        return np.array(Image.fromarray(a.astype(np.float32)).resize(
            (target_shape[1], target_shape[0]), Image.BILINEAR))

    combined = (_match(ela_z) + _match(dct_z) + _match(noise_z)) / 3.0
    combined = ndimage.gaussian_filter(combined, sigma=SMOOTH_SIGMA)

    # normalize to [0, 1] for display
    c_min, c_max = combined.min(), combined.max()
    heatmap = (combined - c_min) / (c_max - c_min + 1e-8)

    threshold = np.percentile(combined, THRESHOLD_PERCENTILE)
    mask = (combined > threshold).astype(np.uint8)

    # morphological cleanup: remove speckle, fill small gaps
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3))).astype(np.uint8)
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5))).astype(np.uint8)

    boxes = _extract_boxes(mask, heatmap)

    return {"heatmap": heatmap, "mask": mask, "boxes": boxes}


def plot_localization(image_path, output_path="splicing_localization.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size
    max_dim = 1024
    scale = max_dim / max(w, h)
    if scale < 1.0:
        pil_img_disp = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        pil_img_disp = pil_img

    result = localize_splicing(image_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(pil_img_disp)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(result["heatmap"], cmap="inferno")
    axes[1].set_title("Suspicion heatmap")
    axes[1].axis("off")

    axes[2].imshow(pil_img_disp)
    axes[2].imshow(result["heatmap"], cmap="inferno", alpha=0.45)
    for box in result["boxes"]:
        x0, y0, x1, y1 = box["bbox"]
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  linewidth=2, edgecolor="cyan", facecolor="none")
        axes[2].add_patch(rect)
    axes[2].set_title(f"Overlay + {len(result['boxes'])} candidate region(s)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python splicing_localization.py <image_path> [output_plot.png]")
        sys.exit(1)

    out = sys.argv[2] if len(sys.argv) > 2 else "splicing_localization.png"
    result = plot_localization(sys.argv[1], out)

    print(f"Plot saved to {out}")
    print(f"Found {len(result['boxes'])} candidate tampered region(s):")
    for i, box in enumerate(result["boxes"], 1):
        x0, y0, x1, y1 = box["bbox"]
        print(f"  [{i}] bbox=({x0},{y0})-({x1},{y1})  area={box['area_px']}px  score={box['score']:.3f}")