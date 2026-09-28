"""Multi-scale ELA: 7 JPEG quality levels x (6 global + 32 grid) = 266 features."""

import io
import numpy as np
from PIL import Image

QUALITY_LEVELS = [30, 50, 60, 75, 80, 90, 95]
GRID_SIZE = 4


def load_image_rgb(path_or_image):
    if isinstance(path_or_image, Image.Image):
        img = path_or_image
    else:
        img = Image.open(path_or_image)
    return img.convert("RGB")


def compute_ela(image, quality):
    """|original - recompressed| per pixel per channel, shape (H, W, 3)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    original_arr = np.asarray(image, dtype=np.float32)
    recompressed_arr = np.asarray(recompressed, dtype=np.float32)
    return np.abs(original_arr - recompressed_arr)


def _global_stats(res):
    mean = float(np.mean(res))
    std = float(np.std(res))
    return {
        "mean": mean,
        "std": std,
        "p75": float(np.percentile(res, 75)),
        "p95": float(np.percentile(res, 95)),
        "p99": float(np.percentile(res, 99)),
        "high_residual_fraction": float(np.mean(res > mean + 2 * std)),
    }


def _grid_stats(res, grid_size=GRID_SIZE):
    h, w = res.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    features = {}
    idx = 0
    for row in range(grid_size):
        y0 = row * cell_h
        y1 = (row + 1) * cell_h if row < grid_size - 1 else h
        for col in range(grid_size):
            x0 = col * cell_w
            x1 = (col + 1) * cell_w if col < grid_size - 1 else w
            cell = res[y0:y1, x0:x1]
            features[f"grid{idx}_mean"] = float(np.mean(cell))
            features[f"grid{idx}_std"] = float(np.std(cell))
            idx += 1
    return features


def multi_scale_ela_features(path_or_image, max_dim=1024):
    """Returns (features dict with 266 entries, {quality: grayscale ELA map})."""
    img = load_image_rgb(path_or_image)

    if max_dim is not None:
        w, h = img.size
        scale = max_dim / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    all_features = {}
    ela_maps = {}

    for q in QUALITY_LEVELS:
        residual_gray = compute_ela(img, q).mean(axis=2)
        ela_maps[q] = residual_gray
        for key, val in _global_stats(residual_gray).items():
            all_features[f"q{q}_{key}"] = val
        for key, val in _grid_stats(residual_gray).items():
            all_features[f"q{q}_{key}"] = val

    return all_features, ela_maps


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python multi_scale_ela.py <image_path>")
        sys.exit(1)
    feats, _ = multi_scale_ela_features(sys.argv[1])
    print(f"Extracted {len(feats)} features")