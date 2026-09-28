"""Pixel & RGB forensic analysis (step 4): 52 features. Usage: python pixel_analysis.py image.jpg [plot.png]"""

import numpy as np
from PIL import Image
from scipy import ndimage, stats

CHANNELS = ["R", "G", "B"]
GRID_SIZE = 4

_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
_LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def load_image_rgb(path_or_image, max_dim=1024):
    if isinstance(path_or_image, Image.Image):
        img = path_or_image
    else:
        img = Image.open(path_or_image)
    img = img.convert("RGB")
    if max_dim is not None:
        w, h = img.size
        scale = max_dim / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def channel_histograms(arr):
    hists = np.zeros((3, 256), dtype=np.float64)
    for c in range(3):
        counts = np.bincount(arr[..., c].ravel(), minlength=256).astype(np.float64)
        hists[c] = counts / counts.sum()
    return hists


def _entropy(hist):
    p = hist[hist > 0]
    return float(-np.sum(p * np.log2(p)))


def _gap_fraction(hist):
    occupied = np.nonzero(hist)[0]
    if len(occupied) < 2:
        return 0.0
    span = occupied[-1] - occupied[0] + 1
    return float((span - len(occupied)) / span)


def estimate_noise_sigma(gray):
    """Immerkaer fast noise standard-deviation estimate."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    conv = ndimage.convolve(gray.astype(np.float64), _NOISE_KERNEL, mode="reflect")
    conv = conv[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * np.sum(np.abs(conv)))


def _noise_grid(gray, grid_size=GRID_SIZE):
    h, w = gray.shape
    out = np.zeros((grid_size, grid_size), dtype=np.float64)
    for r in range(grid_size):
        y0, y1 = r * h // grid_size, (r + 1) * h // grid_size
        for c in range(grid_size):
            x0, x1 = c * w // grid_size, (c + 1) * w // grid_size
            out[r, c] = estimate_noise_sigma(gray[y0:y1, x0:x1])
    return out


def pixel_features(path_or_image, max_dim=1024):
    """Returns (features dict, {"histograms": (3,256), "noise_grid": (4,4)})."""
    arr = load_image_rgb(path_or_image, max_dim=max_dim)
    f = {}
    hists = channel_histograms(arr)

    for c, name in enumerate(CHANNELS):
        ch = arr[..., c].astype(np.float64).ravel()
        f[f"{name}_mean"] = float(ch.mean())
        f[f"{name}_std"] = float(ch.std())
        if ch.std() > 0:
            f[f"{name}_skew"] = float(stats.skew(ch))
            f[f"{name}_kurtosis"] = float(stats.kurtosis(ch))
        else:
            f[f"{name}_skew"] = 0.0
            f[f"{name}_kurtosis"] = 0.0
        f[f"{name}_median"] = float(np.median(ch))
        f[f"{name}_p1"] = float(np.percentile(ch, 1))
        f[f"{name}_p99"] = float(np.percentile(ch, 99))
        f[f"{name}_clip_low"] = float(np.mean(ch == 0))
        f[f"{name}_clip_high"] = float(np.mean(ch == 255))
        f[f"{name}_hist_entropy"] = _entropy(hists[c])
        f[f"{name}_hist_gap_fraction"] = _gap_fraction(hists[c])

    r = arr[..., 0].astype(np.float64)
    g = arr[..., 1].astype(np.float64)
    b = arr[..., 2].astype(np.float64)

    def _corr(a, bb):
        if a.std() == 0 or bb.std() == 0:
            return 0.0
        return float(np.corrcoef(a.ravel(), bb.ravel())[0, 1])

    f["corr_RG"] = _corr(r, g)
    f["corr_RB"] = _corr(r, b)
    f["corr_GB"] = _corr(g, b)
    for n1, c1, n2, c2 in [("R", r, "G", g), ("R", r, "B", b), ("G", g, "B", b)]:
        d = np.abs(c1 - c2)
        f[f"absdiff_{n1}{n2}_mean"] = float(d.mean())
        f[f"absdiff_{n1}{n2}_std"] = float(d.std())

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    f["noise_sigma"] = estimate_noise_sigma(gray)
    lap = ndimage.convolve(gray, _LAPLACIAN_KERNEL, mode="reflect")
    f["laplacian_variance"] = float(lap.var())
    for c, name in enumerate(CHANNELS):
        f[f"{name}_noise_sigma"] = estimate_noise_sigma(arr[..., c].astype(np.float64))

    ngrid = _noise_grid(gray)
    f["noise_grid_mean"] = float(ngrid.mean())
    f["noise_grid_std"] = float(ngrid.std())
    f["noise_grid_range"] = float(ngrid.max() - ngrid.min())
    f["noise_grid_cv"] = float(ngrid.std() / ngrid.mean()) if ngrid.mean() > 0 else 0.0

    flat = arr.reshape(-1, 3)
    packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    f["unique_color_ratio"] = float(len(np.unique(packed)) / len(packed))

    return f, {"histograms": hists, "noise_grid": ngrid}


def plot_analysis(path, output_path="pixel_analysis.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = load_image_rgb(path)
    features, extras = pixel_features(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(arr)
    axes[0].set_title("Image")
    axes[0].axis("off")

    for c, (name, color) in enumerate(zip(CHANNELS, ["r", "g", "b"])):
        axes[1].plot(extras["histograms"][c], color=color, label=name, linewidth=0.8)
    axes[1].set_title("RGB histograms")
    axes[1].set_xlabel("Pixel value")
    axes[1].legend()

    im = axes[2].imshow(extras["noise_grid"], cmap="viridis")
    axes[2].set_title("Local noise sigma (4x4 grid)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    return features


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pixel_analysis.py <image_path> [output_plot.png]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "pixel_analysis.png"
    feats = plot_analysis(sys.argv[1], out)
    print(f"Extracted {len(feats)} features. Plot saved to {out}\n")
    for k in ["noise_sigma", "laplacian_variance", "noise_grid_std", "noise_grid_cv",
              "corr_RG", "corr_RB", "corr_GB", "unique_color_ratio"]:
        print(f"{k:>22}: {feats[k]:.4f}")