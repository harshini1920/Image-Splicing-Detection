"""
DCT / JPEG Blockiness / FFT Frequency Analysis (pipeline step 7)
-------------------------------------------------------------------
Three complementary feature families that look for frequency-domain and
JPEG block-grid evidence of tampering or recompression:

  1. DCT blockiness (8 features)
     JPEG compresses in independent 8x8 pixel blocks. Each recompression
     leaves a faint grid of intensity jumps at block boundaries. We compare
     pixel differences AT the 8x8 grid lines against differences just off
     the grid; a spliced/edited region often has a different (or missing)
     blocking pattern than the rest of the image.

  2. FFT radial energy bands (6 features)
     The 2-D frequency spectrum of a natural image falls off smoothly from
     low to high frequency. Resampling, resizing, or splicing a region from
     a different source disturbs this falloff. We bin the spectrum into
     concentric radial bands and report each band's share of total energy,
     plus a high/low frequency energy ratio.

  3. Color statistics (12 features)
     Skewness and kurtosis per channel, inter-channel correlation already
     covered in pixel_analysis.py is intentionally NOT duplicated here;
     this module focuses on color-channel shape (skew/kurtosis) since that
     complements the simpler mean/std already extracted elsewhere.

Usage:
    from dct_fft_analysis import frequency_features

    features, extras = frequency_features("image.jpg")

CLI:
    python dct_fft_analysis.py image.jpg [output_plot.png]
"""

import numpy as np
from PIL import Image
from scipy import stats
from scipy.fft import fft2, fftshift

BLOCK_SIZE = 8
FFT_RESIZE = 256  # grayscale image is resized to this before FFT
RADIAL_BANDS = [(0, 16), (16, 32), (32, 64), (64, 96), (96, 128)]


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
    return img


def to_gray(arr_rgb):
    return 0.299 * arr_rgb[..., 0] + 0.587 * arr_rgb[..., 1] + 0.114 * arr_rgb[..., 2]


# ---------------------------------------------------------------------------
# 1. DCT blockiness
# ---------------------------------------------------------------------------

def _blockiness_map(gray, block_size=BLOCK_SIZE):
    """
    Returns a (H, W) map where each pixel's value is the absolute intensity
    jump across the nearest 8x8 block boundary, measured along rows and
    columns and averaged. High values cluster where JPEG blocking is strong.
    """
    h, w = gray.shape
    diff_h = np.zeros_like(gray)
    diff_v = np.zeros_like(gray)

    # horizontal differences across vertical block-boundary columns
    for x in range(block_size, w, block_size):
        diff_h[:, x - 1:x + 1] = np.abs(gray[:, x:x + 1] - gray[:, x - 1:x]) if x < w else 0

    # vertical differences across horizontal block-boundary rows
    for y in range(block_size, h, block_size):
        diff_v[y - 1:y + 1, :] = np.abs(gray[y:y + 1, :] - gray[y - 1:y, :]) if y < h else 0

    return diff_h, diff_v


def dct_blockiness_features(gray, block_size=BLOCK_SIZE):
    """8 features summarizing JPEG block-grid strength and its spatial spread."""
    h, w = gray.shape
    diff_h, diff_v = _blockiness_map(gray, block_size)

    on_grid = float(np.mean(diff_h) + np.mean(diff_v))

    # off-grid baseline: same measurement but offset by half a block,
    # i.e. differences NOT aligned with the JPEG grid
    offset = block_size // 2
    diff_h_off = np.abs(np.diff(gray, axis=1))
    diff_v_off = np.abs(np.diff(gray, axis=0))
    off_grid = float(np.mean(diff_h_off) + np.mean(diff_v_off))

    blockiness_ratio = on_grid / (off_grid + 1e-6)

    # spatial variance of blockiness: split image into a 4x4 grid, compute
    # on-grid blockiness strength per cell, then look at how much it varies
    cell_scores = []
    gh, gw = h // 4, w // 4
    for r in range(4):
        for c in range(4):
            y0, y1 = r * gh, (r + 1) * gh if r < 3 else h
            x0, x1 = c * gw, (c + 1) * gw if c < 3 else w
            cell_on = float(np.mean(diff_h[y0:y1, x0:x1]) + np.mean(diff_v[y0:y1, x0:x1]))
            cell_scores.append(cell_on)
    cell_scores = np.array(cell_scores)

    return {
        "dct_on_grid_strength": on_grid,
        "dct_off_grid_strength": off_grid,
        "dct_blockiness_ratio": float(blockiness_ratio),
        "dct_cellwise_mean": float(cell_scores.mean()),
        "dct_cellwise_std": float(cell_scores.std()),
        "dct_cellwise_range": float(cell_scores.max() - cell_scores.min()),
        "dct_cellwise_cv": float(cell_scores.std() / cell_scores.mean()) if cell_scores.mean() > 0 else 0.0,
        "dct_block_size_used": float(block_size),
    }


# ---------------------------------------------------------------------------
# 2. FFT radial energy bands
# ---------------------------------------------------------------------------

def fft_radial_energy_features(gray, size=FFT_RESIZE, bands=RADIAL_BANDS):
    """6 features: fractional energy in each radial band + high/low freq ratio."""
    img = Image.fromarray(gray.astype(np.uint8)).resize((size, size), Image.LANCZOS)
    g = np.asarray(img, dtype=np.float64)

    spectrum = np.abs(fftshift(fft2(g)))
    power = spectrum ** 2

    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    total_energy = power.sum()
    feats = {}
    band_energies = []
    for i, (r0, r1) in enumerate(bands):
        mask = (radius >= r0) & (radius < r1)
        e = power[mask].sum()
        frac = float(e / total_energy) if total_energy > 0 else 0.0
        feats[f"fft_band{i}_frac"] = frac
        band_energies.append(e)

    low_energy = band_energies[0] + band_energies[1]  # bands 0-1: radius < 32
    high_energy = sum(band_energies[3:])  # bands 3-4: radius >= 64
    feats["fft_high_low_ratio"] = float(high_energy / (low_energy + 1e-6))

    return feats


# ---------------------------------------------------------------------------
# 3. Color shape statistics (skew/kurtosis, complementing pixel_analysis.py)
# ---------------------------------------------------------------------------

def color_shape_features(arr_rgb):
    """12 features: skew, kurtosis, IQR, MAD per channel."""
    feats = {}
    for c, name in enumerate(["R", "G", "B"]):
        ch = arr_rgb[..., c].astype(np.float64).ravel()
        if ch.std() > 0:
            skew = float(stats.skew(ch))
            kurt = float(stats.kurtosis(ch))
        else:
            skew, kurt = 0.0, 0.0
        q75, q25 = np.percentile(ch, [75, 25])
        iqr = float(q75 - q25)
        mad = float(np.median(np.abs(ch - np.median(ch))))
        feats[f"{name}_skew2"] = skew
        feats[f"{name}_kurtosis2"] = kurt
        feats[f"{name}_iqr"] = iqr
        feats[f"{name}_mad"] = mad
    return feats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def frequency_features(path_or_image, max_dim=1024):
    """
    Compute all step-7 features for one image.

    Returns
    -------
    features : dict[str, float]   (26 features total)
    extras   : dict with "blockiness_h", "blockiness_v", "fft_spectrum_log"
               for visualization
    """
    img = load_image_rgb(path_or_image, max_dim=max_dim)
    arr = np.asarray(img, dtype=np.float64)
    gray = to_gray(arr)

    features = {}
    features.update(dct_blockiness_features(gray))
    features.update(fft_radial_energy_features(gray))
    features.update(color_shape_features(arr))

    diff_h, diff_v = _blockiness_map(gray)
    gray_small = np.asarray(
        Image.fromarray(gray.astype(np.uint8)).resize((FFT_RESIZE, FFT_RESIZE), Image.LANCZOS),
        dtype=np.float64,
    )
    spectrum_log = np.log1p(np.abs(fftshift(fft2(gray_small))))

    extras = {
        "blockiness_h": diff_h,
        "blockiness_v": diff_v,
        "fft_spectrum_log": spectrum_log,
    }
    return features, extras


def plot_analysis(path, output_path="dct_fft_analysis.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = load_image_rgb(path)
    arr = np.asarray(img)
    features, extras = frequency_features(path)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    axes[0].imshow(arr)
    axes[0].set_title("Image")
    axes[0].axis("off")

    blockiness_total = extras["blockiness_h"] + extras["blockiness_v"]
    im1 = axes[1].imshow(blockiness_total, cmap="inferno")
    axes[1].set_title("JPEG block-grid strength")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(extras["fft_spectrum_log"], cmap="magma")
    axes[2].set_title("FFT magnitude spectrum (log)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    band_fracs = [features[f"fft_band{i}_frac"] for i in range(len(RADIAL_BANDS))]
    axes[3].bar(range(len(band_fracs)), band_fracs, color="steelblue")
    axes[3].set_title("FFT radial band energy")
    axes[3].set_xlabel("Band (low -> high freq)")
    axes[3].set_ylabel("Fraction of total energy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    return features


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dct_fft_analysis.py <image_path> [output_plot.png]")
        sys.exit(1)

    out = sys.argv[2] if len(sys.argv) > 2 else "dct_fft_analysis.png"
    feats = plot_analysis(sys.argv[1], out)
    print(f"Extracted {len(feats)} features. Plot saved to {out}\n")
    for k in ["dct_blockiness_ratio", "dct_cellwise_std", "dct_cellwise_cv",
              "fft_high_low_ratio", "fft_band0_frac", "fft_band4_frac",
              "R_skew2", "R_kurtosis2"]:
        print(f"{k:>24}: {feats[k]:.4f}")