"""Save a side-by-side ELA heatmap figure. Usage: python visualize_ela.py image.jpg [out.png]"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_scale_ela import multi_scale_ela_features, load_image_rgb, QUALITY_LEVELS

DISPLAY_QUALITIES = [30, 50, 75, 90, 95]


def visualize(image_path, output_path="ela_visualization.png"):
    img = load_image_rgb(image_path)
    features, ela_maps = multi_scale_ela_features(image_path)

    n = len(DISPLAY_QUALITIES) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, q in enumerate(DISPLAY_QUALITIES, start=1):
        amplified = np.clip(ela_maps[q] * 10, 0, 255).astype(np.uint8)
        axes[i].imshow(amplified, cmap="hot")
        axes[i].set_title(f"ELA q={q}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    print(f"Saved visualization to {output_path}")
    print(f"Extracted {len(features)} features\n")
    print(f"{'q':>4} {'mean':>8} {'std':>8} {'p95':>8} {'high_res_frac':>15}")
    for q in QUALITY_LEVELS:
        print(
            f"{q:>4} {features[f'q{q}_mean']:>8.3f} {features[f'q{q}_std']:>8.3f} "
            f"{features[f'q{q}_p95']:>8.3f} {features[f'q{q}_high_residual_fraction']:>15.4f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_ela.py <image_path> [output_path.png]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "ela_visualization.png"
    visualize(sys.argv[1], out)