"""
Batch Feature Extraction over CASIA v2.0 (pipeline step 10 prep)
---------------------------------------------------------------------
Runs the fusion pipeline over every image in data/external/casia/{authentic,tampered}
and writes one JSON line per image to an output file.

FORMAT-CONTROLLED BY DEFAULT: defaults to the JPEG-only subset (authentic
.jpg/.jpeg, tampered .jpg/.jpeg) to avoid the TIFF/JPEG format-confound
shortcut described in the paper. Use --include-non-jpeg to process the
full mixed-format dataset instead.

Resumable: if the output file already exists, already-processed filenames
are skipped, so an interrupted run can be restarted with the same command.

Usage:
    python extract_dataset_features.py --limit 20 --log-every 5
    python extract_dataset_features.py
    python extract_dataset_features.py --include-non-jpeg --output ../../outputs/evaluation/casia_features_mixed.jsonl
"""

import argparse
import json
import os
import sys
import time
import traceback

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FUSION_DIR = os.path.join(_SRC_DIR, "fusion")
if _FUSION_DIR not in sys.path:
    sys.path.insert(0, _FUSION_DIR)

from feature_pipeline import extract_all_features

JPEG_EXTS = {".jpg", ".jpeg"}
ALL_IMAGE_EXTS = {".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".png"}

DEFAULT_CASIA_DIR = os.path.join(_SRC_DIR, "..", "data", "external", "casia")
DEFAULT_OUTPUT = os.path.join(_SRC_DIR, "..", "outputs", "evaluation", "casia_features.jsonl")


def _list_images(folder, jpeg_only):
    exts = JPEG_EXTS if jpeg_only else ALL_IMAGE_EXTS
    out = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in exts:
            out.append(os.path.join(folder, fname))
    return out


def _load_already_done(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["filename"])
                except Exception:
                    continue
    return done


def extract_dataset(
    casia_dir=DEFAULT_CASIA_DIR,
    output_path=DEFAULT_OUTPUT,
    jpeg_only=True,
    limit=None,
    log_every=25,
):
    authentic_dir = os.path.join(casia_dir, "authentic")
    tampered_dir = os.path.join(casia_dir, "tampered")

    authentic_files = _list_images(authentic_dir, jpeg_only)
    tampered_files = _list_images(tampered_dir, jpeg_only)

    if limit is not None:
        authentic_files = authentic_files[:limit]
        tampered_files = tampered_files[:limit]

    tasks = [(p, 0) for p in authentic_files] + [(p, 1) for p in tampered_files]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    already_done = _load_already_done(output_path)

    print(f"Authentic images queued: {len(authentic_files)}")
    print(f"Tampered images queued:  {len(tampered_files)}")
    print(f"Total: {len(tasks)}  (already done, will skip: {len(already_done)})")
    print(f"Output: {output_path}\n")

    n_done = 0
    n_skipped = 0
    n_errors = 0
    errors = []
    t_start = time.time()

    with open(output_path, "a") as out_f:
        for i, (path, label) in enumerate(tasks, 1):
            filename = os.path.basename(path)
            if filename in already_done:
                n_skipped += 1
                continue

            try:
                result = extract_all_features(path)
                record = {
                    "filename": filename,
                    "path": path,
                    "label": label,
                    "features": result["features"],
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                n_done += 1
            except Exception as e:
                n_errors += 1
                errors.append({"filename": filename, "error": str(e)})
                traceback.print_exc(limit=1)

            if i % log_every == 0 or i == len(tasks):
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = len(tasks) - i
                eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
                print(
                    f"[{i}/{len(tasks)}] done={n_done} skipped={n_skipped} "
                    f"errors={n_errors} | {rate:.2f} img/s | ETA {eta_min:.1f} min"
                )

    print(f"\nFinished. Processed {n_done} new images, skipped {n_skipped}, {n_errors} errors.")
    if errors:
        err_path = output_path.replace(".jsonl", "_errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Error details saved to {err_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-extract fusion features over CASIA v2.0")
    parser.add_argument("--casia-dir", default=DEFAULT_CASIA_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--include-non-jpeg", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    extract_dataset(
        casia_dir=args.casia_dir,
        output_path=args.output,
        jpeg_only=not args.include_non_jpeg,
        limit=args.limit,
        log_every=args.log_every,
    )