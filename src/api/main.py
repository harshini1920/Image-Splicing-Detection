"""
FastAPI backend.

Endpoints:
    GET  /health   -- liveness check
    POST /analyze  -- upload an image, get back the full feature analysis

Run with:
    uvicorn src.api.main:app --reload --app-dir .
"""

import os
import sys
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FUSION_DIR = os.path.join(_SRC_DIR, "fusion")
if _FUSION_DIR not in sys.path:
    sys.path.insert(0, _FUSION_DIR)

from feature_pipeline import extract_all_features

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MAX_FILE_SIZE_MB = 20

app = FastAPI(
    title="Image Splicing Detection API",
    description="Upload an image to extract forensic features (metadata, pixel/RGB, multi-scale ELA, cross-quality ratios, DCT/FFT).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _summarize_raw(raw):
    meta = raw["metadata"]
    return {
        "exif": meta["exif"],
        "file_info": meta["file_info"],
        "flags": meta["flags"],
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accepts an image upload, runs the full feature-extraction pipeline, and
    returns the flat feature vector plus human-readable metadata flags.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max is {MAX_FILE_SIZE_MB} MB.",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = extract_all_features(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")
    finally:
        os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "num_features": len(result["features"]),
        "processing_time_sec": round(result["timing_sec"], 3),
        "metadata_summary": _summarize_raw(result["raw"]),
        "features": result["features"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)