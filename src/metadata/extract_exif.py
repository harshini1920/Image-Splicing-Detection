"""
Metadata extraction and inconsistency flagging.
Covers EXIF (via piexif) and basic file-format checks.
"""

import piexif
from PIL import Image
from datetime import datetime


def _decode(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode(errors="ignore").strip("\x00").strip()
        except Exception:
            return str(value)
    return value


def extract_exif(image_path: str) -> dict:
    result = {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime_original": None,
        "datetime_modified": None,
        "gps_present": False,
    }

    try:
        exif_dict = piexif.load(image_path)
    except Exception:
        return result

    zeroth = exif_dict.get("0th", {})
    exif_ifd = exif_dict.get("Exif", {})
    gps_ifd = exif_dict.get("GPS", {})

    if zeroth or exif_ifd:
        result["has_exif"] = True

    result["camera_make"] = _decode(zeroth.get(piexif.ImageIFD.Make))
    result["camera_model"] = _decode(zeroth.get(piexif.ImageIFD.Model))
    result["software"] = _decode(zeroth.get(piexif.ImageIFD.Software))
    result["datetime_modified"] = _decode(zeroth.get(piexif.ImageIFD.DateTime))
    result["datetime_original"] = _decode(exif_ifd.get(piexif.ExifIFD.DateTimeOriginal))
    result["gps_present"] = len(gps_ifd) > 0

    return result


def check_file_format(image_path: str) -> dict:
    info = {"format": None, "mode": None, "size": None}
    try:
        with Image.open(image_path) as img:
            info["format"] = img.format
            info["mode"] = img.mode
            info["size"] = list(img.size)
    except Exception:
        pass
    return info


EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "affinity", "paint.net",
    "pixlr", "canva", "snapseed", "picsart",
]


def flag_inconsistencies(exif_data: dict, file_info: dict) -> list:
    flags = []

    if not exif_data["has_exif"]:
        flags.append("No EXIF metadata present (common in edited, re-saved, or AI images).")

    software = (exif_data.get("software") or "").lower()
    if any(kw in software for kw in EDITING_SOFTWARE_KEYWORDS):
        flags.append(f"Editing software detected: '{exif_data['software']}'.")

    if exif_data["has_exif"] and not exif_data["camera_make"] and not exif_data["camera_model"]:
        flags.append("EXIF present but camera make/model missing.")

    dt_orig = exif_data.get("datetime_original")
    dt_mod = exif_data.get("datetime_modified")
    if dt_orig and dt_mod:
        try:
            fmt = "%Y:%m:%d %H:%M:%S"
            if datetime.strptime(dt_mod, fmt) < datetime.strptime(dt_orig, fmt):
                flags.append("Modified timestamp is earlier than original timestamp.")
        except Exception:
            pass

    if file_info.get("format") == "TIFF":
        flags.append("TIFF file - note the CASIA v2 format-leakage caveat.")

    if not flags:
        flags.append("No obvious metadata inconsistencies detected.")

    return flags


def analyze_metadata(image_path: str) -> dict:
    exif_data = extract_exif(image_path)
    file_info = check_file_format(image_path)
    return {
        "exif": exif_data,
        "file_info": file_info,
        "flags": flag_inconsistencies(exif_data, file_info),
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python src/metadata/extract_exif.py <image_path>")
        sys.exit(1)

    print(json.dumps(analyze_metadata(sys.argv[1]), indent=2, default=str))