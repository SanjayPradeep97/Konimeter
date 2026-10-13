"""
Text files some instruments write beside each image.

A JEOL SEM saves ``S1_0001.tif`` together with ``S1_0001.txt``, and the text
file is the only place the image's scale is recorded: the TIFF carries no pixel
size at all. Lines look like::

    $CM_INSTRUMENT JSM 6500F
    $CM_IMAGE_RES 1280x1024
    $$SM_COLUMN_MODE SEM
    $$SM_MICRON_BAR 160
    $$SM_MICRON_MARKER 1µm

``SM_MICRON_BAR`` is the length of the printed scale bar in pixels and
``SM_MICRON_MARKER`` its label, so their ratio is the pixel size.
"""

import os
import re
import unicodedata

# Lengths as JEOL writes them; µ arrives as Latin-1 0xB5 and is decoded as such.
_UNIT_NM = {"nm": 1.0, "um": 1e3, "µm": 1e3, "μm": 1e3, "mm": 1e6}


def find_sidecar(image_path):
    """The ``.txt`` file with the same name as the image, or None."""
    stem, _ = os.path.splitext(str(image_path))
    folder = os.path.dirname(stem) or "."
    base = os.path.basename(stem).lower()
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    for name in names:
        root, ext = os.path.splitext(name)
        if root.lower() == base and ext.lower() == ".txt":
            return os.path.join(folder, name)
    return None


def read_sidecar(image_path, limit=64_000):
    """The sidecar's text, or None when there is none or it cannot be read."""
    path = find_sidecar(image_path)
    if path is None:
        return None
    try:
        with open(path, "rb") as handle:
            raw = handle.read(limit)
    except OSError:
        return None
    # JEOL writes Latin-1, so µ is byte 0xB5; UTF-8 sidecars decode the same way
    # apart from non-ASCII, which only ever appears in units here.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _field(text, name):
    match = re.search(rf"^\s*\$+{name}\s+(.*?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_jeol(text):
    """
    What a JEOL sidecar says, or None if the text is not one.

    Returns:
        dict with ``instrument``, ``column_mode`` ('SEM', 'TEM' or None),
        ``width`` (image width the bar was drawn for, or None), and
        ``nm_per_px`` (None when the bar fields are missing or unreadable).
    """
    if not text or "$CM_" not in text:
        return None
    instrument = _field(text, "CM_INSTRUMENT")
    mode = (_field(text, "SM_COLUMN_MODE") or "").upper() or None
    result = {"instrument": instrument, "column_mode": mode,
              "width": None, "nm_per_px": None}

    resolution = _field(text, "CM_IMAGE_RES") or ""
    match = re.match(r"(\d+)\s*[x×]\s*(\d+)", resolution)
    if match:
        result["width"] = int(match.group(1))

    bar_px = _field(text, "SM_MICRON_BAR")
    marker = _field(text, "SM_MICRON_MARKER")
    if bar_px and marker:
        marker = unicodedata.normalize("NFKC", marker).replace(" ", "")
        found = re.match(r"([\d.]+)\s*(nm|um|µm|μm|mm)$", marker, re.IGNORECASE)
        try:
            pixels = float(bar_px)
            if found and pixels > 0:
                result["nm_per_px"] = (float(found.group(1))
                                       * _UNIT_NM[found.group(2).lower()] / pixels)
        except ValueError:
            pass
    return result
