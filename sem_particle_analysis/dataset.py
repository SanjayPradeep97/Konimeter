"""
Training data from analysed frames.

Every frame an analyst saves has been segmented and then corrected by eye, so
its mask is a label worth keeping. ``TrainingSet`` writes each one as an image
and mask pair, in the mask format the classification pipeline already reads,
with a row in ``index.csv`` saying where it came from. A folder analysed for
one study is then also a labelled dataset, for training a classifier or
fine-tuning the segmentation model.

Layout, in the folder that holds the results file::

    konimeter_dataset/
        README.md                 what is here, written once
        index.csv                 one row per frame: source, scale, model, ...
        images/<stem>.png         the frame exactly as segmented (databar trimmed)
        masks/<stem>_mask.png     8-bit; 255 = particle, 0 = not
        labels/<stem>_labels.png  16-bit; 0 = background, i = the i-th particle
                                  of the frame's row in the results file
        valid/<stem>_valid.png    8-bit; 255 = specimen, 0 = left out of analysis
                                  (aperture vignette, grid bar, burned-in scale
                                  bar), so those pixels are not labels either way

Frames are keyed by file name without extension, as the results file matches
them. Saving a frame again replaces its files and its row; ``remove`` deletes
both.
"""

import csv
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

DIRNAME = "konimeter_dataset"

INDEX_COLUMNS = [
    "stem", "image", "mask", "labels", "valid",
    "source_file", "source_sha256", "width", "height", "rows_trimmed",
    "num_particles", "nm_per_px", "scale_method", "modality", "particles",
    "model", "checkpoint", "saved_utc", "konimeter_version",
]

README = """\
# Konimeter training data

Written by Konimeter each time an analysed frame was saved. Each frame's mask
was segmented with SAM and then checked, and corrected where needed, by the
analyst.

| Path | Contents |
| --- | --- |
| `images/<stem>.png` | the frame exactly as segmented: the databar, if any, trimmed off the bottom (`rows_trimmed` in the index) |
| `masks/<stem>_mask.png` | 8-bit; 255 = particle, 0 = not |
| `labels/<stem>_labels.png` | 16-bit instance labels; 0 = background, *i* = the *i*-th particle of the frame's row in the results file |
| `valid/<stem>_valid.png` | 8-bit; 255 = specimen, 0 = left out of analysis (aperture vignette, grid bar, burned-in scale bar). Pixels at 0 are not labels either way |
| `index.csv` | one row per frame: source file and its SHA-256, size, scale and how it was set, instrument kind, particle polarity, SAM model and checkpoint, when it was saved |

A frame saved again replaced its files and its row.
"""


def sha256_file(path, chunk_size=1 << 20):
    """SHA-256 of a file, read in chunks so large TIFFs don't fill memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instance_labels(regions, shape):
    """
    Label image of the measured particles, numbered in measurement order.

    Built from the regions rather than from a relabelling of the mask, so label
    *i* is exactly the *i*-th entry of the results row and nothing smaller than
    the minimum particle size, which was not measured, is labelled.

    Args:
        regions: skimage RegionProps of the measured particles, in order.
        shape: (height, width) of the frame.

    Returns:
        np.ndarray: uint16 labels, 0 for background.
    """
    if len(regions) > np.iinfo(np.uint16).max:
        raise ValueError(f"{len(regions)} particles is more than a 16-bit label image holds")
    labels = np.zeros(shape[:2], dtype=np.uint16)
    for number, region in enumerate(regions, start=1):
        labels[region.coords[:, 0], region.coords[:, 1]] = number
    return labels


def _as_image(array):
    """A frame as a PIL image, single-channel when its channels are identical."""
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[2] == 3 and (
            np.array_equal(array[..., 0], array[..., 1])
            and np.array_equal(array[..., 0], array[..., 2])):
        array = array[..., 0]
    if array.dtype != np.uint8:
        raise ValueError(f"frames are saved as 8-bit; got {array.dtype}")
    return Image.fromarray(array)


def _write_png(image, path):
    """Write through a temporary file, so an interrupted save leaves no torn PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


class TrainingSet:
    """The image and mask pairs, with their index, in one dataset folder."""

    def __init__(self, root):
        self.root = Path(root)

    @classmethod
    def beside(cls, results_file):
        """The dataset that belongs with a results file: in the same folder."""
        return cls(Path(results_file).parent / DIRNAME)

    @property
    def index_path(self):
        return self.root / "index.csv"

    def paths(self, stem):
        return {
            "image": self.root / "images" / f"{stem}.png",
            "mask": self.root / "masks" / f"{stem}_mask.png",
            "labels": self.root / "labels" / f"{stem}_labels.png",
            "valid": self.root / "valid" / f"{stem}_valid.png",
        }

    def rows(self):
        """The index, one dict per frame."""
        if not self.index_path.exists():
            return []
        with open(self.index_path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def save(self, stem, image, labels, valid=None, **record):
        """
        Write one frame's image, masks and index row, replacing any earlier ones.

        Args:
            stem: The frame's file name without extension.
            image: The frame as segmented, uint8, (H, W) or (H, W, 3).
            labels: uint16 instance labels of the same height and width.
            valid: Boolean mask of the specimen, or None when all of it is.
            **record: Further index columns (see INDEX_COLUMNS).

        Returns:
            Path: the dataset folder.
        """
        image = np.asarray(image)
        labels = np.asarray(labels)
        height, width = image.shape[:2]
        if labels.shape != (height, width):
            raise ValueError(f"labels are {labels.shape}, the frame is {(height, width)}")
        if valid is None:
            valid = np.ones((height, width), dtype=bool)
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (height, width):
            raise ValueError(f"valid mask is {valid.shape}, the frame is {(height, width)}")

        paths = self.paths(stem)
        _write_png(_as_image(image), paths["image"])
        _write_png(Image.fromarray(np.where(labels > 0, 255, 0).astype(np.uint8)),
                   paths["mask"])
        _write_png(Image.fromarray(labels.astype(np.uint16)), paths["labels"])
        _write_png(Image.fromarray(np.where(valid, 255, 0).astype(np.uint8)),
                   paths["valid"])

        row = {column: "" for column in INDEX_COLUMNS}
        row.update({key: ("" if value is None else value) for key, value in record.items()
                    if key in row})
        row.update(stem=stem, width=width, height=height,
                   num_particles=int(labels.max()),
                   **{key: path.relative_to(self.root).as_posix()
                      for key, path in paths.items()})
        self._write_index([r for r in self.rows() if r.get("stem") != stem] + [row])

        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text(README, encoding="utf-8")
        return self.root

    def remove(self, stem):
        """Delete a frame's files and its index row. Returns True if it was there."""
        found = False
        for path in self.paths(stem).values():
            if path.exists():
                path.unlink()
                found = True
        rows = self.rows()
        kept = [r for r in rows if r.get("stem") != stem]
        if len(kept) != len(rows):
            self._write_index(kept)
            found = True
        return found

    def _write_index(self, rows):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_name("index.csv.part")
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, self.index_path)
