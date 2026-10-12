"""
Pixel size read from TIFF metadata — tier 1, the exact one.

None of this needs OCR, which is the point: metadata is what succeeds on most
instrument exports, and it must not be gated on a heavy optional dependency.
"""

import sys

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from PIL import Image

from sem_particle_analysis import ScaleDetector
from sem_particle_analysis import scale_calibration as sc


@pytest.fixture(scope="module")
def detector():
    """A detector with no OCR reader built — metadata needs none."""
    return ScaleDetector(use_gpu=False)


def write_tiff(path, description=None, resolution=None, resolution_unit=None,
               shape=(64, 64)):
    """Write a small TIFF carrying the requested tags."""
    data = np.zeros(shape, dtype=np.uint8)
    kwargs = {}
    if description is not None:
        kwargs["description"] = description
    if resolution is not None:
        kwargs["resolution"] = resolution
    if resolution_unit is not None:
        kwargs["resolutionunit"] = resolution_unit
    tifffile.imwrite(str(path), data, **kwargs)
    return path


class TestNoOCRRequired:
    def test_constructing_a_detector_builds_no_ocr_reader(self, detector):
        # EasyOCR takes seconds to load and pulls a large dependency tree; a
        # metadata-only run must not pay for it.
        assert detector._reader is None

    def test_reading_metadata_never_touches_the_reader(self, detector, tmp_path):
        path = write_tiff(tmp_path / "fei.tif", description="[Scan]\nPixelWidth=2.5e-9\n")
        detector.detect_scale_from_metadata(str(path))
        assert detector._reader is None


class TestFEI:
    def test_reads_pixelwidth_in_metres_and_converts_to_nanometres(self, detector, tmp_path):
        # FEI writes PixelWidth in metres; 2.5e-9 m is 2.5 nm/px.
        path = write_tiff(tmp_path / "fei.tif",
                          description="[User]\nDate=01/01/2026\n[Scan]\nPixelWidth=2.5e-9\nPixelHeight=2.5e-9\n")
        result = detector.detect_scale_from_metadata(str(path))
        assert result["conversion"] == pytest.approx(2.5)
        assert result["manufacturer"] == "fei"
        assert result["pixel_length"] == 1

    def test_falls_back_to_pixelheight(self, detector, tmp_path):
        path = write_tiff(tmp_path / "fei.tif",
                          description="[Scan]\nPixelHeight=1.0e-8\n")
        assert detector.detect_scale_from_metadata(str(path))["conversion"] == pytest.approx(10.0)


class TestGenericPatterns:
    @pytest.mark.parametrize(
        "description,expected",
        [
            ("pixel size = 10 nm", 10.0),
            ("PixelSize: 5.5 um", 5500.0),
            ("2.5 nm/pixel", 2.5),
            ("resolution = 4 nm", 4.0),
        ],
    )
    def test_reads_a_pixel_size_written_in_plain_text(self, detector, tmp_path, description, expected):
        path = write_tiff(tmp_path / "generic.tif", description=description)
        result = detector.detect_scale_from_metadata(str(path))
        assert result["conversion"] == pytest.approx(expected)


class TestResolutionTags:
    def test_reads_a_centimetre_resolution(self, detector, tmp_path):
        # 1e6 px/cm -> 1e-6 cm/px -> 10 nm/px
        path = write_tiff(tmp_path / "res.tif", resolution=(1e6, 1e6), resolution_unit=3)
        result = detector.detect_scale_from_metadata(str(path))
        assert result["conversion"] == pytest.approx(10.0, rel=1e-3)

    @pytest.mark.parametrize("dpi", [72, 96, 300])
    def test_ignores_standard_display_dpi(self, detector, tmp_path, dpi):
        # 96 dpi is the scanner/display default, not an SEM calibration. Trusting
        # it would report ~265,000 nm/px and rescale every measurement.
        path = write_tiff(tmp_path / f"dpi{dpi}.tif", resolution=(dpi, dpi), resolution_unit=2)
        with pytest.raises(ValueError):
            detector.detect_scale_from_metadata(str(path))


class TestRejections:
    def test_rejects_a_non_tiff(self, detector, tmp_path):
        path = tmp_path / "image.png"
        path.write_bytes(b"not a tiff")
        with pytest.raises(ValueError):
            detector.detect_scale_from_metadata(str(path))

    def test_raises_when_the_file_carries_no_pixel_size(self, detector, tmp_path):
        path = write_tiff(tmp_path / "bare.tif", description="just some text")
        with pytest.raises(ValueError):
            detector.detect_scale_from_metadata(str(path))

    def test_rejects_a_sub_atomic_pixel_size(self, detector, tmp_path):
        # A misplaced exponent must fail loudly rather than rescale the dataset.
        path = write_tiff(tmp_path / "tiny.tif", description="[Scan]\nPixelWidth=1e-15\n")
        with pytest.raises(ValueError):
            detector.detect_scale_from_metadata(str(path))

    def test_rejects_an_absurdly_large_pixel_size(self, detector, tmp_path):
        path = write_tiff(tmp_path / "huge.tif", description="[Scan]\nPixelWidth=1.0\n")
        with pytest.raises(ValueError):
            detector.detect_scale_from_metadata(str(path))


class TestValidationBounds:
    @pytest.mark.parametrize("value", [None, 0, -1])
    def test_non_positive_values_are_invalid(self, detector, value):
        _, validity, _ = detector._validate_pixel_size(value)
        assert validity == "invalid"

    def test_boundaries_of_the_plausible_range(self, detector):
        low, high = detector.VALID_PIXEL_SIZE_RANGE
        assert detector._validate_pixel_size(low)[1] == "valid"
        assert detector._validate_pixel_size(high)[1] == "valid"
        assert detector._validate_pixel_size(low / 2)[1] == "invalid"
        assert detector._validate_pixel_size(high * 2)[1] == "invalid"


class TestTierOne:
    """What the Scale tab's first tier gets back from a file."""

    def test_a_recorded_pixel_size_is_the_scale(self, detector, tmp_path):
        path = write_tiff(tmp_path / "fei.tif", description="[Scan]\nPixelWidth=2.5e-9\n")
        calibration = sc.from_metadata(detector, path)
        assert calibration.method == "metadata"
        assert calibration.nm_per_px == pytest.approx(2.5)

    def test_a_tiff_without_one_is_refused(self, detector, tmp_path):
        path = write_tiff(tmp_path / "bare.tif", description="nothing useful")
        with pytest.raises(sc.ScaleError):
            sc.from_metadata(detector, path)

    def test_a_png_is_refused_with_the_reason(self, detector, tmp_path):
        path = tmp_path / "frame.png"
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(path)
        with pytest.raises(sc.ScaleError, match="only TIFF files record a pixel size"):
            sc.from_metadata(detector, path)

    def test_no_ocr_is_needed(self, detector, tmp_path, monkeypatch):
        # Metadata must work where EasyOCR is not installed, and must not
        # download OCR models where it is.
        monkeypatch.setitem(sys.modules, "easyocr", None)
        path = write_tiff(tmp_path / "fei.tif", description="[Scan]\nPixelWidth=2.5e-9\n")
        assert sc.from_metadata(detector, path).nm_per_px == pytest.approx(2.5)
