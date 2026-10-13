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
    @pytest.mark.parametrize("resolution, unit", [((72, 72), 2), ((96, 96), 2),
                                                  ((300, 300), 2), ((1e6, 1e6), 3)])
    def test_the_resolution_tags_are_never_a_scale(self, detector, tmp_path,
                                                   resolution, unit):
        # Even a plausible-looking value: there is no telling a calibration from
        # a print setting, and in practice it is always the print setting.
        path = write_tiff(tmp_path / "res.tif", resolution=resolution, resolution_unit=unit)
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


class TestUnitsAreReadAsWritten:
    """A misread unit is a silent factor of a thousand on every measurement."""

    @pytest.mark.parametrize("mu", ["µ", "μ"])   # micro sign, Greek mu
    def test_either_mu_means_micrometres(self, detector, tmp_path, mu):
        path = write_tiff(tmp_path / "jeol.tif",
                          description=f"JEOL JSM\nPixelSize = 0.5 {mu}m\n".encode("utf-8"))
        result = detector.detect_scale_from_metadata(str(path))
        assert result["conversion"] == pytest.approx(500.0)

    def test_a_zeiss_pixel_size_in_nm_is_not_taken_for_metres(self, detector):
        assert detector._parse_zeiss_metadata(
            "SmartSEM\nPixel Size = 2.791 nm\n", {}) == pytest.approx(2.791)

    def test_a_bare_ap_pixel_size_is_in_metres(self, detector):
        assert detector._parse_zeiss_metadata(
            "AP_PIXEL_SIZE = 2.791e-09\n", {}) == pytest.approx(2.791)


class TestAPrintResolutionIsIgnored:
    def test_the_tiff_resolution_tag_is_not_a_scale(self, detector, tmp_path):
        # Programs write a print resolution here. On the 17 test-set files that
        # had nothing else it was wrong every time, by up to 130,000 times.
        path = write_tiff(tmp_path / "dpi400.tif", resolution=(400, 400), resolution_unit=2)
        with pytest.raises(sc.ScaleError):
            sc.from_metadata(detector, path)


class TestZeissStoredResolution:
    def tag(self, pixel, image_pixel, store):
        return {34118: {"ap_pixel_size": ("Pixel Size", pixel, "nm"),
                        "ap_image_pixel_size": ("Image Pixel Size", image_pixel, "nm"),
                        "dp_image_store": ("Store resolution", store),
                        "ap_width": ("Width", 108.0, "µm")}}

    def test_an_image_stored_at_2048_uses_the_stored_image_pixel_size(self, detector):
        # Pixel Size is the reference scan's; at 2048 x 1536 each stored pixel
        # is half of it. Taking it made these frames twice too large.
        tags = self.tag(105.5, 52.73, "2048 * 1536")
        assert detector._parse_zeiss_cz_sem_tag(tags) == pytest.approx(52.73)

    def test_the_text_form_prefers_image_pixel_size(self, detector):
        text = "SmartSEM\nPixel Size = 105.5 nm\nImage Pixel Size = 52.73 nm\n"
        assert detector._parse_zeiss_metadata(text, {}) == pytest.approx(52.73)


JEOL_SIDECAR = (
    "$CM_FORMAT TIFF\n$CM_INSTRUMENT JSM 6500F\n$CM_MAG 15000\n"
    "$$SM_MICRON_BAR 160\n$$SM_MICRON_MARKER 1\xb5m\n$$SM_COLUMN_MODE SEM\n"
    "$CM_IMAGE_RES 1280x1024\n")


class TestJeolSidecar:
    """A JEOL SEM records the scale only in a .txt beside the image."""

    def write(self, tmp_path, name="S1_0001", ext=".tif", width=1280, text=JEOL_SIDECAR):
        image = tmp_path / f"{name}{ext}"
        if ext == ".tif":
            tifffile.imwrite(str(image), np.zeros((16, width), np.uint8),
                             resolution=(96, 96), resolutionunit=2)
        else:
            from PIL import Image
            Image.fromarray(np.zeros((16, width), np.uint8)).save(image)
        (tmp_path / f"{name}.txt").write_bytes(text.encode("latin-1"))
        return image

    def test_the_pixel_size_comes_from_the_bar_fields(self, detector, tmp_path):
        image = self.write(tmp_path)
        calibration = sc.from_metadata(detector, image)
        assert calibration.nm_per_px == pytest.approx(1000 / 160)
        assert calibration.trustworthy

    def test_it_works_for_a_jpeg_too(self, detector, tmp_path):
        image = self.write(tmp_path, ext=".jpg")
        assert sc.from_metadata(detector, image).nm_per_px == pytest.approx(6.25)

    def test_an_image_saved_at_half_size_has_pixels_twice_as_large(self, detector, tmp_path):
        image = self.write(tmp_path, width=640)
        assert sc.from_metadata(detector, image).nm_per_px == pytest.approx(12.5)

    def test_a_sidecar_without_the_bar_fields_is_not_a_scale(self, detector, tmp_path):
        image = self.write(tmp_path, text="$CM_INSTRUMENT JSM 6500F\n")
        with pytest.raises(sc.ScaleError):
            sc.from_metadata(detector, image)

    def test_the_extension_may_be_upper_case(self, detector, tmp_path):
        image = self.write(tmp_path)
        (tmp_path / "S1_0001.txt").rename(tmp_path / "S1_0001.TXT")
        assert sc.from_metadata(detector, image).nm_per_px == pytest.approx(6.25)


class TestAmtCamera:
    def test_xpixcal_is_pixels_per_unit(self, detector, tmp_path):
        description = ("NIOSH\r8/30/2023\r40000\rXpixCal=620.662000\r"
                       "YpixCal=620.662000\rUnit=um\r##fv3\r2048")
        path = write_tiff(tmp_path / "amt.tif", description=description,
                          resolution=(293, 293), resolution_unit=2)
        calibration = sc.from_metadata(detector, path)
        assert calibration.nm_per_px == pytest.approx(1000 / 620.662)
        assert calibration.trustworthy
