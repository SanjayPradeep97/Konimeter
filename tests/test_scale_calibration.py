"""
Scale calibration arithmetic and guard rails.

Every reported size is a pixel count multiplied by nm_per_px, so an error here
rescales a whole dataset silently. These tests need neither OCR nor SAM.
"""

import math

import pytest

from sem_particle_analysis import scale_calibration as sc


class TestValidateNmPerPx:
    @pytest.mark.parametrize("value", [0, -1, -0.001])
    def test_rejects_non_positive(self, value):
        with pytest.raises(sc.ScaleError):
            sc.validate_nm_per_px(value)

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
    def test_rejects_non_finite(self, value):
        with pytest.raises(sc.ScaleError):
            sc.validate_nm_per_px(value)

    def test_rejects_outside_electron_microscopy_range(self):
        low, high = sc.VALID_NM_PER_PX
        with pytest.raises(sc.ScaleError):
            sc.validate_nm_per_px(low / 10)
        with pytest.raises(sc.ScaleError):
            sc.validate_nm_per_px(high * 10)

    @pytest.mark.parametrize("value", [0.01, 1.0, 2.5, 1000.0])
    def test_accepts_plausible_values(self, value):
        assert sc.validate_nm_per_px(value) == value


class TestUnitConversion:
    @pytest.mark.parametrize(
        "value,unit,expected",
        [
            (1, "nm", 1.0),
            (1, "µm", 1000.0),
            (1, "um", 1000.0),
            (2.5, "µm", 2500.0),
            (1, "mm", 1_000_000.0),
            (1, "Å", 0.1),
            (1, "pm", 0.001),
        ],
    )
    def test_to_nanometres(self, value, unit, expected):
        assert sc.to_nanometres(value, unit) == pytest.approx(expected)

    def test_micron_spellings_agree(self):
        assert sc.to_nanometres(1, "µm") == sc.to_nanometres(1, "um")

    def test_rejects_unknown_unit(self):
        with pytest.raises(sc.ScaleError):
            sc.to_nanometres(1, "furlong")

    @pytest.mark.parametrize("value", [0, -5, None])
    def test_rejects_bad_length(self, value):
        with pytest.raises(sc.ScaleError):
            sc.to_nanometres(value, "nm")

    @pytest.mark.parametrize(
        "nanometres,expected",
        [(500, "500 nm"), (1000, "1 µm"), (2500, "2.5 µm"), (1_000_000, "1 mm")],
    )
    def test_format_length_picks_microscopist_units(self, nanometres, expected):
        assert sc.format_length(nanometres) == expected


class TestFromTwoPoints:
    def test_computes_nm_per_pixel_from_a_horizontal_bar(self):
        # A 200 px bar labelled 500 nm is 2.5 nm/px.
        cal = sc.from_two_points((100, 50), (300, 50), 500, "nm")
        assert cal.nm_per_px == pytest.approx(2.5)
        assert cal.pixel_length == pytest.approx(200)
        assert cal.scale_nm == pytest.approx(500)

    def test_uses_euclidean_distance_for_a_tilted_bar(self):
        cal = sc.from_two_points((0, 0), (30, 40), 500, "nm")
        assert cal.pixel_length == pytest.approx(50)
        assert cal.nm_per_px == pytest.approx(10)

    def test_converts_the_typed_unit(self):
        # 1 µm over 200 px is 5 nm/px, not 0.005.
        cal = sc.from_two_points((0, 0), (200, 0), 1, "µm")
        assert cal.nm_per_px == pytest.approx(5.0)

    def test_is_trusted_without_confirmation(self):
        # Nothing here was machine-read, so there is no glyph to second-guess.
        cal = sc.from_two_points((0, 0), (200, 0), 500, "nm")
        assert cal.confirmed is True
        assert cal.trustworthy is True
        assert cal.warning is None

    def test_rejects_coincident_points(self):
        with pytest.raises(sc.ScaleError):
            sc.from_two_points((100, 100), (100, 101), 500, "nm")

    def test_rejects_an_implausible_result(self):
        # 1 mm across 2 px would be 500,000 nm/px — outside the valid range.
        with pytest.raises(sc.ScaleError):
            sc.from_two_points((0, 0), (2, 0), 1, "mm")


class TestScaleCalibration:
    def test_metadata_is_trusted_but_ocr_is_not(self):
        metadata = sc.ScaleCalibration(nm_per_px=2.5, method="metadata")
        ocr = sc.ScaleCalibration(nm_per_px=2.5, method="box_ocr")
        assert metadata.trustworthy is True
        assert ocr.trustworthy is False

    def test_confirming_makes_an_ocr_reading_trustworthy(self):
        ocr = sc.ScaleCalibration(nm_per_px=2.5, method="box_ocr")
        ocr.confirmed = True
        assert ocr.trustworthy is True

    def test_construction_validates(self):
        with pytest.raises(sc.ScaleError):
            sc.ScaleCalibration(nm_per_px=-1, method="metadata")

    def test_to_dict_carries_the_conversion_key_the_app_reads(self):
        # callbacks and the analyzer both read scale_info['conversion']; renaming
        # or dropping it silently drops every image back to pixel units.
        data = sc.ScaleCalibration(nm_per_px=2.5, method="metadata").to_dict()
        assert data["conversion"] == pytest.approx(2.5)
        assert data["nm_per_px"] == pytest.approx(2.5)
        assert data["method_label"] == "file metadata"

    def test_summary_mentions_value_and_provenance(self):
        cal = sc.from_two_points((0, 0), (200, 0), 500, "nm")
        summary = cal.summary()
        assert "2.5" in summary
        assert "two points" in summary

    def test_method_label_falls_back_to_the_raw_method(self):
        assert sc.ScaleCalibration(nm_per_px=1.0, method="odd").method_label == "odd"


class TestFromMetadataErrors:
    def test_raises_scale_error_when_detector_finds_nothing(self):
        class NoScale:
            def detect_scale_from_metadata(self, file_path):
                return {}

        with pytest.raises(sc.ScaleError):
            sc.from_metadata(NoScale(), "image.tif")

    def test_wraps_detector_exceptions(self):
        class Broken:
            def detect_scale_from_metadata(self, file_path):
                raise RuntimeError("no metadata")

        with pytest.raises(sc.ScaleError):
            sc.from_metadata(Broken(), "image.tif")


class TestNormaliseBox:
    def test_orders_corners_and_clamps_to_the_image(self):
        # Dragged bottom-right to top-left, and off the edge of a 100x80 image.
        x0, y0, x1, y1 = sc._normalise_box((120, 90, -10, -5), 100, 80)
        assert (x0, y0) == (0, 0)
        assert x1 <= 100 and y1 <= 80
        assert x0 < x1 and y0 < y1

    def test_math_dist_matches_pixel_length(self):
        cal = sc.from_two_points((10, 10), (10, 60), 100, "nm")
        assert cal.pixel_length == pytest.approx(math.dist((10, 10), (10, 60)))


class TestProvenance:
    """
    What the results file records about where a scale came from.

    The plain method name means something vouched for it; the "+unconfirmed"
    suffix marks the rows worth going back to.
    """

    def test_metadata_needs_no_qualification(self):
        cal = sc.ScaleCalibration(nm_per_px=2.5, method="metadata")
        assert cal.provenance == "metadata"

    def test_an_unchecked_reading_is_marked(self):
        cal = sc.ScaleCalibration(nm_per_px=2.5, method="box_ocr")
        assert cal.provenance == "box_ocr+unconfirmed"

    def test_confirming_removes_the_mark(self):
        cal = sc.ScaleCalibration(nm_per_px=2.5, method="box_ocr")
        cal.confirmed = True
        assert cal.provenance == "box_ocr"

    def test_two_clicked_points_are_vouched_for_as_they_are_made(self):
        cal = sc.from_two_points((0, 0), (100, 0), 1, "µm")
        assert cal.provenance == "two_points"
