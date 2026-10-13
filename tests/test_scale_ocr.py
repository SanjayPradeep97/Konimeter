"""
Reading a printed scale bar — tier 2.

Checked against synthetic micrographs whose bar length in pixels and printed
label are known exactly, so a misread shows up as a wrong nm/px rather than as a
plausible-looking number.

Marked slow: EasyOCR downloads and loads detection models on first use.
"""

import numpy as np
import pytest

from synthetic import (databar_region, make_burnin_micrograph, make_fei_frame,
                       make_jeol_frame, make_micrograph, make_zeiss_frame)

pytestmark = pytest.mark.slow


class TestReadsASyntheticBar:
    def test_recovers_the_known_nm_per_pixel(self, scale_detector):
        # 500 nm printed over a bar drawn exactly 200 px long -> 2.5 nm/px.
        image, truth = make_micrograph(bar_length_px=200, scale_text="500 nm",
                                       scale_nm=500.0)
        result = scale_detector.detect_scale_bar(image, **databar_region(truth))
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.05)
        assert result["scale_nm"] == pytest.approx(500.0)
        # Exactly: the bar covers 200 pixels, and a reading one short overstates
        # nm/px by 1/200 for every measurement made with it.
        assert result["pixel_length"] == 200

    def test_reads_a_micron_label(self, scale_detector):
        image, truth = make_micrograph(bar_length_px=250, scale_text="1 um",
                                       scale_nm=1000.0)
        result = scale_detector.detect_scale_bar(image, **databar_region(truth))
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.1)

    def test_reads_a_bar_on_a_light_databar(self, scale_detector):
        # Not every instrument draws white-on-black.
        image, truth = make_micrograph(databar_gray=235, bar_gray=0,
                                       bar_length_px=200, scale_text="500 nm",
                                       scale_nm=500.0)
        result = scale_detector.detect_scale_bar(image, **databar_region(truth))
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.1)


class TestGuardRails:
    def test_a_non_standard_value_is_flagged_for_confirmation(self, scale_detector):
        # Instruments label bars with a 1/2/5 mantissa. "700 nm" is off that grid
        # and is nearly always a misread digit — usable, but it must warn.
        image, truth = make_micrograph(bar_length_px=200, scale_text="700 nm",
                                       scale_nm=700.0)
        result = scale_detector.detect_scale_bar(image, **databar_region(truth))
        assert result.get("warning")

    def test_a_standard_value_is_not_flagged(self, scale_detector):
        image, truth = make_micrograph(bar_length_px=200, scale_text="500 nm",
                                       scale_nm=500.0)
        result = scale_detector.detect_scale_bar(image, **databar_region(truth))
        assert not result.get("warning")

    def test_a_region_with_no_bar_raises(self, scale_detector):
        image, truth = make_micrograph()
        with pytest.raises(ValueError):
            # The middle of the micrograph, nowhere near the databar.
            scale_detector.detect_scale_bar(
                image, region_x=0.5, region_y=0.3,
                region_width=0.2, region_height=0.1)

    def test_a_blank_image_raises(self, scale_detector):
        blank = np.full((400, 600, 3), 128, dtype=np.uint8)
        with pytest.raises(ValueError):
            scale_detector.detect_scale_bar(blank)


class TestStandardValueClassification:
    @pytest.mark.parametrize("nanometres", [100.0, 200.0, 500.0, 1000.0, 2500.0])
    def test_grid_values_are_standard(self, scale_detector, nanometres):
        assert scale_detector._is_standard_scale_value(nanometres)

    @pytest.mark.parametrize("nanometres", [700.0, 1100.0, 333.0])
    def test_off_grid_values_are_not_standard(self, scale_detector, nanometres):
        assert not scale_detector._is_standard_scale_value(nanometres)




class TestBurnedInBar:
    """
    A TEM-style frame: light field, no databar, solid dark bar inside the image.

    On the real set this layout was read with the wrong polarity on most frames,
    giving scale errors up to 2.6x with no error raised.
    """

    def test_reads_a_dark_bar_on_a_light_field(self, scale_detector):
        image, truth = make_burnin_micrograph(bar_length_px=128, scale_text="1 um",
                                              scale_nm=1000.0)
        result = scale_detector.detect_scale_bar_anywhere(image)
        assert result["polarity_used"] == "dark"
        assert result["pixel_length"] == pytest.approx(truth["bar_length_px"], abs=6)
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.06)

    def test_measures_the_bar_not_the_field(self, scale_detector):
        # The failure mode was a long ragged run through the noisy background
        # winning on width. Its length bore no relation to the bar's.
        image, truth = make_burnin_micrograph(bar_length_px=96, noise=14.0,
                                              scale_text="500 nm", scale_nm=500.0)
        result = scale_detector.detect_scale_bar_anywhere(image)
        assert result["pixel_length"] == pytest.approx(96, abs=8)

    @pytest.mark.parametrize("length,text,nanometres",
                             [(104, "0.5 um", 500.0), (223, "5 um", 5000.0),
                              (155, "2 um", 2000.0)])
    def test_recovers_various_bars(self, scale_detector, length, text, nanometres):
        image, truth = make_burnin_micrograph(bar_length_px=length, scale_text=text,
                                              scale_nm=nanometres)
        result = scale_detector.detect_scale_bar_anywhere(image)
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.08)

    def test_a_clear_reading_is_not_flagged(self, scale_detector):
        # Warning on every polarity disagreement made a correct reading lose to
        # an unflagged wrong one, because the region sweep prefers unflagged.
        image, _ = make_burnin_micrograph(bar_length_px=128, scale_text="1 um",
                                          scale_nm=1000.0)
        assert not scale_detector.detect_scale_bar_anywhere(image).get("warning")


class TestTheAutomaticReadSearchesEverywhere:
    """
    When a file has no pixel size, the Scale tab reads the printed bar with
    ``detect_scale_bar_anywhere``. Looking only in the bottom-right suits an SEM
    databar and misses a TEM bar burned into the bottom-left completely.
    """

    def test_it_finds_a_bottom_left_bar(self, scale_detector):
        image, truth = make_burnin_micrograph(bar_length_px=128, scale_text="1 um",
                                              scale_nm=1000.0)
        result = scale_detector.detect_scale_bar_anywhere(image)
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.06)

    def test_it_finds_a_smaller_bottom_left_bar(self, scale_detector):
        image, truth = make_burnin_micrograph(bar_length_px=104, scale_text="0.5 um",
                                              scale_nm=500.0)
        result = scale_detector.detect_scale_bar_anywhere(image)
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.08)

    def test_it_still_finds_a_databar_bar(self, scale_detector):
        image, truth = make_micrograph(bar_length_px=200, scale_text="500 nm",
                                       scale_nm=500.0, bar_left=620)
        result = scale_detector.detect_scale_bar_anywhere(image)
        assert result["conversion"] == pytest.approx(truth["nm_per_px"], rel=0.08)

    def test_a_box_restricts_the_search_to_the_box(self, scale_detector):
        # The box drawn on the Scale tab means "look here", not "look here first".
        image, _ = make_burnin_micrograph(bar_length_px=128)
        with pytest.raises(ValueError):
            scale_detector.detect_scale_bar(
                image, region_x=0.75, region_y=0.5, region_width=0.3, region_height=0.1)


class TestTheBarBesideItsLabel:
    """
    The label is found first and the bar beside it measured. Taking the largest
    line in the region instead measured a databar divider, half an FEI bar, the
    "=" of "Mag =", and gave Zeiss bars their ticks' outer edges.
    """

    def test_a_zeiss_bar_is_measured_between_its_ticks(self, scale_detector):
        image, truth = make_zeiss_frame()
        found = scale_detector.detect_scale_bar_anywhere(image)
        assert found["scale_nm"] == pytest.approx(truth["scale_nm"])
        assert found["pixel_length"] == pytest.approx(truth["bar_length_px"], abs=1)

    def test_a_date_is_not_read_as_angstroms(self, scale_detector):
        # "20 Aug" in the Zeiss strip once parsed as 20 Å.
        image, truth = make_zeiss_frame()
        found = scale_detector.detect_scale_bar_anywhere(image)
        assert found["scale_nm"] != pytest.approx(2.0)

    def test_an_fei_bar_split_by_its_label_is_one_bar(self, scale_detector):
        image, truth = make_fei_frame()
        found = scale_detector.detect_scale_bar_anywhere(image)
        assert found["scale_nm"] == pytest.approx(truth["scale_nm"])
        assert found["pixel_length"] == pytest.approx(truth["bar_length_px"], abs=2)

    def test_a_jeol_label_among_instrument_settings(self, scale_detector):
        # 15.0kV, X15,000 and 15.0mm are all numbers with units; only 1µm is
        # the bar's.
        image, truth = make_jeol_frame()
        found = scale_detector.detect_scale_bar_anywhere(image)
        assert found["scale_nm"] == pytest.approx(truth["scale_nm"])
        assert found["pixel_length"] == pytest.approx(truth["bar_length_px"], abs=1)

    def test_the_mark_covers_the_bar_and_label_only(self, scale_detector):
        image, truth = make_zeiss_frame()
        found = scale_detector.detect_scale_bar_anywhere(image)
        x, y, w, h = found["mark"]
        assert w < image.shape[1] // 3


class TestLabelValues:
    """Repairs to what OCR makes of a label, checked without OCR."""

    @pytest.mark.parametrize("text, nanometres", [
        ("5 µm", 5000), ("0 5 Hmn", 500), ("Ium", 1000), ("U2 um", 200),
        (".5 um", 500), ("05 µm", 500), ("20 um*", 20000), ("400 nm", 400),
        ("1 mm", 1_000_000), ("200 nm)", 200), ("0.5 Mm", 500),
    ])
    def test_repairs(self, scale_detector, text, nanometres):
        assert scale_detector._label_value(text) == pytest.approx(nanometres)

    @pytest.mark.parametrize("text", ["15.0kV", "X15,000", "Teneo", "Mag =", "20 Aug", "WD"])
    def test_not_labels(self, scale_detector, text):
        assert scale_detector._label_value(text) is None
