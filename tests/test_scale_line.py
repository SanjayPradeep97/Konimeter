"""
Choosing which horizontal run is the scale bar.

Every case here is drawn from a failure seen on the real NIOSH TEM set, where
picking the widest run instead of the most bar-like one put the wrong polarity
on most frames and produced scale errors of up to 2.6x — silently, because a
wrong reading still looks like a number.

No OCR needed: this is the geometry half of the detector.
"""

import numpy as np
import pytest

from sem_particle_analysis import ScaleDetector


@pytest.fixture(scope="module")
def detector():
    return ScaleDetector(use_gpu=False)


def blank(height=120, width=400):
    return np.zeros((height, width), dtype=np.uint8)


def solid_bar(image, x, y, width, thickness):
    """A drawn bar: a filled rectangle."""
    image[y:y + thickness, x:x + width] = 255
    return image


def ragged_run(image, x, y, width, thickness, keep=0.5, seed=0):
    """A run of the same extent but only partly filled, as thresholded noise is."""
    rng = np.random.default_rng(seed)
    patch = rng.random((thickness, width)) < keep
    image[y:y + thickness, x:x + width][patch] = 255
    return image


class TestPrefersASolidBar:
    def test_a_solid_bar_is_found(self, detector):
        image = solid_bar(blank(), x=40, y=60, width=150, thickness=12)
        left, right, row, spanned, area = detector._find_scale_line_morphological(
            image, max_thickness=20)
        assert right - left == pytest.approx(149, abs=2)
        assert not spanned
        assert area == pytest.approx(150 * 12, rel=0.05)

    def test_a_wider_ragged_run_does_not_beat_a_solid_bar(self, detector):
        # The real failure: a ragged run through thresholded background was
        # wider than the bar, so it won.
        image = blank()
        solid_bar(image, x=40, y=70, width=150, thickness=12)
        ragged_run(image, x=10, y=20, width=300, thickness=10, keep=0.45)

        left, right, _row, _spanned, _area = detector._find_scale_line_morphological(
            image, max_thickness=20)
        assert 38 <= left <= 42, "picked the ragged run, not the bar"
        assert right - left == pytest.approx(149, abs=3)

    def test_a_ragged_run_alone_is_rejected(self, detector):
        image = ragged_run(blank(), x=10, y=20, width=300, thickness=10, keep=0.4)
        assert detector._find_scale_line_morphological(image, max_thickness=20) is None

    def test_fill_threshold_is_the_deciding_property(self, detector):
        dense = ragged_run(blank(), x=20, y=40, width=200, thickness=8, keep=0.99)
        sparse = ragged_run(blank(), x=20, y=40, width=200, thickness=8, keep=0.5)
        assert detector._find_scale_line_morphological(dense, max_thickness=20) is not None
        assert detector._find_scale_line_morphological(sparse, max_thickness=20) is None


class TestPrefersTheThickerBarOverItsOwnEdge:
    def test_a_thin_sliver_slightly_wider_does_not_win(self, detector):
        # Thresholding a bar at the wrong level yields a one- or two-pixel-tall
        # line along its edge, a few pixels wider than the bar itself. It is
        # solid, so only ranking by area keeps it from winning and measuring long.
        image = blank()
        solid_bar(image, x=40, y=60, width=150, thickness=12)
        solid_bar(image, x=36, y=40, width=158, thickness=2)

        left, right, _row, _spanned, area = detector._find_scale_line_morphological(
            image, max_thickness=20)
        assert right - left == pytest.approx(149, abs=3), "measured the edge, not the bar"
        assert area > 158 * 2


class TestExistingGuardsStillHold:
    def test_a_component_thicker_than_the_limit_is_rejected(self, detector):
        image = solid_bar(blank(), x=10, y=10, width=300, thickness=60)
        assert detector._find_scale_line_morphological(image, max_thickness=20) is None

    def test_a_square_blob_is_rejected_by_the_aspect_test(self, detector):
        image = solid_bar(blank(), x=50, y=40, width=30, thickness=18)
        assert detector._find_scale_line_morphological(image, max_thickness=20) is None

    def test_an_empty_image_yields_nothing(self, detector):
        assert detector._find_scale_line_morphological(blank(), max_thickness=20) is None

    def test_a_bar_spanning_the_crop_is_flagged_as_such(self, detector):
        image = blank(height=60, width=200)
        solid_bar(image, x=0, y=25, width=200, thickness=8)
        result = detector._find_scale_line_morphological(image, max_thickness=20)
        assert result is not None
        assert result[3] is True, "should report that it ran to the crop edge"

    def test_a_bar_with_clear_margins_beats_one_spanning_the_crop(self, detector):
        image = blank(height=120, width=300)
        solid_bar(image, x=0, y=20, width=300, thickness=10)   # spans
        solid_bar(image, x=60, y=70, width=120, thickness=10)  # clear margins
        left, right, _row, spanned, _area = detector._find_scale_line_morphological(
            image, max_thickness=20)
        assert spanned is False
        assert right - left == pytest.approx(119, abs=3)
