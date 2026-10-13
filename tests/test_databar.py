"""
Databar detection and cropping.

The instrument databar has to come off before segmentation, or its text and
rules segment into spurious particles. But the height reported here is what
the Scale tab trims, so overestimating it silently discards real
micrograph — and claiming a databar on an image that has none discards up to 15%
of the frame for nothing.

Needs no OCR: the databar is found geometrically.
"""

import numpy as np
import pytest

from sem_particle_analysis import ScaleDetector
from synthetic import (make_burnin_micrograph, make_grid_bar_frame, make_jeol_frame,
                       make_micrograph, make_zeiss_frame)


@pytest.fixture(scope="module")
def detector():
    return ScaleDetector(use_gpu=False)


class TestFindsTheEdge:
    def test_measures_a_dark_databar_exactly(self, detector):
        image, truth = make_micrograph(image_height=704, databar_height=96)
        found = detector.detect_databar(image)
        assert found["has_databar"] is True
        # The edge is a real boundary, so this should be exact, not approximate.
        assert found["databar_height"] == truth["databar_height"]

    def test_measures_a_light_databar_exactly(self, detector):
        # Not every instrument draws white-on-black.
        image, truth = make_micrograph(image_height=704, databar_height=96,
                                       databar_gray=235, bar_gray=0)
        found = detector.detect_databar(image)
        assert found["has_databar"] is True
        assert found["databar_height"] == truth["databar_height"]

    @pytest.mark.parametrize("databar_height", [48, 72, 96, 120])
    def test_does_not_overshoot_at_various_heights(self, detector, databar_height):
        # The old largest-window-first scan reported the biggest window that
        # looked uniform rather than the edge, so it ate into the micrograph.
        image, truth = make_micrograph(image_height=800, databar_height=databar_height)
        found = detector.detect_databar(image)
        assert found["databar_height"] <= truth["databar_height"], "crop would eat micrograph"
        assert found["databar_height"] == truth["databar_height"]

    def test_reports_the_fraction_consistently(self, detector):
        image, truth = make_micrograph(image_height=704, databar_height=96)
        found = detector.detect_databar(image)
        total = truth["image_height"] + truth["databar_height"]
        assert found["databar_fraction"] == pytest.approx(found["databar_height"] / total)


class TestRejectsNonDatabars:
    def test_a_bare_micrograph_has_no_databar(self, detector):
        # A low-contrast frame used to read as one big databar and lose 15% of
        # its height to the crop.
        image, _ = make_micrograph(databar_height=0)
        assert detector.detect_databar(image)["has_databar"] is False

    def test_a_uniform_image_has_no_databar(self, detector):
        flat = np.full((600, 800, 3), 60, dtype=np.uint8)
        assert detector.detect_databar(flat)["has_databar"] is False

    def test_pure_noise_has_no_databar(self, detector):
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, size=(600, 800, 3), dtype=np.uint8)
        assert detector.detect_databar(noise)["has_databar"] is False

    def test_a_strip_too_thin_to_be_a_bar_is_ignored(self, detector):
        image, _ = make_micrograph(databar_height=0)
        image[-4:, :] = 0        # a 4px black edge, below DATABAR_MIN_HEIGHT_PX
        assert detector.detect_databar(image)["has_databar"] is False

    def test_a_strip_taller_than_the_limit_is_ignored(self, detector):
        # Beyond DATABAR_MAX_FRACTION this is not a databar, it is half the
        # picture, and cropping it would throw the measurement away.
        image, _ = make_micrograph(image_height=704, databar_height=0)
        image[-300:, :] = 0
        assert detector.detect_databar(image)["has_databar"] is False


class TestMetadataTakesPrecedence:
    def test_fei_scan_resolution_gives_the_height_directly(self, detector):
        # FEI records the scan height; anything below it is databar.
        image, _ = make_micrograph(image_height=704, databar_height=96)
        metadata = {"raw_tags": {34682: {"Image": {"ResolutionY": 704}}}}
        found = detector.detect_databar(image, metadata=metadata)
        assert found["has_databar"] is True
        assert found["databar_height"] == 96

    def test_a_scan_height_matching_the_image_means_no_databar(self, detector):
        image, _ = make_micrograph(image_height=704, databar_height=96)
        total = image.shape[0]
        metadata = {"raw_tags": {34682: {"Image": {"ResolutionY": total}}}}
        # No rows are unaccounted for, so metadata reports nothing and the
        # geometric pass takes over.
        assert detector.detect_databar(image, metadata=metadata)["databar_height"] == 96

    def test_unusable_metadata_falls_through_to_the_image(self, detector):
        image, truth = make_micrograph(image_height=704, databar_height=96)
        metadata = {"raw_tags": {34682: {"Image": {"ResolutionY": "not a number"}}}}
        found = detector.detect_databar(image, metadata=metadata)
        assert found["databar_height"] == truth["databar_height"]


class TestCropping:
    def test_cropping_the_measured_height_leaves_the_micrograph_intact(self, detector):
        image, truth = make_micrograph(image_height=704, databar_height=96)
        found = detector.detect_databar(image)
        kept = image[: image.shape[0] - found["databar_height"]]
        assert kept.shape[0] == truth["image_height"]


class TestCasesFromTheRealInstruments:
    """
    Failures seen on the NIOSH set, kept so they cannot come back.

    Both were silent: one cropped away part of the micrograph, the other cropped
    a frame that had nothing to crop.
    """

    def test_a_wide_rule_inside_the_bar_does_not_end_it(self, detector):
        # FEI draws the scale bar across nearly half the databar's width, so the
        # row carrying it looks nothing like the background. Stopping there
        # reported a 96px databar as 35px and left the rest to be segmented.
        image, truth = make_micrograph(
            width=1536, image_height=1024, databar_height=96,
            bar_length_px=700,        # ~46% of the width, as FEI draws it
            bar_left=760, scale_text="1 mm", scale_nm=1_000_000.0,
        )
        found = detector.detect_databar(image)
        assert found["has_databar"] is True
        assert found["databar_height"] == truth["databar_height"]

    def test_a_grid_bar_across_the_corner_is_not_a_databar(self, detector):
        # A TEM specimen grid bar is dark, large and contrasts strongly with the
        # film, but it is micrograph. Cropping it discards data — and on the real
        # set it would have removed up to 13% of nine frames.
        assert detector.detect_databar(make_grid_bar_frame())["has_databar"] is False

    def test_background_shading_towards_the_bottom_is_not_a_databar(self, detector):
        # Uneven illumination leaves the bottom of a TEM frame a little darker
        # and featureless. It passes every other test, and on four NIOSH frames
        # was cropped off with 5-9% of the micrograph; what it lacks is a step.
        image = np.full((1040, 1350, 3), 200, dtype=np.uint8)
        image[:900] = 210
        rows = np.linspace(210, 180, 140).astype(np.uint8)
        image[900:] = rows[:, None, None]
        rng = np.random.default_rng(1)
        noisy = np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
        noisy[:600, 300:400] = 40          # a particle well above, for contrast
        assert detector.detect_databar(noisy)["has_databar"] is False

    def test_a_burned_in_scale_bar_frame_has_no_databar(self, detector):
        # TEM frames carry the bar inside the image; there is no strip to trim.
        image, _ = make_burnin_micrograph()
        assert detector.detect_databar(image)["has_databar"] is False


class TestStripsTheTestSetTurnedUp:
    """Two databar layouts the detector never found on the 286 test images."""

    def test_an_inset_strip_framed_by_rules(self, detector):
        # Zeiss: a white strip boxed by black rules, a few rows above the edge,
        # with micrograph below it. The crop starts at the strip's top rule.
        image, truth = make_zeiss_frame()
        found = detector.detect_databar(image)
        assert found["has_databar"] is True
        assert image.shape[0] - found["databar_height"] == truth["databar_top"]

    def test_a_strip_with_the_bar_in_its_top_rows(self, detector):
        # JEOL: the bar crosses the strip's first rows, which made them look like
        # the ragged top of a grid bar.
        image, truth = make_jeol_frame()
        found = detector.detect_databar(image)
        assert found["has_databar"] is True
        assert image.shape[0] - found["databar_height"] == truth["databar_top"]
