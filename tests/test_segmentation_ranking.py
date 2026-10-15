"""
Mask candidate ranking.

SAM's full-image box prompt returns three candidates, and either polarity of each
may be the one holding the objects — six possibilities, of which five are wrong.
Picking by confidence score or by area chooses wrongly often enough to matter, so
the ranking is by contrast. Getting this wrong analyses the background and
reports plausible-looking numbers for it, which is the worst failure mode in the
pipeline. None of this needs SAM weights.
"""

import numpy as np
import pytest

from sem_particle_analysis import ParticleSegmenter


BACKGROUND = 40
BRIGHT = 200
DARK = 5


def scene(feature_value, background=BACKGROUND, size=200, square=60):
    """A grey frame with one square of ``feature_value``, plus its boolean mask."""
    image = np.full((size, size, 3), background, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    lo = (size - square) // 2
    mask[lo:lo + square, lo:lo + square] = True
    image[mask] = feature_value
    return image, mask


@pytest.fixture
def segmenter():
    # rank_candidates never touches the model, so no checkpoint is needed.
    return ParticleSegmenter(sam_model=None)


class TestPolarity:
    def test_picks_the_side_holding_bright_features(self, segmenter):
        # SAM returned the objects directly, so the un-inverted mask is right.
        image, objects = scene(BRIGHT)
        best = segmenter.rank_candidates(image, np.array([objects]), top_k=3)
        assert best, "a high-contrast square should rank"
        assert best[0]["inverted"] is False
        assert np.array_equal(best[0]["mask"], objects)

    def test_picks_the_inverted_side_when_sam_returned_the_background(self, segmenter):
        # The same scene, but SAM labelled the background. The objects are what
        # the raw mask leaves out, so the winner must be the inverted candidate.
        image, objects = scene(BRIGHT)
        best = segmenter.rank_candidates(image, np.array([~objects]), top_k=3)
        assert best
        assert best[0]["inverted"] is True
        assert np.array_equal(best[0]["mask"], objects)

    def test_handles_dark_features_on_a_bright_field(self, segmenter):
        # The TEM case: electron-dense features are darker than the support film.
        image, objects = scene(DARK, background=BRIGHT)
        best = segmenter.rank_candidates(image, np.array([objects]), top_k=3)
        assert best
        assert np.array_equal(best[0]["mask"], objects)


class TestDarkFeaturesHint:
    """
    The expected polarity orders the candidates; it never hides any.

    It used to be a filter, and on frames whose instrument was guessed wrong the
    right outline was thrown away, leaving nothing to choose or only background.
    """

    def test_a_wrong_expectation_still_offers_the_objects(self, segmenter):
        image, objects = scene(BRIGHT)
        best = segmenter.rank_candidates(image, np.array([objects]),
                                         dark_features=True)
        assert best and np.array_equal(best[0]["mask"], objects)
        assert best[0]["darker"] is False

    def test_dark_features_false_accepts_bright_objects(self, segmenter):
        image, objects = scene(BRIGHT)
        best = segmenter.rank_candidates(image, np.array([objects]),
                                         dark_features=False)
        assert best and np.array_equal(best[0]["mask"], objects)

    def test_dark_features_true_accepts_dark_objects(self, segmenter):
        image, objects = scene(DARK, background=BRIGHT)
        best = segmenter.rank_candidates(image, np.array([objects]),
                                         dark_features=True)
        assert best and np.array_equal(best[0]["mask"], objects)

    def test_the_expected_side_comes_first_even_at_lower_contrast(self, segmenter):
        # A large dark particle filling 60% of a bright frame. It and the
        # background around it are both plausible outlines with the same
        # contrast; only the expectation can tell them apart.
        image = np.full((200, 200, 3), BRIGHT, dtype=np.uint8)
        particle = np.zeros((200, 200), dtype=bool)
        particle[:, :120] = True
        image[particle] = DARK
        dark_first = segmenter.rank_candidates(image, np.array([particle]),
                                               dark_features=True)
        bright_first = segmenter.rank_candidates(image, np.array([particle]),
                                                 dark_features=False)
        assert np.array_equal(dark_first[0]["mask"], particle)
        assert np.array_equal(bright_first[0]["mask"], ~particle)
        # Both sides are offered either way round.
        assert len(dark_first) == len(bright_first) == 2

    def test_without_an_expectation_the_smaller_side_comes_first(self, segmenter):
        image = np.full((200, 200, 3), BRIGHT, dtype=np.uint8)
        particle = np.zeros((200, 200), dtype=bool)
        particle[:, :70] = True                 # 35% of the frame
        image[particle] = DARK
        ranked = segmenter.rank_candidates(image, np.array([~particle]))
        assert np.array_equal(ranked[0]["mask"], particle)


class TestPlausibilityFilters:
    def test_a_particle_filling_most_of_the_frame_is_offered(self, segmenter):
        # At a 35% ceiling every large aggregate in the NIOSH demo set was
        # thrown out, and the analyst was offered nothing or the background.
        image = np.full((200, 200, 3), BRIGHT, dtype=np.uint8)
        particle = np.zeros((200, 200), dtype=bool)
        particle[10:180, 10:180] = True          # 72% of the frame
        image[particle] = DARK
        ranked = segmenter.rank_candidates(image, np.array([particle]),
                                           dark_features=True)
        assert ranked and np.array_equal(ranked[0]["mask"], particle)

    def test_a_mask_covering_almost_everything_is_read_as_its_complement(self, segmenter):
        image = np.full((200, 200, 3), BACKGROUND, dtype=np.uint8)
        nearly_all = np.ones((200, 200), dtype=bool)
        nearly_all[90:110, 90:110] = False      # a 1% particle; the rest is 99%
        image[~nearly_all] = BRIGHT
        ranked = segmenter.rank_candidates(image, np.array([nearly_all]))
        assert ranked[0]["inverted"] is True and ranked[0]["plausible"]
        # The 99% side is still offered, last, for the analyst to judge.
        assert len(ranked) == 2 and not ranked[1]["plausible"]

    def test_a_large_mask_is_read_as_its_smaller_complement(self, segmenter):
        # 90% foreground is implausible, but its 10% complement is not — and here
        # the complement really is the darker, distinct region. The ranker should
        # flip polarity rather than give up.
        image = np.full((200, 200, 3), BRIGHT, dtype=np.uint8)
        big = np.zeros((200, 200), dtype=bool)
        big[:180, :] = True
        image[~big] = BACKGROUND

        ranked = segmenter.rank_candidates(image, np.array([big]))
        assert ranked
        assert ranked[0]["inverted"] is True
        assert ranked[0]["fraction"] == pytest.approx(0.1)

    def test_a_candidate_covering_almost_nothing_fails_the_checks(self, segmenter):
        image = np.full((200, 200, 3), BACKGROUND, dtype=np.uint8)
        speck = np.zeros((200, 200), dtype=bool)
        speck[0:3, 0:3] = True          # 9 px of 40000 -> below MIN fraction
        image[speck] = BRIGHT
        ranked = segmenter.rank_candidates(image, np.array([speck]))
        assert ranked and not any(c["plausible"] for c in ranked)

    def test_a_low_contrast_candidate_fails_the_checks(self, segmenter):
        # Only 3 grey levels above background — below MIN_CONTRAST. Without this
        # a mask over background speckle outranks a real object.
        image, objects = scene(BACKGROUND + 3)
        ranked = segmenter.rank_candidates(image, np.array([objects]))
        assert ranked and not any(c["plausible"] for c in ranked)

    def test_failing_the_checks_ranks_after_passing_never_hides(self, segmenter):
        # Small particles on a porous filter: every one of SAM's outlines failed
        # the checks, and the analyst was offered nothing to start from.
        image = np.full((200, 200, 3), BACKGROUND, dtype=np.uint8)
        faint = np.zeros((200, 200), dtype=bool)
        faint[20:60, 20:60] = True
        strong = np.zeros((200, 200), dtype=bool)
        strong[120:160, 120:160] = True
        image[faint] = BACKGROUND + 3
        image[strong] = BRIGHT
        ranked = segmenter.rank_candidates(image, np.array([faint, strong]), top_k=3)
        assert len(ranked) == 3
        assert np.array_equal(ranked[0]["mask"], strong) and ranked[0]["plausible"]
        assert [c["plausible"] for c in ranked] == sorted(
            (c["plausible"] for c in ranked), reverse=True)

    def test_accepts_a_candidate_just_above_the_contrast_floor(self, segmenter):
        image, objects = scene(BACKGROUND + 40)
        assert segmenter.rank_candidates(image, np.array([objects]))


class TestOrdering:
    def test_higher_contrast_ranks_first(self, segmenter):
        image = np.full((200, 200, 3), BACKGROUND, dtype=np.uint8)
        faint = np.zeros((200, 200), dtype=bool)
        faint[20:60, 20:60] = True
        strong = np.zeros((200, 200), dtype=bool)
        strong[120:160, 120:160] = True
        image[faint] = BACKGROUND + 30
        image[strong] = BRIGHT

        ranked = segmenter.rank_candidates(image, np.array([faint, strong]), top_k=4)
        assert ranked[0]["contrast"] >= ranked[-1]["contrast"]
        assert np.array_equal(ranked[0]["mask"], strong)

    def test_respects_top_k(self, segmenter):
        image, objects = scene(BRIGHT)
        masks = np.array([objects, objects, objects])
        assert len(segmenter.rank_candidates(image, masks, top_k=2)) == 2

    def test_each_candidate_reports_its_provenance(self, segmenter):
        image, objects = scene(BRIGHT)
        candidate = segmenter.rank_candidates(image, np.array([objects]))[0]
        assert set(candidate) >= {"mask", "mask_index", "inverted", "fraction",
                                  "contrast", "score"}
        assert candidate["mask_index"] == 0
        assert candidate["fraction"] == pytest.approx(objects.mean())
        # No scores were set by a real SAM run, so score is simply absent.
        assert candidate["score"] is None


class TestExclude:
    def test_excluded_pixels_are_removed_from_every_candidate(self, segmenter):
        # A scale bar burned into the frame is high contrast and would otherwise
        # be measured as a particle.
        image = np.full((200, 200, 3), BACKGROUND, dtype=np.uint8)
        objects = np.zeros((200, 200), dtype=bool)
        objects[40:90, 40:90] = True    # a real particle
        bar = np.zeros((200, 200), dtype=bool)
        bar[180:188, 20:120] = True     # the burned-in bar
        image[objects] = BRIGHT
        image[bar] = BRIGHT

        ranked = segmenter.rank_candidates(image, np.array([objects | bar]),
                                           exclude=bar)
        assert ranked
        assert not (ranked[0]["mask"] & bar).any()
        assert (ranked[0]["mask"] & objects).any()

    def test_nothing_excluded_is_ever_offered(self, segmenter):
        image, objects = scene(BRIGHT)
        ranked = segmenter.rank_candidates(image, np.array([objects]), exclude=objects)
        assert all(not (c["mask"] & objects).any() for c in ranked)


class TestErrors:
    def test_ranking_without_masks_is_an_error(self, segmenter):
        image, _ = scene(BRIGHT)
        with pytest.raises(RuntimeError):
            segmenter.rank_candidates(image, masks=None)

    def test_falls_back_to_the_last_segmentation(self, segmenter):
        image, objects = scene(BRIGHT)
        segmenter.masks = np.array([objects])
        assert segmenter.rank_candidates(image)

    def test_accepts_a_greyscale_image(self, segmenter):
        image, objects = scene(BRIGHT)
        assert segmenter.rank_candidates(image[..., 0], np.array([objects]))


class TestMaskGeometryHelpers:
    def test_roi_box_pads_and_stays_inside_the_frame(self, segmenter):
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 30:50] = True
        (x0, y0, x1, y1) = segmenter._compute_roi_box(mask, pad=10)[0]
        assert 0 <= x0 < x1 <= 100
        assert 0 <= y0 < y1 <= 100
        assert x0 == 20 and y0 == 30

    def test_roi_box_of_an_empty_mask_is_the_whole_frame(self, segmenter):
        box = segmenter._compute_roi_box(np.zeros((80, 60), dtype=bool))[0]
        assert list(box) == [0, 0, 60, 80]

    def test_iou_of_identical_masks_is_one(self, segmenter):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True
        assert segmenter._mask_iou(mask, mask) == pytest.approx(1.0)

    def test_iou_of_disjoint_masks_is_zero(self, segmenter):
        a = np.zeros((50, 50), dtype=bool); a[0:10, 0:10] = True
        b = np.zeros((50, 50), dtype=bool); b[30:40, 30:40] = True
        assert segmenter._mask_iou(a, b) == 0.0

    def test_iou_of_two_empty_masks_is_zero_not_a_crash(self, segmenter):
        empty = np.zeros((10, 10), dtype=bool)
        assert segmenter._mask_iou(empty, empty) == 0.0
