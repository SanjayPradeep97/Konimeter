"""
Telling specimen from everything else in the frame.

Non-specimen area is the pipeline's largest source of wrong numbers, because it
is also the highest-contrast thing present. On the real set an SEM aperture
vignette *was* the five reported "particles", at a median 119 µm, and a TEM grid
bar produced 221 spurious ones. Both are the same shape of thing — a large,
near-black region reaching the frame edge — so one rule covers them.
"""

import numpy as np
import pytest

from sem_particle_analysis import region as reg
from synthetic import make_burnin_micrograph, make_grid_bar_frame, make_vignetted_frame


class TestBlockedArea:
    def test_finds_the_corners_of_a_vignetted_frame(self):
        image, truth = make_vignetted_frame()
        blocked = reg.blocked_area(image)
        corners = truth["corner_mask"]
        # Most of the true corner area is caught...
        assert (blocked & corners).sum() > 0.9 * corners.sum()
        # ...and the illuminated disc is essentially untouched. The margin grows
        # the region slightly inwards, which is deliberate.
        inside = truth["illuminated"]
        assert (blocked & inside).sum() < 0.08 * inside.sum()

    def test_leaves_the_particles_alone(self):
        image, truth = make_vignetted_frame()
        blocked = reg.blocked_area(image)
        assert not (blocked & truth["particle_mask"]).any()

    def test_finds_a_grid_bar(self):
        blocked = reg.blocked_area(make_grid_bar_frame())
        assert 0.1 < blocked.mean() < 0.6, "a grid bar covers a big slice, not everything"

    def test_a_clean_frame_has_nothing_blocked(self):
        image, _ = make_burnin_micrograph()
        assert not reg.blocked_area(image).any()

    def test_a_uniform_frame_has_nothing_blocked(self):
        assert not reg.blocked_area(np.full((300, 400), 120, dtype=np.uint8)).any()

    def test_a_dark_object_floating_in_the_field_is_sample(self):
        # A dark agglomerate that does not reach the edge is specimen, however
        # dark it is. Only edge-connected regions are beam-blocked.
        image = np.full((400, 400), 200, dtype=np.uint8)
        image[150:250, 150:250] = 0
        assert not reg.blocked_area(image).any()

    def test_a_small_dark_edge_touch_is_not_blocked(self):
        # A particle clipped by the frame edge is still a particle.
        image = np.full((400, 400), 200, dtype=np.uint8)
        image[0:12, 0:12] = 0
        assert not reg.blocked_area(image).any()

    def test_a_large_dark_edge_region_is_blocked(self):
        image = np.full((400, 400), 200, dtype=np.uint8)
        image[0:120, 0:200] = 0
        blocked = reg.blocked_area(image)
        assert blocked.any()
        assert blocked[10, 10]
        assert not blocked[350, 350]


class TestParticlesAreNeverExcluded:
    """
    On real SEM filters the background is as dark as a grid bar. The first rule
    filled in every particle sitting on it and grew into their edges, so on
    dozens of the 286 test images real particles were left out of analysis.
    """

    def filter_frame(self):
        rng = np.random.default_rng(3)
        image = rng.normal(6, 3, size=(400, 400)).clip(0, 255)
        particles = np.zeros((400, 400), dtype=bool)
        yy, xx = np.mgrid[0:400, 0:400]
        for cy, cx, r in ((100, 100, 30), (250, 300, 45), (330, 90, 12)):
            particles |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        image[particles] = 220
        return image.astype(np.uint8), particles

    def test_a_bright_particle_on_dark_background_is_kept(self):
        image, particles = self.filter_frame()
        blocked = reg.blocked_area(image)
        assert not (blocked & particles).any()

    def test_the_particle_edges_are_not_eaten_into(self):
        # The margin grows the region into dark pixels only.
        image, particles = self.filter_frame()
        region, _ = reg.analysable_region(image)
        assert region[particles].all()

    def test_a_textured_dark_object_at_the_edge_is_sample(self):
        # A dense TEM aggregate running off the frame: dark, large, at the edge,
        # but textured, which a beam-blocked area never is.
        rng = np.random.default_rng(4)
        image = np.full((400, 400), 200.0)
        image[0:200, 0:220] = rng.normal(25, 18, size=(200, 220))
        assert not reg.blocked_area(image.clip(0, 255).astype(np.uint8)).any()


class TestAnalysableRegion:
    def test_excludes_the_blocked_area(self):
        image, truth = make_vignetted_frame()
        region, info = reg.analysable_region(image)
        assert info["analysable_fraction"] < 1.0
        assert not region[0, 0], "the corner should be out"
        assert region[image.shape[0] // 2, image.shape[1] // 2]

    def test_a_clean_frame_is_entirely_analysable(self):
        image, _ = make_burnin_micrograph()
        region, info = reg.analysable_region(image)
        assert info["analysable_fraction"] == 1.0
        assert region.all()

    def test_excludes_a_named_box(self):
        # The burned-in scale bar: known position, must not be measured.
        image, truth = make_burnin_micrograph()
        x0, y0, w, h = truth["bar_box"]
        region, info = reg.analysable_region(image, exclude_boxes=[(x0, y0, w, h)])
        assert info["excluded_boxes"] == 1
        assert not region[y0 + h // 2, x0 + w // 2]
        assert region[100, 500]

    def test_a_box_at_the_frame_edge_gets_only_its_margin(self):
        # The far edge was computed from the near edge after clamping it to the
        # frame, so a box near the left or top excluded up to a margin's worth
        # of specimen too much on its other side.
        image = np.full((100, 100), 128, dtype=np.uint8)
        region, _ = reg.analysable_region(image, exclude_boxes=[(2, 3, 20, 10)],
                                          margin=6)
        excluded_cols = np.where(~region.all(axis=0))[0]
        excluded_rows = np.where(~region.all(axis=1))[0]
        assert (excluded_cols.min(), excluded_cols.max()) == (0, 2 + 20 + 6 - 1)
        assert (excluded_rows.min(), excluded_rows.max()) == (0, 3 + 10 + 6 - 1)

    def test_a_box_inside_the_frame_gets_its_margin_all_round(self):
        image = np.full((100, 100), 128, dtype=np.uint8)
        region, _ = reg.analysable_region(image, exclude_boxes=[(40, 50, 20, 10)],
                                          margin=6)
        cols = np.where(~region.all(axis=0))[0]
        rows = np.where(~region.all(axis=1))[0]
        assert (cols.min(), cols.max()) == (34, 65)
        assert (rows.min(), rows.max()) == (44, 65)

    def test_ignores_malformed_boxes(self):
        image, _ = make_burnin_micrograph()
        region, info = reg.analysable_region(image, exclude_boxes=[None, (1, 2, 3)])
        assert info["excluded_boxes"] == 0
        assert region.all()

    def test_accepts_a_precomputed_blocked_mask(self):
        image, _ = make_burnin_micrograph()
        blocked = np.zeros(image.shape[:2], dtype=bool)
        blocked[0:50, :] = True
        region, info = reg.analysable_region(image, blocked=blocked)
        # Recorded rounded, for display.
        assert info["blocked_fraction"] == pytest.approx(blocked.mean(), abs=1e-5)
        assert not region[10, 10]

    def test_info_says_what_was_excluded(self):
        image, _ = make_vignetted_frame()
        _region, info = reg.analysable_region(image)
        assert set(info) == {"blocked_fraction", "excluded_boxes", "analysable_fraction"}


class TestUsable:
    def test_a_mostly_clear_frame_is_usable(self):
        assert reg.usable(np.ones((100, 100), dtype=bool))

    def test_a_mostly_blocked_frame_is_not(self):
        region = np.zeros((100, 100), dtype=bool)
        region[:10] = True     # 10% left
        assert not reg.usable(region)

    def test_the_threshold_is_adjustable(self):
        region = np.zeros((100, 100), dtype=bool)
        region[:30] = True
        assert reg.usable(region, minimum=0.2)
        assert not reg.usable(region, minimum=0.5)

    def test_no_region_means_no_restriction(self):
        assert reg.usable(None)


class TestNoDeprecationWarnings:
    def test_the_region_path_is_clean(self):
        import warnings

        image, _ = make_vignetted_frame()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reg.analysable_region(image)
        offenders = [f"{w.category.__name__}: {w.message}" for w in caught
                     if issubclass(w.category, (DeprecationWarning, FutureWarning))]
        assert not offenders, offenders
