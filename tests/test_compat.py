"""
scikit-image compatibility shims.

These pin the exact size-filter boundary. scikit-image 0.26 changed
``remove_small_objects`` from an exclusive to an inclusive threshold, which moves
the boundary by one area unit — silently, and in the direction of discarding
particles the documented contract says to keep.
"""

import warnings

import numpy as np
import pytest
from skimage import morphology

from sem_particle_analysis import _compat
from sem_particle_analysis import ParticleAnalyzer


def component_of_area(area, shape=(60, 60)):
    """A single connected component of exactly ``area`` pixels."""
    mask = np.zeros(shape, dtype=bool)
    flat = mask.reshape(-1)
    flat[:area] = True
    return flat.reshape(shape)


class TestRemoveObjectsSmallerThan:
    def test_keeps_a_component_of_exactly_min_size(self):
        # The documented contract: a smallest particle of N ignores those *below* N.
        mask = component_of_area(30)
        assert _compat.remove_objects_smaller_than(mask, 30).sum() == 30

    def test_drops_a_component_one_pixel_below_min_size(self):
        mask = component_of_area(29)
        assert not _compat.remove_objects_smaller_than(mask, 30).any()

    def test_keeps_a_component_above_min_size(self):
        mask = component_of_area(31)
        assert _compat.remove_objects_smaller_than(mask, 31).sum() == 31

    @pytest.mark.parametrize("min_size", [0, 1])
    def test_no_op_for_trivial_thresholds(self, min_size):
        mask = component_of_area(1)
        assert np.array_equal(_compat.remove_objects_smaller_than(mask, min_size), mask)

    def test_emits_no_deprecation_warning(self):
        mask = component_of_area(30)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _compat.remove_objects_smaller_than(mask, 30)
        assert not [w for w in caught
                    if issubclass(w.category, (DeprecationWarning, FutureWarning))]


class TestMorphologyShims:
    @pytest.mark.parametrize("shim", [_compat.binary_opening, _compat.binary_closing])
    def test_returns_boolean(self, shim):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        assert shim(mask, morphology.disk(1)).dtype == bool

    def test_opening_removes_an_isolated_speck(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        mask[2, 2] = True
        opened = _compat.binary_opening(mask, morphology.disk(1))
        assert not opened[2, 2]
        assert opened[20, 20]

    def test_closing_bridges_a_one_pixel_gap(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:20] = True
        mask[10:30, 21:30] = True
        closed = _compat.binary_closing(mask, morphology.disk(1))
        assert closed[20, 20]

    @pytest.mark.parametrize("shim", [_compat.binary_opening, _compat.binary_closing])
    def test_emits_no_deprecation_warning(self, shim):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shim(mask, morphology.disk(1))
        assert not [w for w in caught
                    if issubclass(w.category, (DeprecationWarning, FutureWarning))]


class TestAnalyzerAgreesWithItsOwnSizeFilter:
    def test_morphology_and_region_filter_use_the_same_boundary(self):
        # analyze_mask filters regions with `area >= min_size`. If the cleanup
        # step drops area == min_size first, the two disagree and the documented
        # boundary is off by one.
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:56, 50:60] = True   # 60 px, survives the radius-1 opening

        analyzer = ParticleAnalyzer(min_size=1)
        analyzer.analyze_mask(mask, remove_border=False)
        area = int(analyzer.regions[0].area)

        at_boundary = ParticleAnalyzer(min_size=area)
        count, _ = at_boundary.analyze_mask(mask, min_size=area, remove_border=False)
        assert count == 1, f"a particle of area {area} was dropped at min_size={area}"

        above = ParticleAnalyzer(min_size=area + 1)
        assert above.analyze_mask(mask, min_size=area + 1, remove_border=False)[0] == 0

    def test_whole_measurement_path_is_deprecation_free(self):
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:80, 50:80] = True
        mask[120:150, 120:150] = True

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analyzer = ParticleAnalyzer(conversion_factor=2.5, min_size=30)
            analyzer.analyze_mask(mask, remove_border=True)
            analyzer.merge_particles([r.label for r in analyzer.regions])
            analyzer.get_measurements(in_nm=True)

        offenders = [f"{w.category.__name__}: {w.message}" for w in caught
                     if issubclass(w.category, (DeprecationWarning, FutureWarning))]
        assert not offenders, offenders
