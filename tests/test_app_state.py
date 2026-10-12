"""
Application state: the undo history that makes editing recoverable.

Applying an edit used to be a point of no return — the click-level undo stack
was cleared on Apply, and nothing recorded the mask beforehand. Since the
analyst applies constantly to see the result, one wrong Apply meant
re-segmenting the image from scratch, part way through a folder of hundreds.

Imports the state module directly rather than through the app, so this runs
without Gradio.
"""

import numpy as np
import pytest

from sem_analysis_app.state import AppState
from sem_particle_analysis import ParticleAnalyzer
from sem_particle_analysis import scale_calibration as sc
from synthetic import make_disk_mask


@pytest.fixture
def state_with_particles():
    mask, _ = make_disk_mask(shape=(300, 300), centers_radii=(
        (70, 70, 22), (150, 90, 22), (220, 200, 22)))
    analyzer = ParticleAnalyzer(conversion_factor=2.0, min_size=30)
    analyzer.analyze_mask(mask, remove_border=False)

    state = AppState()
    state.analyzer = analyzer
    return state


class TestMaskSnapshots:
    def test_a_snapshot_restores_a_deleted_particle(self, state_with_particles):
        state = state_with_particles
        before = len(state.analyzer.regions)

        state.snapshot_mask()
        state.analyzer.delete_particles([state.analyzer.regions[0].label])
        assert len(state.analyzer.regions) == before - 1

        assert state.restore_mask() is True
        assert len(state.analyzer.regions) == before

    def test_snapshots_stack(self, state_with_particles):
        state = state_with_particles
        counts = [len(state.analyzer.regions)]
        for _ in range(2):
            state.snapshot_mask()
            state.analyzer.delete_particles([state.analyzer.regions[0].label])
            counts.append(len(state.analyzer.regions))

        assert state.restore_mask() is True
        assert len(state.analyzer.regions) == counts[1]
        assert state.restore_mask() is True
        assert len(state.analyzer.regions) == counts[0]

    def test_restoring_with_no_snapshot_reports_so(self, state_with_particles):
        assert state_with_particles.restore_mask() is False

    def test_restoring_without_an_analyzer_is_harmless(self):
        assert AppState().restore_mask() is False

    def test_snapshotting_without_an_analyzer_is_harmless(self):
        state = AppState()
        state.snapshot_mask()
        assert state.mask_history == []

    def test_the_stack_is_bounded(self, state_with_particles):
        # Each snapshot is a full-frame boolean mask; an unbounded stack would
        # grow without limit across a long editing session.
        state = state_with_particles
        for _ in range(state.MAX_MASK_HISTORY + 8):
            state.snapshot_mask()
        assert len(state.mask_history) == state.MAX_MASK_HISTORY

    def test_the_oldest_snapshot_is_the_one_dropped(self, state_with_particles):
        state = state_with_particles
        state.MAX_MASK_HISTORY  # documented cap
        first = state.analyzer.mask.copy()
        state.snapshot_mask()
        for _ in range(state.MAX_MASK_HISTORY):
            state.analyzer.delete_particles([state.analyzer.regions[0].label]) \
                if state.analyzer.regions else None
            state.snapshot_mask()
        assert not np.array_equal(state.mask_history[0], first)

    def test_a_snapshot_is_a_copy_not_a_reference(self, state_with_particles):
        # Holding a reference would make the snapshot follow later edits and
        # restore nothing.
        state = state_with_particles
        state.snapshot_mask()
        state.analyzer.mask[:] = False
        state.analyzer._relabel()
        assert len(state.analyzer.regions) == 0
        assert state.restore_mask() is True
        assert len(state.analyzer.regions) > 0


    def test_undo_brings_back_exactly_what_was_there(self, state_with_particles):
        # Restoring used to clean the snapshot again, so Undo could come back
        # with areas a little different from the ones it was undoing to.
        state = state_with_particles
        labels = state.analyzer.labeled_mask.copy()
        areas = [r.area for r in state.analyzer.regions]

        state.snapshot_mask()
        state.analyzer.delete_particles([state.analyzer.regions[1].label])
        state.restore_mask()

        assert np.array_equal(state.analyzer.labeled_mask, labels)
        assert [r.area for r in state.analyzer.regions] == areas


class TestSelections:
    def test_a_near_miss_is_anchored_inside_the_particle(self, state_with_particles):
        # A click may land a few pixels off the particle it selects. Anchored to
        # the background pixel clicked, the selection resolved to nothing at
        # Apply and silently vanished.
        state = state_with_particles
        labels = state.analyzer.labeled_mask
        label = int(labels[70, 70])
        cols = np.where(labels[70] == label)[0]
        click = (int(cols.max()) + 5, 70)
        assert labels[click[1], click[0]] == 0

        state.anchor_selection(label, click, labels)

        x, y = state.pending_anchors[label]
        assert labels[y, x] == label
        assert (x, y) == (int(cols.max()), 70)
        assert state.resolve_label(label, labels) == label

    def test_a_direct_hit_is_anchored_where_it_was_clicked(self, state_with_particles):
        state = state_with_particles
        labels = state.analyzer.labeled_mask
        state.anchor_selection(int(labels[70, 70]), (70, 70), labels)
        assert state.pending_anchors[int(labels[70, 70])] == (70, 70)

    def test_a_relabel_drops_the_click_history(self, state_with_particles):
        # Undo into a history that still names the old labels outlined and
        # edited whichever particles had inherited them.
        state = state_with_particles
        state.save_pending_state()
        state.reindex_pending(state.analyzer.labeled_mask)
        assert state.undo_history == []

    def test_clearing_pending_edits_keeps_applied_ones(self, state_with_particles):
        state = state_with_particles
        state.snapshot_mask()
        state.pending_deletes = [1]
        state.pending_anchors = {1: (70, 70)}
        state.point_refine_points = [(1, 1)]
        state.save_pending_state()

        state.clear_pending_edits()

        assert (state.pending_deletes, state.pending_anchors,
                state.point_refine_points, state.undo_history) == ([], {}, [], [])
        assert len(state.mask_history) == 1


class TestResetClearsHistory:
    def test_opening_a_new_image_forgets_the_old_one(self, state_with_particles):
        state = state_with_particles
        state.snapshot_mask()
        assert state.mask_history

        state.reset_image_state()
        assert state.mask_history == []
        assert state.undo_history == []

    def test_reset_clears_the_frame_geometry_too(self, state_with_particles):
        state = state_with_particles
        state.modality = object()
        state.analysable_region = np.ones((10, 10), dtype=bool)
        state.reset_image_state()
        assert state.modality is None
        assert state.analysable_region is None


class TestScaleDoesNotOutliveItsImage:
    """
    A calibration belongs to one image.

    scale_calibration was never declared on AppState and never cleared by
    reset_image_state — it was attached on first use and read back through
    getattr. So opening a new image cleared scale_info but left the previous
    image's calibration in place, and the frame header and panel, which read the
    calibration, went on reporting the old nm/px.
    """

    def test_it_is_declared_not_conjured(self):
        # Present from the start, so clearing it is possible at all.
        assert AppState().scale_calibration is None

    def test_opening_a_new_image_clears_it(self):
        state = AppState()
        state.scale_calibration = sc.ScaleCalibration(nm_per_px=2.5, method="metadata")
        state.scale_info = state.scale_calibration.to_dict()

        state.reset_image_state()

        assert state.scale_calibration is None
        assert state.scale_info is None

    def test_scale_info_and_calibration_are_cleared_together(self):
        # They disagreeing is what made the stale reading invisible: scale_info
        # said "no scale", the calibration said 2.5 nm/px, and the header read
        # the calibration.
        state = AppState()
        state.scale_calibration = sc.ScaleCalibration(nm_per_px=2.5, method="metadata")
        state.scale_info = state.scale_calibration.to_dict()
        state.reset_image_state()
        assert (state.scale_calibration is None) == (state.scale_info is None)


class TestScaleIsRememberedPerImage:
    @pytest.fixture
    def state(self):
        state = AppState()
        state.image_paths = [r"C:\images\a.tif", r"C:\images\b.tif"]
        state.current_index = 0
        return state

    def test_a_calibration_comes_back_for_the_same_image(self, state):
        cal = sc.from_two_points((0, 0), (200, 0), 500, "nm")
        state.remember_scale(cal)
        state.reset_image_state()
        assert state.recall_scale() is cal

    def test_another_image_does_not_inherit_it(self, state):
        state.remember_scale(sc.from_two_points((0, 0), (200, 0), 500, "nm"))
        state.current_index = 1
        assert state.recall_scale() is None

    def test_each_image_keeps_its_own(self, state):
        first = sc.from_two_points((0, 0), (200, 0), 500, "nm")
        state.remember_scale(first)
        state.current_index = 1
        second = sc.from_two_points((0, 0), (100, 0), 500, "nm")
        state.remember_scale(second)

        state.current_index = 0
        assert state.recall_scale() is first
        state.current_index = 1
        assert state.recall_scale() is second
        assert first.nm_per_px != second.nm_per_px

    def test_clearing_forgets_it_so_the_next_visit_detects_afresh(self, state):
        state.remember_scale(sc.from_two_points((0, 0), (200, 0), 500, "nm"))
        state.forget_scale()
        assert state.recall_scale() is None

    def test_confirming_is_reflected_in_what_was_stored(self, state):
        # confirm_scale mutates the calibration in place; the stored one is the
        # same object, so it must see the change rather than a stale copy.
        cal = sc.ScaleCalibration(nm_per_px=2.5, method="box_ocr")
        state.remember_scale(cal)
        assert state.recall_scale().trustworthy is False
        cal.confirmed = True
        assert state.recall_scale().trustworthy is True

    def test_remembering_without_an_image_is_harmless(self):
        state = AppState()
        state.remember_scale(sc.ScaleCalibration(nm_per_px=2.5, method="metadata"))
        assert state.scale_by_image == {}
        assert state.recall_scale() is None

    def test_an_index_off_the_end_recalls_nothing(self, state):
        state.current_index = 99
        assert state.current_path() is None
        assert state.recall_scale() is None

    def test_the_memory_survives_moving_between_images(self, state):
        cal = sc.from_two_points((0, 0), (200, 0), 500, "nm")
        state.remember_scale(cal)
        for index in (1, 0, 1, 0):
            state.current_index = index
            state.reset_image_state()
        assert state.recall_scale() is cal


class TestTheAcceptedScaleOutlivesTheImage:
    """
    Unlike a calibration, the yardstick belongs to the run.

    It exists to answer "have I already agreed to a scale like this one?", which
    is a question about the folder being worked through. Clearing it per image
    would make every reading look like the first one and stop on all of them.
    """

    def test_accepting_records_the_number(self):
        state = AppState()
        state.accept_scale(sc.ScaleCalibration(nm_per_px=7.8125, method="box_ocr"))
        assert state.scale_baseline == 7.8125

    def test_opening_the_next_image_keeps_it(self):
        state = AppState()
        state.accept_scale(sc.ScaleCalibration(nm_per_px=7.8125, method="box_ocr"))
        state.reset_image_state()
        assert state.scale_baseline == 7.8125
        assert state.scale_calibration is None

    def test_a_later_acceptance_replaces_it(self):
        # A change of magnification part way through a folder is asked about
        # once, and then the new value is what the rest are judged against.
        state = AppState()
        state.accept_scale(sc.ScaleCalibration(nm_per_px=7.8125, method="box_ocr"))
        state.accept_scale(sc.ScaleCalibration(nm_per_px=19.6, method="box_ocr"))
        assert state.scale_baseline == 19.6

    def test_accepting_nothing_is_harmless(self):
        state = AppState()
        state.accept_scale(None)
        assert state.scale_baseline is None
