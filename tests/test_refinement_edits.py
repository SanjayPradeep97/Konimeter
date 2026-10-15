"""
Correcting a mask by clicking, end to end through the Refine callbacks.

Each case here was a way for an analyst's correction to be lost or to change a
number nobody asked to change: a near-miss click that vanished at Apply, a
Redraw that deleted a particle and added nothing back, an Add that measured the
burned-in scale bar, an edit that shifted the areas of untouched particles, and
undo histories that outlived the mask, tool or candidate they described.

SAM is replaced by a stub that returns a fixed mask, so this needs gradio but
no model weights.
"""

import types

import numpy as np
import pytest

from sem_particle_analysis import ParticleAnalyzer
from synthetic import make_disk_mask

pytest.importorskip("gradio")


def no_progress(*args, **kwargs):
    pass


def click(x, y):
    """A click on frame pixel (x, y), as the browser reports it: in the
    coordinates of the working view, which enlarges small frames."""
    from sem_analysis_app.state import state
    from sem_analysis_app.visualization import display_factor

    factor = display_factor(state.cropped_image.shape)
    # Anywhere inside the enlarged pixel lands on the same frame pixel.
    return types.SimpleNamespace(index=(x * factor + factor // 2,
                                        y * factor + factor // 2))


def stub_sam(mask):
    """A segmenter whose every answer is ``mask``."""
    def predict(point_coords, point_labels, box=None, mask_input=None,
                multimask_output=True):
        n = 3 if multimask_output else 1
        return np.stack([mask] * n), np.linspace(0.9, 0.7, n), np.zeros((n, 64, 64))

    return types.SimpleNamespace(
        encode_image=lambda image: None,
        refine_with_sam=lambda *args, **kwargs: (mask, 0.9),
        sam_model=types.SimpleNamespace(predictor=types.SimpleNamespace(predict=predict)),
        selected_mask=None, selected_mask_index=None,
    )


@pytest.fixture
def app():
    """The app state, emptied for the test and put back afterwards."""
    from sem_analysis_app.state import state

    saved = dict(vars(state))
    state.reset_image_state()
    state.min_particle_size = 30
    yield state
    vars(state).clear()
    vars(state).update(saved)


def analyse(state, mask, remove_border=False, min_size=30):
    analyzer = ParticleAnalyzer(conversion_factor=1.0, min_size=min_size)
    analyzer.analyze_mask(mask, remove_border=remove_border, border_buffer=4)
    state.analyzer = analyzer
    state.cropped_image = np.zeros(mask.shape + (3,), dtype=np.uint8)
    state.min_particle_size = min_size
    return analyzer


def areas(analyzer):
    return sorted(int(r.area) for r in analyzer.regions)


@pytest.fixture
def three(app):
    mask, _ = make_disk_mask(shape=(300, 300), centers_radii=(
        (60, 60, 20), (150, 150, 20), (240, 240, 20)))
    return app, analyse(app, mask)


class TestANearMissIsKept:
    """A click up to CLICK_TOLERANCE_PX off a particle selects it — and must
    still name it at Apply, not the background pixel that was clicked."""

    def test_remove(self, three):
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           handle_image_click)
        state, analyzer = three
        state.click_mode = "delete"
        handle_image_click(click(176, 150))           # 6 px right of the middle disk

        status = apply_refinement_changes(progress=no_progress)[2]

        assert "Deleted 1 particle |" in status
        assert not analyzer.mask[150, 150]
        assert len(analyzer.regions) == 2

    def test_merge(self, app):
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           handle_image_click)
        mask, _ = make_disk_mask(shape=(200, 200), centers_radii=((70, 100, 15), (130, 100, 15)))
        analyzer = analyse(app, mask)
        app.click_mode = "merge"
        handle_image_click(click(50, 100))            # 5 px left of the left disk
        handle_image_click(click(150, 100))           # 5 px right of the right disk

        status = apply_refinement_changes(progress=no_progress)[2]

        assert status.startswith("✅ Merged 2 particles"), status
        assert len(analyzer.regions) == 1

    def test_redraw_replaces_rather_than_adds(self, three):
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           handle_image_click)
        state, analyzer = three
        smaller = np.zeros((300, 300), dtype=bool)
        smaller[140:160, 140:160] = True
        state.segmenter = stub_sam(smaller)
        state.click_mode = "point_refine"
        handle_image_click(click(175, 150))           # 5 px outside the middle disk

        apply_refinement_changes(progress=no_progress)

        middle = analyzer.labeled_mask == analyzer.labeled_mask[150, 150]
        assert middle.sum() == analyzer._prepare(smaller, 30, False, 4).sum()
        assert len(analyzer.regions) == 3


class TestARedrawNeverJustDeletes:
    def test_a_preview_apply_would_reject_is_not_offered(self, app):
        # A 2 px wide redraw passed the old size-only check and was then
        # removed entirely by the opening at Apply, after the fibre it was
        # replacing had been deleted.
        from sem_analysis_app.callbacks.refinement import handle_image_click
        mask = np.zeros((200, 200), dtype=bool)
        mask[96:104, 40:160] = True
        analyse(app, mask)
        thin = np.zeros((200, 200), dtype=bool)
        thin[99:101, 40:160] = True
        app.segmenter = stub_sam(thin)
        app.click_mode = "point_refine"

        _viz, status = handle_image_click(click(100, 100))

        assert app.point_refine_preview_mask is None
        assert "No preview" in status

    def test_apply_refuses_rather_than_delete(self, app):
        # However the preview got there, Apply checks it with its own cleanup
        # before deleting anything.
        from sem_analysis_app.callbacks.refinement import apply_refinement_changes
        mask = np.zeros((200, 200), dtype=bool)
        mask[96:104, 40:160] = True
        analyzer = analyse(app, mask)
        before = analyzer.mask.copy()
        thin = np.zeros((200, 200), dtype=bool)
        thin[99:101, 40:160] = True
        label = int(analyzer.labeled_mask[100, 100])
        app.point_refine_particle = label
        app.anchor_selection(label, (100, 100))
        app.point_refine_points, app.point_refine_labels = [(100, 100)], [1]
        app.point_refine_preview_mask = thin

        status = apply_refinement_changes(progress=no_progress)[2]

        assert "Redraw not applied" in status
        assert np.array_equal(analyzer.mask, before)
        assert app.mask_history == []

    def test_the_slider_reaches_the_analyzer(self, app):
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           handle_image_click,
                                                           set_min_particle_size)
        mask, _ = make_disk_mask(shape=(200, 200), centers_radii=((100, 100, 8), (40, 40, 10)))
        analyzer = analyse(app, mask)
        set_min_particle_size(10)
        assert analyzer.min_size == 10

        small = np.zeros((200, 200), dtype=bool)
        small[97:102, 97:102] = True                  # 25 px, 21 after the opening
        app.segmenter = stub_sam(small)
        app.click_mode = "point_refine"
        handle_image_click(click(100, 100))
        apply_refinement_changes(progress=no_progress)

        assert analyzer.mask[100, 100]
        assert len(analyzer.regions) == 2

    def test_raising_the_slider_does_not_refilter_what_is_measured(self, three):
        # The new size applies to what is drawn from now on. Applying it to the
        # whole mask at the next edit would make an unrelated edit delete
        # particles.
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           handle_image_click,
                                                           set_min_particle_size)
        state, analyzer = three
        set_min_particle_size(5000)
        state.click_mode = "delete"
        handle_image_click(click(60, 60))
        apply_refinement_changes(progress=no_progress)
        assert len(analyzer.regions) == 2


class TestAnAdditionIsMeasuredLikeAnyParticle:
    @pytest.fixture
    def framed(self, app):
        from sem_particle_analysis import region as reg
        mask, _ = make_disk_mask(shape=(300, 300), centers_radii=((150, 150, 20), (5, 80, 20)))
        image = np.full((300, 300, 3), 120, dtype=np.uint8)
        app.analysable_region, _ = reg.analysable_region(image, exclude_boxes=[(200, 260, 80, 20)])
        analyzer = analyse(app, mask, remove_border=True)
        return app, analyzer

    def test_what_the_analyst_adds_is_not_cut_to_the_analysable_area(self, framed):
        # The analyst decides what is a particle. The area the app judged
        # analysable once cut the shadowed faces of particles out of every
        # addition, and the outline could not be corrected.
        from sem_analysis_app.callbacks.refinement import handle_image_click
        state, analyzer = framed
        inside_exclusion = np.zeros((300, 300), dtype=bool)
        inside_exclusion[262:278, 205:275] = True
        state.segmenter = stub_sam(inside_exclusion)
        state.click_mode = "add"

        handle_image_click(click(240, 270))

        assert len(state.pending_add_masks) == 1
        assert state.pending_add_masks[0][265:275, 210:270].all()

    def test_an_edge_particle_re_added_measures_the_same(self, framed):
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           handle_image_click)
        state, analyzer = framed
        first = int((analyzer.labeled_mask == analyzer.labeled_mask[80, 10]).sum())
        state.click_mode = "delete"
        handle_image_click(click(10, 80))
        apply_refinement_changes(progress=no_progress)

        yy, xx = np.mgrid[0:300, 0:300]
        state.segmenter = stub_sam((xx - 5) ** 2 + (yy - 80) ** 2 <= 400)
        state.click_mode = "add"
        handle_image_click(click(10, 80))
        apply_refinement_changes(progress=no_progress)

        assert int((analyzer.labeled_mask == analyzer.labeled_mask[80, 10]).sum()) == first


class TestAnEditTouchesOnlyWhatItEdits:
    def test_delete_then_undo(self, app):
        from scipy import ndimage as ndi
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           handle_image_click,
                                                           undo_last_action)
        noise = ndi.gaussian_filter(np.random.default_rng(3).normal(size=(512, 512)), 4)
        analyzer = analyse(app, noise > np.quantile(noise, 0.85), remove_border=True)
        labels = analyzer.labeled_mask.copy()
        before = {tuple(r.coords[0]): int(r.area) for r in analyzer.regions}
        victim = max(analyzer.regions, key=lambda r: r.area)
        row, col = victim.coords[len(victim.coords) // 2]

        app.click_mode = "delete"
        handle_image_click(click(int(col), int(row)))
        apply_refinement_changes(progress=no_progress)

        after = {tuple(r.coords[0]): int(r.area) for r in analyzer.regions}
        del before[tuple(victim.coords[0])]
        assert after == before

        undo_last_action()
        assert np.array_equal(analyzer.labeled_mask, labels)


class TestUndoHistoryDoesNotOutliveItsContext:
    def test_switching_tool_abandons_a_redraw_for_good(self, three):
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           handle_image_click,
                                                           set_click_mode,
                                                           undo_last_action)
        state, analyzer = three
        before = areas(analyzer)
        smaller = np.zeros((300, 300), dtype=bool)
        smaller[140:160, 140:160] = True
        state.segmenter = stub_sam(smaller)
        set_click_mode("point_refine")
        handle_image_click(click(150, 150))
        handle_image_click(click(152, 152))

        set_click_mode("delete")
        undo_last_action()

        assert state.point_refine_preview_mask is None
        apply_refinement_changes(progress=no_progress)
        assert areas(analyzer) == before

    def test_undo_after_dropping_edge_particles_restores_them(self, app):
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           clear_edge_particles,
                                                           handle_image_click,
                                                           undo_last_action)
        mask, _ = make_disk_mask(shape=(300, 300), centers_radii=(
            (150, 3, 12), (60, 80, 20), (150, 150, 20), (240, 240, 20)))
        analyzer = analyse(app, mask)
        app.click_mode = "delete"
        handle_image_click(click(150, 150))
        handle_image_click(click(240, 240))
        clear_edge_particles(10)

        undo_last_action()

        assert len(analyzer.regions) == 4
        # The queued clicks follow their particles through both renumberings.
        assert sorted(app.pending_deletes) == sorted(
            [int(analyzer.labeled_mask[150, 150]), int(analyzer.labeled_mask[240, 240])])
        apply_refinement_changes(progress=no_progress)
        assert not analyzer.mask[150, 150] and not analyzer.mask[240, 240]
        assert analyzer.mask[80, 60]

    def test_choosing_another_candidate_starts_afresh(self, app):
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           handle_image_click)
        from sem_analysis_app.callbacks.segmentation import select_mask_and_analyze
        first, _ = make_disk_mask(shape=(300, 300), centers_radii=(
            (60, 60, 20), (150, 150, 20), (240, 240, 20)))
        second, _ = make_disk_mask(shape=(300, 300), centers_radii=(
            (60, 60, 22), (150, 150, 22), (240, 240, 22), (60, 240, 22)))
        app.cropped_image = np.zeros((300, 300, 3), dtype=np.uint8)
        app.segmenter = stub_sam(np.zeros((300, 300), dtype=bool))
        app.candidates = [{"mask": m.astype(bool), "mask_index": i}
                          for i, m in enumerate((first, second))]
        app.masks = [c["mask"] for c in app.candidates]
        app.scale_info = {"conversion": 2.0}

        select_mask_and_analyze("Option 1", progress=no_progress)
        app.click_mode = "delete"
        handle_image_click(click(60, 60))
        apply_refinement_changes(progress=no_progress)
        handle_image_click(click(150, 150))

        select_mask_and_analyze("Option 2", progress=no_progress)

        assert app.pending_deletes == [] and app.pending_anchors == {}
        assert app.undo_history == [] and app.mask_history == []
        assert len(app.analyzer.regions) == 4


class TestUndoUpdatesTheResults:
    def test_every_path_returns_five_outputs(self, three):
        from sem_analysis_app.callbacks.refinement import (apply_refinement_changes,
                                                           handle_image_click,
                                                           undo_last_action)
        state, analyzer = three
        assert len(undo_last_action()) == 5                       # nothing to undo

        state.click_mode = "delete"
        handle_image_click(click(60, 60))
        out = undo_last_action()                                  # a queued click
        assert len(out) == 5 and "0 undo steps remaining" in out[2]

        handle_image_click(click(60, 60))
        apply_refinement_changes(progress=no_progress)
        _viz, refine_results, status, results, stats = undo_last_action()   # an Apply
        assert len(results) == 3 and len(refine_results) == 3
        assert list(stats["Metric"]) == ["Mean", "Median", "Std Dev", "Min", "Max"]
        assert "0 further steps" in status

        state.analyzer = None
        assert len(undo_last_action()) == 5


class TestTablesSayTheirUnit:
    def test_pixels_are_not_called_nanometres(self):
        from sem_analysis_app.visualization import (create_results_dataframe,
                                                    create_summary_statistics_table)
        mask, _ = make_disk_mask(shape=(200, 200), centers_radii=((100, 100, 20),))
        analyzer = ParticleAnalyzer(conversion_factor=None)
        analyzer.analyze_mask(mask, remove_border=False)
        measurements = analyzer.get_measurements(in_nm=True)

        assert list(create_results_dataframe(measurements).columns) == [
            "Particle ID", "Area (px²)", "Diameter (px)", "Aspect ratio"]
        assert list(create_summary_statistics_table(measurements).columns) == [
            "Metric", "Area (px²)", "Diameter (px)", "Aspect ratio"]

    def test_nanometres_when_there_is_a_scale(self):
        from sem_analysis_app.visualization import create_results_dataframe
        mask, _ = make_disk_mask(shape=(200, 200), centers_radii=((100, 100, 20),))
        analyzer = ParticleAnalyzer(conversion_factor=2.0)
        analyzer.analyze_mask(mask, remove_border=False)
        assert "Area (nm²)" in create_results_dataframe(analyzer.get_measurements()).columns

    def test_an_empty_table_still_says_its_unit(self):
        from sem_analysis_app.visualization import create_results_dataframe
        analyzer = ParticleAnalyzer(conversion_factor=None)
        analyzer.analyze_mask(np.zeros((50, 50), dtype=bool), remove_border=False)
        assert "Diameter (px)" in create_results_dataframe(analyzer.get_measurements()).columns


class TestTheWorkingView:
    def test_a_small_frame_is_enlarged_by_a_whole_number(self):
        from sem_analysis_app.visualization import display_factor, for_display

        frame = np.zeros((512, 512, 3), np.uint8)
        assert display_factor(frame.shape) == 2
        assert for_display(frame).shape == (1024, 1024, 3)

    def test_a_large_frame_is_left_alone(self):
        from sem_analysis_app.visualization import display_factor, for_display

        frame = np.zeros((2048, 2048, 3), np.uint8)
        assert display_factor(frame.shape) == 1
        assert for_display(frame) is frame

    def test_particle_numbers_grow_with_the_frame(self):
        # Shown at a third of its size, a 2048px frame's numbers were a few
        # screen pixels tall.
        from sem_analysis_app.visualization import _drawing_sizes

        assert _drawing_sizes((2048, 2048))["font_scale"] >= 2 * _drawing_sizes((1024, 1024))["font_scale"] * 0.9
        assert _drawing_sizes((512, 512))["font_scale"] >= 0.6
