"""
The Segment & Refine tab: SAM's candidates offered as thumbnails, the best put
to use straight away unless that would throw corrections away, and the bar and
measurements that say where the analyst is.
"""
import numpy as np
import pytest

pytest.importorskip("gradio")

from sem_analysis_app.callbacks import scale_tab, segmentation  # noqa: E402
from sem_analysis_app.state import state  # noqa: E402
from synthetic import make_disk_mask  # noqa: E402


def no_progress(*args, **kwargs):
    pass


@pytest.fixture(autouse=True)
def fresh_state():
    saved = dict(vars(state))
    state.reset_image_state()
    yield
    vars(state).clear()
    vars(state).update(saved)


class FakeSegmenter:
    """Three candidates: two discs, then the background."""
    def __init__(self, shape=(200, 200)):
        disks, _ = make_disk_mask(shape=shape, centers_radii=((60, 60, 20), (140, 130, 25)))
        one, _ = make_disk_mask(shape=shape, centers_radii=((60, 60, 20),))
        self.masks = [disks, one, ~disks]
        self.selected_mask = None
        self.selected_mask_index = None

    def rank_candidates(self, image, masks, top_k=3, dark_features=None, exclude=None):
        return [{"mask": m, "mask_index": i} for i, m in enumerate(masks[:top_k])]


@pytest.fixture
def frame():
    image = np.full((200, 200, 3), 200, np.uint8)
    state.current_image = state.cropped_image = image
    state.image_paths = ["/data/A_0001.tif", "/data/B_0002.tif"]
    state.current_index = 0
    state.segmenter = FakeSegmenter()
    state.scale_info = {"conversion": 2.0}
    return state


class TestTheCandidates:
    def test_they_are_offered_as_thumbnails_best_first(self, frame):
        gallery, status = segmentation.rerank()
        captions = [caption for _image, caption in gallery["value"]]
        assert captions == ["Best", "Option 2", "Option 3"]
        assert gallery["selected_index"] == 0 and status == ""

    def test_the_best_is_put_to_use_straight_away(self, frame):
        segmentation.rerank()
        viz, _table, status, *_ = segmentation.use_best_candidate(progress=no_progress)
        assert frame.analyzer is not None and len(frame.analyzer.regions) == 2
        assert "2 particles from the best candidate" in status

    def test_clicking_a_thumbnail_uses_that_one(self, frame):
        class Clicked:
            index = 1
        segmentation.rerank()
        segmentation.choose_candidate(Clicked(), progress=no_progress)
        assert frame.selected_mask_index == 1 and len(frame.analyzer.regions) == 1

    def test_corrections_are_not_thrown_away_by_a_re_sort(self, frame):
        segmentation.rerank()
        segmentation.use_best_candidate(progress=no_progress)
        frame.pending_deletes = [1]                  # a click, queued
        gallery, status = segmentation.rerank()
        assert gallery["selected_index"] is None and "corrections are kept" in status
        assert segmentation.use_best_candidate(progress=no_progress) == (
            {"__type__": "update"},) * 5
        assert frame.pending_deletes == [1]

    def test_nothing_to_offer_says_what_to_do(self, frame):
        frame.segmenter.rank_candidates = lambda *a, **k: []
        gallery, status = segmentation.rerank()
        assert gallery["value"] == [] and "Choose Add" in status

    def test_nothing_to_offer_starts_an_empty_outline_to_add_to(self, frame):
        # Small particles on a porous filter: SAM's whole-frame masks separate
        # nothing. The Add tool then had no analysis to add to and refused
        # every click, so the image could not be measured at all.
        frame.segmenter.rank_candidates = lambda *a, **k: []
        segmentation.rerank()
        assert frame.analyzer is not None and len(frame.analyzer.regions) == 0

    def test_opening_such_an_image_has_the_add_tool_ready(self, frame):
        from sem_analysis_app.callbacks import workflow

        segmenter = frame.segmenter
        segmenter.segment_image = lambda image, multimask_output=True: (segmenter.masks, None)
        segmenter.rank_candidates = lambda *a, **k: []
        workflow.auto_process_current_image(progress=no_progress)
        assert frame.click_mode == "add"

    def test_clear_all_with_nothing_in_use_gives_an_empty_outline(self, frame):
        from sem_analysis_app.callbacks import refinement

        _viz, _table, status, *_ = refinement.clear_all_particles()
        assert not status.startswith("❌") and frame.analyzer is not None


class TestWhereTheAnalystIs:
    def test_the_bar_names_the_image_and_its_scale(self, frame):
        from sem_particle_analysis import scale_calibration as sc

        frame.scale_calibration = sc.ScaleCalibration(nm_per_px=2.0, method="box_ocr")
        frame.microscope = "TEM"
        bar = scale_tab.frame_header()
        assert "A_0001.tif" in bar and "1 of 2" in bar
        assert "2 nm/px · unconfirmed" in bar and "TEM" in bar

    def test_no_scale_and_no_microscope_are_flagged(self, frame):
        frame.microscope = None
        bar = scale_tab.frame_header()
        assert "No scale: sizes in pixels" in bar and "Microscope not set" in bar

    def test_measurements_in_a_few_numbers(self, frame):
        segmentation.rerank()
        segmentation.use_best_candidate(progress=no_progress)
        _bar, summary = scale_tab.work_status()
        assert "Particles" in summary and ">2<" in summary
        assert "Median diameter" in summary and "nm" in summary

    def test_large_sizes_are_written_out(self, frame):
        frame.scale_info = {"conversion": 200.0}
        segmentation.rerank()
        segmentation.use_best_candidate(progress=no_progress)
        summary = scale_tab.measurement_summary()
        assert "e+" not in summary and "," in summary

    def test_before_anything_is_measured(self):
        assert "Nothing measured yet" in scale_tab.measurement_summary()


class TestTheScaleCheckOnTheWorkingTab:
    def test_both_tabs_get_the_bars_value(self):
        # One update shared by two fields left the second at 0: Gradio takes
        # the value out of an update as it applies it.
        from sem_particle_analysis import scale_calibration as sc

        state.scale_calibration = sc.ScaleCalibration(
            nm_per_px=1000.0 / 128, method="box_ocr", scale_nm=1000.0, pixel_length=128.0)
        *_, value, unit, _row, work_value, work_unit, _tab = scale_tab.check_outputs()
        assert value is not work_value and unit is not work_unit
        assert work_value["value"] == value["value"] == 1.0
