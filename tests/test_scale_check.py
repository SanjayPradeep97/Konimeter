"""
Showing what was measured, and interrupting only when it is worth it.

Two things a reported nm/px cannot tell you: which pixels were counted, and
whether anyone has looked. The first is answered by drawing the measured span on
the bar it was measured on; the second decides whether working through a folder
stops on this image or carries straight on.

Both matter because the failure is silent. C1_0030 in the NIOSH set reads its
"1 µm" label as "7 µm" — a plausible number, a seven-fold error in every
measurement of that frame, and obvious the moment the span is drawn with its
value beside it.
"""

import numpy as np
import pytest

from sem_particle_analysis import scale_calibration as sc
from synthetic import make_burnin_micrograph


def box_ocr_calibration(box=(30, 900, 300, 1000), line=(3, 131, 60), **kwargs):
    """A tier 2 calibration as ``from_box_ocr`` builds one."""
    fields = dict(nm_per_px=7.8125, method="box_ocr", scale_nm=1000.0,
                  pixel_length=128.0,
                  extra={"box": list(box) if box else None,
                         "line_coords": list(line) if line else None})
    fields.update(kwargs)
    return sc.ScaleCalibration(**fields)


class TestWhereTheMeasurementWasTaken:
    def test_two_clicked_points_are_the_span(self):
        cal = sc.from_two_points((40, 980), (168, 980), 1, "µm")
        assert sc.measured_span(cal) == ((40.0, 980.0), (168.0, 980.0))

    def test_a_box_reading_is_placed_by_the_box_it_was_read_in(self):
        # line_coords come back relative to the crop the detector was handed,
        # so they are meaningless without the box's own origin — using them raw
        # would draw the span in the top-left corner of the frame.
        cal = box_ocr_calibration(box=(30, 900, 300, 1000), line=(3, 131, 60))
        # Columns 3..131 are the bar's own pixels: 129 px, edge to edge.
        assert sc.measured_span(cal) == ((33.0, 960.0), (162.0, 960.0))

    def test_an_automatic_reading_is_placed_by_its_search_region(self):
        # The automatic sweep records where it looked as (x, y, w, h) rather
        # than as corners.
        cal = box_ocr_calibration(box=None, line=(3, 131, 60),
                                  extra={"region": (30, 900, 270, 100),
                                         "line_coords": (3, 131, 60)})
        # Columns 3..131 are the bar's own pixels: 129 px, edge to edge.
        assert sc.measured_span(cal) == ((33.0, 960.0), (162.0, 960.0))

    def test_metadata_measured_no_bar_so_there_is_no_span(self):
        cal = sc.ScaleCalibration(nm_per_px=2.5, method="metadata")
        assert sc.measured_span(cal) is None
        assert sc.search_box(cal) is None

    def test_a_reading_with_no_recorded_line_has_no_span(self):
        cal = sc.ScaleCalibration(nm_per_px=2.5, method="box_ocr",
                                  extra={"box": [0, 0, 10, 10]})
        assert sc.measured_span(cal) is None


@pytest.fixture(scope="module")
def bar_frame():
    """A TEM-style frame with a known bar, and a calibration measuring it."""
    image, truth = make_burnin_micrograph(bar_left=33, bar_length_px=128,
                                          bar_thickness=12, scale_nm=1000.0)
    x0, y0, width, height = truth["bar_box"]
    cal = sc.from_two_points((x0, y0 + height // 2),
                             (x0 + width, y0 + height // 2), 1, "µm")
    return image, truth, cal


class TestWhatIsLeftOutOfAnalysis:
    """
    A bar printed on the picture is left out of measurement. Leaving out the
    whole region that was searched, a fifth of the frame's width, discarded
    specimen; the reader now records exactly where the bar and its label are.
    """

    def test_the_reader_marks_the_bar_and_label(self):
        cal = sc.ScaleCalibration(nm_per_px=2.0, method="box_ocr", scale_nm=500,
                                  pixel_length=250,
                                  extra={"region": [0, 900, 300, 100],
                                         "mark": [20, 940, 260, 40]})
        assert sc.printed_area(cal, (1024, 1024)) == (20, 940, 260, 40)

    def test_clicked_points_get_a_band_round_them(self):
        cal = sc.from_two_points((100, 950), (300, 950), 1, "µm")
        x, y, w, h = sc.printed_area(cal, (1000, 1000))
        assert x < 100 and x + w > 300
        assert y < 950 < y + h

    def test_without_a_mark_the_searched_region_is_used(self):
        cal = sc.ScaleCalibration(nm_per_px=2.0, method="box_ocr", scale_nm=500,
                                  pixel_length=250, extra={"region": [0, 900, 300, 100]})
        assert sc.printed_area(cal, (1024, 1024)) == (0, 900, 300, 100)

    def test_metadata_prints_nothing(self):
        cal = sc.ScaleCalibration(nm_per_px=2.0, method="metadata")
        assert sc.printed_area(cal, (1024, 1024)) is None


class TestTheCheckImage:
    def test_it_shows_the_span_where_the_bar_actually_is(self, bar_frame):
        from sem_analysis_app.visualization import render_scale_check, _SPAN_COLOR

        image, truth, cal = bar_frame
        view = render_scale_check(image, cal)
        assert view is not None

        # The marks are drawn in one colour that appears nowhere in a greyscale
        # micrograph, so finding them is finding the drawing.
        marked = (np.abs(view.astype(int) - np.array(_SPAN_COLOR)).sum(axis=2) < 90)
        columns = np.where(marked.any(axis=0))[0]
        assert columns.size, "nothing was drawn"

        # Left and right ticks bracket the bar, and the width they bracket is the
        # measured length once the zoom is taken back out.
        x0, _y0, bar_width, _h = truth["bar_box"]
        zoom = (columns.max() - columns.min()) / bar_width
        assert 1.0 <= zoom <= 10.0
        assert abs((columns.max() - columns.min()) / zoom - bar_width) < 3

    def test_it_says_what_the_measurement_was(self, bar_frame):
        # The caption is the other half: a span drawn without its value cannot
        # catch a misread label, which is the failure it exists to catch.
        from sem_analysis_app.visualization import render_scale_check

        image, _truth, cal = bar_frame
        view = render_scale_check(image, cal)
        # The caption sits on a band added below the crop, so the view is taller
        # than the region it came from.
        assert view.shape[0] > 0
        band = view[-30:]
        assert (band > 200).any(), "no light text on the caption band"

    def test_metadata_has_nothing_to_show(self, bar_frame):
        from sem_analysis_app.visualization import render_scale_check

        image, _truth, _cal = bar_frame
        cal = sc.ScaleCalibration(nm_per_px=2.5, method="metadata")
        assert render_scale_check(image, cal) is None
        assert render_scale_check(image, None) is None
        assert render_scale_check(None, cal) is None

    def test_a_bar_against_the_frame_edge_still_renders(self):
        from sem_analysis_app.visualization import render_scale_check

        image, truth = make_burnin_micrograph(bar_left=2, bar_bottom_margin=4,
                                              bar_length_px=100)
        x0, y0, width, height = truth["bar_box"]
        cal = sc.from_two_points((x0, y0 + height // 2),
                                 (x0 + width, y0 + height // 2), 500, "nm")
        assert render_scale_check(image, cal) is not None


@pytest.fixture
def app_state():
    """The app's process-wide state, restored afterwards."""
    pytest.importorskip("gradio")
    from sem_analysis_app.state import state

    kept = (state.scale_calibration, state.scale_baseline, state.scale_info,
            state.current_image, state.image_paths, state.current_index,
            state.scale_baselines, state.label_corrections)
    state.scale_calibration = None
    state.scale_baseline = None
    state.scale_info = None
    state.current_image = None
    state.image_paths = []
    state.current_index = 0
    state.scale_baselines = []
    state.label_corrections = {}
    yield state
    (state.scale_calibration, state.scale_baseline, state.scale_info,
     state.current_image, state.image_paths, state.current_index,
     state.scale_baselines, state.label_corrections) = kept


class TestWhenToInterrupt:
    """
    The loop is only fast if an interruption earns itself.

    Stopping on every OCR reading would be as good as stopping on none: the
    analyst learns to click through. Stopping on the readings nothing vouches
    for, and letting through the ones that match a scale already accepted, asks
    once per magnification.
    """

    def test_no_scale_at_all_stops(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        level, message = scale_tab.review()
        assert level == "stop"
        assert "pixels" in message

    def test_a_flagged_reading_stops(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = box_ocr_calibration(
            warning="Unit unreadable in '2urf)' - assumed micrometres")
        level, message = scale_tab.review()
        assert level == "stop"
        assert "Unit unreadable" in message

    def test_metadata_is_taken_as_read(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = sc.ScaleCalibration(nm_per_px=2.5,
                                                          method="metadata")
        assert scale_tab.review() == (None, "")

    def test_clicking_both_ends_needs_no_second_opinion(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = sc.from_two_points((0, 0), (128, 0), 1, "µm")
        assert scale_tab.review()[0] is None

    def test_the_first_ocr_reading_of_a_run_is_offered_for_a_look(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = box_ocr_calibration()
        level, message = scale_tab.review()
        assert level == "check"
        assert "7.812" in message

    def test_a_reading_matching_one_already_accepted_goes_through(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_baseline = 7.8125
        app_state.scale_calibration = box_ocr_calibration(nm_per_px=7.83)
        assert scale_tab.review()[0] is None

    def test_a_different_magnification_is_stopped_for_once(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_baseline = 7.8125
        app_state.scale_calibration = box_ocr_calibration(nm_per_px=19.6)
        level, message = scale_tab.review()
        assert level == "check"
        assert "different magnification" in message

    def test_confirming_makes_it_the_yardstick_for_the_rest(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = box_ocr_calibration()
        assert scale_tab.review()[0] == "check"

        scale_tab.confirm_scale()
        assert scale_tab.review()[0] is None
        assert app_state.scale_baseline == pytest.approx(7.8125)

        # ...and the next image read the same way is no longer interrupted.
        app_state.scale_calibration = box_ocr_calibration(nm_per_px=7.80)
        assert scale_tab.review()[0] is None


class TestWhatTheInterfaceIsTold:
    def test_a_flagged_reading_sends_the_analyst_to_the_scale_tab(self, app_state):
        import gradio as gr
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = box_ocr_calibration(warning="check this")
        span, preview, row, message, check_img, *fields, tab = scale_tab.check_outputs()
        assert isinstance(tab, gr.Tabs)
        # Nothing to click through on the working tab: this one needs the canvas.
        assert row.get("visible") is False

    def test_an_unvouched_reading_is_offered_in_place(self, app_state):
        import gradio as gr
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = box_ocr_calibration()
        span, preview, row, message, check_img, *fields, tab = scale_tab.check_outputs()
        assert not isinstance(tab, gr.Tabs), "should not move the analyst"
        assert row.get("visible") is True
        assert "7.812" in message

    def test_a_settled_scale_says_nothing(self, app_state):
        import gradio as gr
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = sc.ScaleCalibration(nm_per_px=2.5,
                                                          method="metadata")
        span, preview, row, message, check_img, *fields, tab = scale_tab.check_outputs()
        assert not isinstance(tab, gr.Tabs)
        assert row.get("visible") is False

    def test_a_bar_reading_offers_its_value_to_correct(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = box_ocr_calibration()
        *_, value, unit, work_value, work_unit, _tab = scale_tab.check_outputs()
        assert (value["value"], unit["value"]) == (1.0, "µm")
        assert value["visible"] and work_value["visible"]

    def test_metadata_has_no_value_to_correct(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = sc.ScaleCalibration(nm_per_px=2.5, method="metadata")
        *_, value, _unit, _wv, _wu, _tab = scale_tab.check_outputs()
        assert value["visible"] is False

    def test_the_span_channel_carries_the_measurement(self, app_state, bar_frame):
        import json
        from sem_analysis_app.callbacks import scale_tab

        image, truth, cal = bar_frame
        app_state.current_image = image
        app_state.scale_calibration = cal
        payload = json.loads(scale_tab.scale_span_payload())
        assert payload["span"][0][0] == pytest.approx(truth["bar_box"][0])
        assert "1 µm" in payload["label"]
        assert "128 px" in payload["label"]

    def test_nothing_measured_means_an_empty_channel(self, app_state):
        import json
        from sem_analysis_app.callbacks import scale_tab

        assert json.loads(scale_tab.scale_span_payload()) == {}


class TestCanvasCoordinates:
    """
    A frame too large to send whole arrives shrunk, and everything the canvas
    hands back has to be in the coordinates of the frame that will be measured.
    Without the transport scale a 4096px frame's box was committed at roughly
    two-thirds of where it was drawn, and the bar was read somewhere else.
    """

    def test_a_frame_that_fits_is_sent_unchanged(self, app_state):
        import json
        from sem_analysis_app.callbacks import scale_tab

        payload = json.loads(scale_tab._image_payload(np.zeros((512, 512, 3),
                                                               np.uint8)))
        assert payload["px_scale"] == 1
        assert payload["w"] == 512

    def test_an_oversized_frame_reports_how_much_it_shrank(self, app_state):
        import json
        from sem_analysis_app.callbacks import scale_tab

        width = scale_tab.MAX_CANVAS_PX * 2
        payload = json.loads(scale_tab._image_payload(
            np.zeros((width, width, 3), np.uint8)))
        assert payload["px_scale"] == pytest.approx(payload["w"] / width)
        assert payload["px_scale"] < 1


class TestTheResultsFileSaysHowTheScaleWasSet:
    """
    The provenance has to survive the trip into the CSV, not just exist on the
    calibration. Saving is the last point at which the two can be brought
    together — afterwards there is only a float.
    """

    @pytest.fixture
    def ready_to_save(self, app_state, tmp_path):
        from sem_particle_analysis import ParticleAnalyzer, ResultsManager
        from synthetic import make_disk_mask

        mask, _ = make_disk_mask(shape=(200, 200),
                                 centers_radii=((60, 60, 20), (140, 120, 20)))
        analyzer = ParticleAnalyzer(conversion_factor=2.5, min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)

        app_state.analyzer = analyzer
        app_state.image_paths = [str(tmp_path / "a.tif")]
        app_state.current_index = 0
        app_state.results_manager = ResultsManager(
            csv_file=str(tmp_path / "results.csv"))
        yield app_state, tmp_path / "results.csv"
        app_state.analyzer = None
        app_state.results_manager = None

    def saved_method(self, csv_path):
        import pandas as pd

        return pd.read_csv(csv_path)["scale_method"].iloc[0]

    def test_an_ocr_reading_is_saved_as_unconfirmed(self, ready_to_save):
        from sem_analysis_app.callbacks.results import save_current_results

        state, csv_path = ready_to_save
        state.scale_calibration = sc.ScaleCalibration(nm_per_px=2.5,
                                                      method="box_ocr")
        status, _gallery = save_current_results()
        assert status.startswith("✅"), status
        assert self.saved_method(csv_path) == "box_ocr+unconfirmed"

    def test_confirming_first_changes_what_is_saved(self, ready_to_save):
        from sem_analysis_app.callbacks.results import save_current_results
        from sem_analysis_app.callbacks import scale_tab

        state, csv_path = ready_to_save
        state.scale_calibration = sc.ScaleCalibration(nm_per_px=2.5,
                                                      method="box_ocr")
        scale_tab.confirm_scale()
        save_current_results()
        assert self.saved_method(csv_path) == "box_ocr"

    def test_measuring_in_pixels_says_so_rather_than_saying_nothing(self,
                                                                    ready_to_save):
        # An empty cell means the column did not exist yet; "none" means this
        # image was measured with no scale at all.
        from sem_analysis_app.callbacks.results import save_current_results

        state, csv_path = ready_to_save
        state.scale_calibration = None
        save_current_results()
        assert self.saved_method(csv_path) == "none"


class TestAScaleSetAfterTheCandidateIsTheOneMeasured:
    """
    Measurements were fixed when the candidate was chosen, so a scale read or
    cleared afterwards changed what the row said about its scale but not the
    numbers in it.
    """

    @pytest.fixture
    def analysed(self, app_state, tmp_path):
        from sem_particle_analysis import ParticleAnalyzer, ResultsManager
        from synthetic import make_disk_mask

        mask, _ = make_disk_mask(shape=(200, 200), centers_radii=((60, 60, 20),))
        analyzer = ParticleAnalyzer(conversion_factor=2.5, min_size=30)
        analyzer.analyze_mask(mask, remove_border=False)
        app_state.analyzer = analyzer
        app_state.current_image = np.zeros((200, 200, 3), np.uint8)
        app_state.image_paths = [str(tmp_path / "a.tif")]
        app_state.current_index = 0
        app_state.save_dataset = False
        app_state.results_manager = ResultsManager(csv_file=str(tmp_path / "results.csv"))
        yield app_state, tmp_path / "results.csv"
        app_state.analyzer = None
        app_state.results_manager = None

    def saved(self, csv_path):
        import pandas as pd
        return pd.read_csv(csv_path).iloc[-1]

    def test_a_later_reading_rescales_the_saved_numbers(self, analysed):
        from sem_analysis_app.callbacks import scale_tab
        from sem_analysis_app.callbacks.results import save_current_results

        state, csv_path = analysed
        scale_tab._adopt(sc.ScaleCalibration(nm_per_px=5.0, method="two_points",
                                             confirmed=True))
        save_current_results()
        row = self.saved(csv_path)
        assert row["nm_per_px"] == pytest.approx(5.0)
        assert row["scale_method"] == "two_points"
        area_px = float(row["particle_areas_px"].strip("[]"))
        area_nm2 = float(row["particle_areas_nm2"].strip("[]"))
        assert area_nm2 == pytest.approx(area_px * 25.0)

    def test_clearing_the_scale_saves_pixels(self, analysed):
        from sem_analysis_app.callbacks import scale_tab
        from sem_analysis_app.callbacks.results import save_current_results

        state, csv_path = analysed
        scale_tab.clear_scale()
        save_current_results()
        row = self.saved(csv_path)
        assert row["scale_method"] == "none"
        assert np.isnan(row["nm_per_px"])


class TestCorrectingTheBarsValue:
    """
    OCR reads the bar's number wrong far more often than it mismeasures the bar:
    on the test set, "10 µm" came back as "40 Urn" on four frames. Correcting the
    number keeps the measured bar and replaces only the value.
    """

    def test_a_corrected_value_rescales_with_the_same_bar(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = box_ocr_calibration(scale_nm=40000.0,
                                                          nm_per_px=40000.0 / 128)
        scale_tab.confirm_scale(10, "µm")
        cal = app_state.scale_calibration
        assert cal.nm_per_px == pytest.approx(10000.0 / 128)
        assert cal.pixel_length == 128 and cal.trustworthy
        assert "corrected" in cal.detail

    def test_the_same_misreading_is_corrected_next_time(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = box_ocr_calibration(scale_nm=40000.0,
                                                          nm_per_px=40000.0 / 128)
        scale_tab.confirm_scale(10, "µm")
        assert app_state.known_correction(40000.0, 128) == pytest.approx(10000.0)
        assert app_state.known_correction(40000.0, 200) is None

    def test_setting_it_back_forgets_the_correction(self, app_state):
        # Corrected to 20 µm and back to the 40 µm that was read: nothing to
        # remember, or the next 40 µm bar would be "corrected" to 20.
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = box_ocr_calibration(scale_nm=40000.0,
                                                          nm_per_px=40000.0 / 128)
        scale_tab.confirm_scale(20, "µm")
        scale_tab.confirm_scale(40, "µm")
        assert app_state.known_correction(40000.0, 128) is None
        assert app_state.scale_calibration.nm_per_px == pytest.approx(40000.0 / 128)

    def test_an_unchanged_value_just_confirms(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        app_state.scale_calibration = box_ocr_calibration()
        scale_tab.confirm_scale(1, "µm")
        assert app_state.scale_calibration.confirmed
        assert app_state.label_corrections == {}

    def test_every_accepted_magnification_is_remembered(self, app_state):
        from sem_analysis_app.callbacks import scale_tab

        for nm in (1000.0, 2000.0):
            app_state.scale_calibration = box_ocr_calibration(scale_nm=nm, nm_per_px=nm / 128)
            scale_tab.confirm_scale()
        # A reading at the first magnification again goes straight through.
        app_state.scale_calibration = box_ocr_calibration(scale_nm=1000.0,
                                                          nm_per_px=1000.0 / 128)
        assert scale_tab.review()[0] is None
