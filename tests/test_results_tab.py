"""
The results file and the tabs that show it: saving, deleting, summarising,
moving through a folder.

Each test here is a way the app used to lose or misreport work: deleting the
wrong image's row, a spreadsheet edit undone by the next save, a gallery tile
opening the image after it, a summary leaving out particles without saying so.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("gradio")

from sem_particle_analysis import ParticleAnalyzer, ResultsManager
from sem_analysis_app.state import state
from synthetic import make_disk_mask


def measurements(n, nm=True):
    areas = [100.0 + i for i in range(n)]
    return {"num_particles": n, "areas_px": areas, "diameters_px": [10.0] * n,
            "areas": [a * 4 for a in areas] if nm else [],
            "diameters": [20.0] * n if nm else [],
            "unit": "nm" if nm else "pixels", "nm_per_px": 2.0 if nm else None}


@pytest.fixture(autouse=True)
def fresh_state():
    saved = dict(vars(state))
    yield
    vars(state).clear()
    vars(state).update(saved)


@pytest.fixture
def folder(tmp_path):
    for name in ("A_0001.tif", "B_0002.tif", "C_0003.tif"):
        Image.fromarray(np.full((40, 40), 120, np.uint8)).save(tmp_path / name)
    state.image_paths = [str(tmp_path / n) for n in ("A_0001.tif", "B_0002.tif", "C_0003.tif")]
    state.processed_images = {}
    state.results_manager = ResultsManager(csv_file=str(tmp_path / "analysis_results.csv"))
    return tmp_path


class TestTheResultsFile:
    def test_a_row_deleted_in_a_spreadsheet_stays_deleted(self, folder):
        manager = state.results_manager
        manager.add_result("A_0001.tif", measurements(2))
        manager.add_result("B_0002.tif", measurements(3))
        frame = pd.read_csv(manager.csv_file)
        frame[frame.file_name != "A_0001.tif"].to_csv(manager.csv_file, index=False)

        manager.add_result("C_0003.tif", measurements(1))
        assert pd.read_csv(manager.csv_file).file_name.tolist() == ["B_0002.tif", "C_0003.tif"]

    def test_appending_to_a_file_without_a_final_line_break(self, folder):
        manager = state.results_manager
        manager.add_result("A_0001.tif", measurements(2))
        path = manager.csv_file
        text = open(path).read().rstrip("\n")
        open(path, "w").write(text)            # as a spreadsheet may leave it

        manager.add_result("B_0002.tif", measurements(3))
        rows = pd.read_csv(path)
        assert rows.file_name.tolist() == ["A_0001.tif", "B_0002.tif"]
        assert rows.num_particles.tolist() == [2, 3]


class TestDeletingARow:
    def test_deletes_the_image_named_even_after_the_list_moved(self, folder):
        # Delete used to go by position in a list that a re-save had reordered.
        from sem_analysis_app.callbacks.results import delete_result_row

        manager = state.results_manager
        manager.add_result("A_0001.tif", measurements(2))
        manager.add_result("B_0002.tif", measurements(3))
        manager.add_result("A_0001.tif", measurements(4), replace=True)   # now last
        state.sync_processed_from_csv()

        *_, message, gallery = delete_result_row("A_0001.tif")
        assert message.startswith("✅"), message
        assert pd.read_csv(manager.csv_file).file_name.tolist() == ["B_0002.tif"]
        assert not state.is_processed(0) and state.is_processed(1)
        assert gallery[0][1].startswith("⚪")

    def test_a_name_no_longer_in_the_file_is_refused(self, folder):
        from sem_analysis_app.callbacks.results import delete_result_row

        state.results_manager.add_result("B_0002.tif", measurements(3))
        *_, message, _gallery = delete_result_row("A_0001.tif")
        assert message.startswith("❌") and "Refresh" in message
        assert len(pd.read_csv(state.results_manager.csv_file)) == 1


class TestSummaries:
    def test_frames_without_a_scale_are_counted_not_dropped(self, folder):
        from sem_analysis_app.callbacks.results import get_session_summary
        from sem_analysis_app.callbacks.plots import update_histogram_plots

        state.results_manager.add_result("A_0001.tif", measurements(1))
        state.results_manager.add_result("B_0002.tif", measurements(2, nm=False))

        _table, _progress, stats, choices = get_session_summary()
        assert "From 1 of 3 particles" in stats
        assert choices["choices"] == ["A_0001.tif", "B_0002.tif"]

        _plot, status = update_histogram_plots()
        assert "1 particle from 2 images" in status and "2 more" in status

    @pytest.mark.parametrize("axis", ["Log", "Linear"])
    def test_sizes_plot_on_either_axis(self, folder, axis):
        from sem_analysis_app.callbacks.plots import update_histogram_plots

        state.results_manager.add_result("A_0001.tif", measurements(3))
        plot, status = update_histogram_plots(axis)
        assert status.startswith("✅") and plot.ndim == 3

    def test_a_log_axis_spaces_bins_by_ratio(self):
        # Sizes over three orders of magnitude: on a linear axis every particle
        # but the largest lands in the first bar.
        import matplotlib.pyplot as plt
        from sem_analysis_app.visualization import _histogram

        sizes = np.geomspace(10, 10_000, 200)
        fig, ax = plt.subplots()
        _histogram(ax, sizes, "Diameter (nm)", "coral", log_scale=True)
        heights = [patch.get_height() for patch in ax.patches]
        plt.close(fig)
        assert ax.get_xscale() == "log"
        assert max(heights) < 0.1 * len(sizes)

    def test_a_failed_save_leaves_the_gallery_alone(self, folder):
        # Returning None emptied the gallery, which looks like the work was lost.
        from sem_analysis_app.callbacks.results import save_current_results

        state.analyzer = None
        status, gallery = save_current_results()
        assert status.startswith("❌")
        assert gallery == {"__type__": "update"}


class TestTheGallery:
    def test_an_unreadable_image_keeps_its_place(self, folder):
        # Tiles open by position; skipping one made every later tile open the
        # image after it.
        from sem_analysis_app.callbacks.gallery import create_image_gallery

        (folder / "B_0002.tif").write_bytes(b"not an image")
        tiles = create_image_gallery()
        assert len(tiles) == 3
        assert "B_0002.tif" in tiles[1][1] and "cannot be read" in tiles[1][1]
        assert "C_0003.tif" in tiles[2][1]


class TestMovingThroughAFolder:
    @pytest.fixture
    def analysed(self, folder):
        mask, _ = make_disk_mask(shape=(40, 40), centers_radii=((20, 20, 8),))
        analyzer = ParticleAnalyzer(conversion_factor=None, min_size=10)
        analyzer.analyze_mask(mask, remove_border=False)
        state.analyzer = analyzer
        state.cropped_image = np.full((40, 40, 3), 120, np.uint8)
        state.scale_calibration = None
        state.save_dataset = False
        return folder

    def test_the_end_of_the_folder_says_so(self, analysed):
        from sem_analysis_app.callbacks.workflow import save_and_next

        state.current_index = 2
        outputs = save_and_next(progress=lambda *a, **k: None)
        assert "last image" in outputs[0]

    def test_an_unreadable_next_image_is_not_called_the_end(self, analysed):
        from sem_analysis_app.callbacks.workflow import save_and_next

        (analysed / "B_0002.tif").write_bytes(b"not an image")
        state.current_index = 0
        outputs = save_and_next(progress=lambda *a, **k: None)
        assert "could not be opened" in outputs[0] and "All done" not in outputs[0]
