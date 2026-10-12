"""
Opening a folder on the Setup tab: which images it finds, and where results go.

Images used to arrive through a browser upload, which hands the app a copy in
Gradio's temporary cache. The results file was then written beside that copy -
somewhere the operating system clears on its own schedule - while the Setup tab
said results were saved next to the images. These tests pin down that the
results file is where the analyst was told it would be.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("gradio")

from sem_analysis_app.callbacks import setup as setup_tab
from sem_analysis_app.state import state
from sem_particle_analysis.utils import find_images_in_folder


def no_progress(*args, **kwargs):
    pass


def write_png(path):
    Image.fromarray(np.full((16, 16), 128, dtype=np.uint8)).save(path, format="PNG")
    return path


@pytest.fixture(autouse=True)
def fresh_state():
    saved = dict(vars(state))
    state.image_paths, state.results_manager, state.processed_images = [], None, {}
    yield
    vars(state).clear()
    vars(state).update(saved)


@pytest.fixture
def sample_folder(tmp_path):
    folder = tmp_path / "Sample A"
    folder.mkdir()
    for name in ("B_0002.TIF", "A_0001.tif", "C_0003.png"):
        write_png(folder / name)
    write_png(folder / "._A_0001.tif")  # macOS metadata file on an exFAT drive
    (folder / "notes.txt").write_text("not an image")
    (folder / "nested").mkdir()
    write_png(folder / "nested" / "D_0004.png")
    return folder


def load(folder):
    status, gallery, _tabs = setup_tab.load_images_from_folder(str(folder),
                                                               progress=no_progress)
    return status, gallery


class TestFindingImages:
    def test_extensions_match_whatever_their_case(self, sample_folder):
        names = [os.path.basename(p) for p in find_images_in_folder(sample_folder)]
        assert names == ["A_0001.tif", "B_0002.TIF", "C_0003.png"]

    def test_hidden_files_subfolders_and_other_files_are_left_out(self, sample_folder):
        names = {os.path.basename(p) for p in find_images_in_folder(sample_folder)}
        assert "._A_0001.tif" not in names
        assert "D_0004.png" not in names
        assert "notes.txt" not in names


class TestWhereResultsGo:
    def test_results_are_saved_in_the_image_folder(self, sample_folder):
        status, gallery = load(sample_folder)

        assert status.startswith("✅"), status
        assert len(gallery) == 3
        expected = sample_folder / "analysis_results.csv"
        assert expected.exists()
        assert os.path.samefile(state.results_manager.csv_file, expected)

    def test_opening_a_folder_goes_to_the_gallery(self, sample_folder):
        _status, _gallery, tabs = setup_tab.load_images_from_folder(
            str(sample_folder), progress=no_progress)
        assert tabs.selected == 1

    def test_a_pasted_path_may_be_quoted(self, sample_folder):
        # Explorer's "Copy as path" puts double quotes around it.
        status, _ = load(f'  "{sample_folder}"  ')
        assert status.startswith("✅"), status
        assert len(state.image_paths) == 3


class TestOpeningAnotherFolder:
    def test_the_previous_image_is_not_carried_over(self, sample_folder, tmp_path):
        # Saving after opening a second folder used to file the image still on
        # screen from the first folder under the second folder's first name.
        state.current_image = np.zeros((4, 4, 3), np.uint8)
        state.analyzer = object()
        state.scale_calibration = object()
        load(sample_folder)
        assert state.current_image is None
        assert state.analyzer is None
        assert state.scale_calibration is None

        from sem_analysis_app.callbacks.results import save_current_results
        status, _ = save_current_results()
        assert status.startswith("❌")
        assert len(pd.read_csv(sample_folder / "analysis_results.csv")) == 0


class TestResuming:
    def test_opening_a_folder_again_ticks_what_was_saved(self, sample_folder):
        pd.DataFrame({"file_name": ["A_0001.tif"], "num_particles": [7]}).to_csv(
            sample_folder / "analysis_results.csv", index=False)

        status, _ = load(sample_folder)

        assert "1 already analysed" in status
        assert state.is_processed(0)
        assert not state.is_processed(1)
        # Carried on with, not replaced.
        rows = pd.read_csv(sample_folder / "analysis_results.csv")
        assert rows["file_name"].tolist() == ["A_0001.tif"]


class TestRefusals:
    def test_no_path(self):
        status, gallery = load("   ")
        assert status.startswith("❌")

    def test_a_path_that_is_not_a_folder(self, tmp_path):
        status, gallery = load(tmp_path / "missing")
        assert status.startswith("❌")
        assert state.image_paths == []

    def test_a_folder_without_images(self, tmp_path):
        status, gallery = load(tmp_path)
        assert status.startswith("❌") and "No images" in status

    @pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0,
                        reason="needs POSIX permissions and a non-root user")
    def test_a_folder_that_cannot_be_written_says_what_to_do(self, sample_folder):
        sample_folder.chmod(0o555)
        try:
            status, gallery = load(sample_folder)
        finally:
            sample_folder.chmod(0o755)
        assert status.startswith("❌") and "Copy the images" in status
        assert state.image_paths == []

    @pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0,
                        reason="needs POSIX permissions and a non-root user")
    def test_a_read_only_folder_with_results_is_refused_before_any_work(self, sample_folder):
        # The results file existing used to be enough to pass, and the first
        # save then failed after writing half of what it should have.
        pd.DataFrame({"file_name": ["A_0001.tif"], "num_particles": [7]}).to_csv(
            sample_folder / "analysis_results.csv", index=False)
        sample_folder.chmod(0o555)
        try:
            status, _ = load(sample_folder)
        finally:
            sample_folder.chmod(0o755)
        assert status.startswith("❌")
        assert state.image_paths == []


class TestReconnecting:
    """A browser refresh shows the session as it is, not the start-up page."""

    def test_without_a_model_load_images_stays_off(self):
        state.sam_model = None
        *_, load_images = setup_tab.restore_setup()
        assert load_images["interactive"] is False

    def test_with_a_model_it_is_named_and_images_can_be_loaded(self):
        class Device:
            type = "cpu"

        class Model:
            checkpoint = "/weights/sam_vit_b_01ec64.pth"
            model_type = "vit_b"
            device = Device()

        state.sam_model = Model()
        checkpoints, _download, _load_model, status, load_images = setup_tab.restore_setup()
        assert checkpoints["value"] == "/weights/sam_vit_b_01ec64.pth"
        assert "vit_b" in status
        assert load_images["interactive"] is True
