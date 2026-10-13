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


def load(folder, microscope=None):
    status, gallery, _tabs, _microscope = setup_tab.load_images_from_folder(
        str(folder), microscope, progress=no_progress)
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
        _status, _gallery, tabs, _microscope = setup_tab.load_images_from_folder(
            str(sample_folder), "SEM", progress=no_progress)
        assert tabs.selected == 1

    def test_it_stays_on_setup_while_the_microscope_is_unknown(self, sample_folder):
        # The question is asked on Setup, and would go unseen from the Gallery.
        import gradio as gr

        _status, _gallery, tabs, _microscope = setup_tab.load_images_from_folder(
            str(sample_folder), progress=no_progress)
        assert not isinstance(tabs, gr.Tabs)

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
        *_, choose_folder, _scope, _folder = setup_tab.restore_setup()
        assert choose_folder["interactive"] is False

    def test_with_a_model_it_is_named_and_images_can_be_loaded(self):
        class Device:
            type = "cpu"

        class Model:
            checkpoint = "/weights/sam_vit_b_01ec64.pth"
            model_type = "vit_b"
            device = Device()

        state.sam_model = Model()
        (checkpoints, _download, _load_model, status, choose_folder, _scope,
         _folder) = setup_tab.restore_setup()
        assert checkpoints["value"] == "/weights/sam_vit_b_01ec64.pth"
        assert "vit_b" in status
        assert choose_folder["interactive"] is True

    def test_the_open_folder_is_shown_again(self, sample_folder):
        load(sample_folder)
        *_, folder = setup_tab.restore_setup()
        assert os.path.samefile(folder, sample_folder)


class TestTheMicroscope:
    """Which microscope a folder came from sets whether particles are expected
    darker (TEM) or brighter (SEM)."""

    def test_the_files_say_when_they_can(self, tmp_path):
        import tifffile

        for name in ("A_0001.tif", "A_0002.tif"):
            tifffile.imwrite(str(tmp_path / name), np.zeros((32, 32), np.uint8),
                             description="NIOSH\rXpixCal=620.66\rUnit=um")
        status, _ = load(tmp_path)
        assert state.microscope == "TEM"
        assert "TEM from the files" in status

    def test_a_choice_is_kept_and_a_disagreement_is_reported(self, tmp_path):
        import tifffile

        tifffile.imwrite(str(tmp_path / "A_0001.tif"), np.zeros((32, 32), np.uint8),
                         description="NIOSH\rXpixCal=620.66\rUnit=um")
        status, _ = load(tmp_path, microscope="SEM")
        assert state.microscope == "SEM"
        assert "the files say TEM" in status

    def test_silent_files_ask_for_a_choice(self, sample_folder):
        status, _ = load(sample_folder)
        assert state.microscope is None
        assert "choose SEM or TEM" in status

    def test_the_choice_sets_which_way_round_particles_are(self):
        from sem_analysis_app.callbacks.segmentation import _particle_polarity

        state.particle_choice = "auto"
        for choice, dark in (("TEM", True), ("SEM", False)):
            setup_tab.set_microscope(choice)
            assert _particle_polarity() is dark
        state.particle_choice = "bright"     # an explicit choice still wins
        setup_tab.set_microscope("TEM")
        assert _particle_polarity() is False


class TestChoosingTheFolder:
    """The folder is chosen in the system's own dialog; it is never typed."""

    def choose(self, monkeypatch, answer):
        from sem_analysis_app import folder_dialog

        def dialog(start=None):
            if isinstance(answer, Exception):
                raise answer
            return answer
        monkeypatch.setattr(folder_dialog, "choose_folder", dialog)
        return setup_tab.browse_for_folder("SEM", progress=no_progress)

    def test_the_chosen_folder_is_opened(self, monkeypatch, sample_folder):
        shown, status, gallery, tabs, _microscope = self.choose(monkeypatch,
                                                                str(sample_folder))
        assert shown == str(sample_folder)
        assert status.startswith("✅") and len(gallery) == 3
        assert tabs.selected == 1

    def test_cancelling_changes_nothing(self, monkeypatch, sample_folder):
        load(sample_folder)
        outputs = self.choose(monkeypatch, None)
        assert all(output == {"__type__": "update"} for output in outputs)
        assert len(state.image_paths) == 3

    def test_a_folder_without_images_leaves_the_last_one_open(self, monkeypatch,
                                                              sample_folder, tmp_path):
        load(sample_folder)
        empty = tmp_path / "empty"
        empty.mkdir()
        shown, status, *_ = self.choose(monkeypatch, str(empty))
        assert status.startswith("❌") and "No images" in status
        assert shown == {"__type__": "update"}
        assert len(state.image_paths) == 3

    def test_a_dialog_that_cannot_open_says_so(self, monkeypatch):
        from sem_analysis_app.folder_dialog import FolderDialogError

        shown, status, *_ = self.choose(monkeypatch, FolderDialogError("no display"))
        assert status.startswith("❌") and "no display" in status


class TestTheFolderDialog:
    """What the dialog process hands back, on each kind of system."""

    class Finished:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode, self.stdout, self.stderr = returncode, stdout, stderr

    def run(self, monkeypatch, platform, result):
        from sem_analysis_app import folder_dialog

        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return result
        monkeypatch.setattr(folder_dialog.sys, "platform", platform)
        monkeypatch.setattr(folder_dialog.subprocess, "run", fake_run)
        return folder_dialog, calls

    def test_macos_returns_the_folder_without_its_trailing_slash(self, monkeypatch):
        dialog, calls = self.run(monkeypatch, "darwin",
                                 self.Finished(stdout="/Users/me/Sample A/\n"))
        assert dialog.choose_folder() == os.path.normpath("/Users/me/Sample A")
        assert calls[0][0] == "osascript"

    def test_macos_cancel_is_not_an_error(self, monkeypatch):
        dialog, _ = self.run(monkeypatch, "darwin", self.Finished(
            returncode=1, stderr="execution error: User canceled. (-128)"))
        assert dialog.choose_folder() is None

    def test_macos_failure_is_reported(self, monkeypatch):
        dialog, _ = self.run(monkeypatch, "darwin",
                             self.Finished(returncode=1, stderr="something broke"))
        with pytest.raises(dialog.FolderDialogError, match="something broke"):
            dialog.choose_folder()

    def test_a_quote_in_a_folder_name_cannot_break_the_script(self):
        # Built as text, not by making such a folder: Windows forbids the name.
        from sem_analysis_app.folder_dialog import _applescript_text

        assert _applescript_text('say "hi"') == '"say \\"hi\\""'
        assert _applescript_text("back\\slash") == '"back\\\\slash"'

    def test_windows_reads_the_path_tk_prints(self, monkeypatch):
        dialog, calls = self.run(monkeypatch, "win32", self.Finished(
            stdout="C:/Data/Sample A".encode("utf-8")))
        assert dialog.choose_folder() == os.path.normpath("C:/Data/Sample A")
        assert calls[0][0] == sys.executable

    def test_windows_cancel_prints_nothing(self, monkeypatch):
        dialog, _ = self.run(monkeypatch, "win32", self.Finished(stdout=b""))
        assert dialog.choose_folder() is None

    def test_no_tk_is_reported(self, monkeypatch):
        dialog, _ = self.run(monkeypatch, "linux", self.Finished(
            returncode=1, stderr=b"_tkinter.TclError: no display name"))
        with pytest.raises(dialog.FolderDialogError, match="no display"):
            dialog.choose_folder()
