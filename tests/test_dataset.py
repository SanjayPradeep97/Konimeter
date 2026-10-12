"""
Training data kept from saved frames: the image, its masks and an index row.

A saved frame's mask has been checked by the analyst, so the app keeps it as a
label. These tests pin down the format, because the point of keeping it is that
it can be read back by something else: the classification pipeline reads
<stem>_mask.png at 0/255, and the label image has to number particles exactly
as the results row lists them.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sem_particle_analysis import ParticleAnalyzer, ResultsManager
from sem_particle_analysis.dataset import DIRNAME, TrainingSet, instance_labels
from synthetic import make_disk_mask

DISKS = ((60, 60, 20), (140, 120, 14), (60, 150, 9))


@pytest.fixture
def analysed():
    mask, _ = make_disk_mask(shape=(200, 220), centers_radii=DISKS)
    analyzer = ParticleAnalyzer(conversion_factor=2.0, min_size=30)
    analyzer.analyze_mask(mask, remove_border=False)
    frame = np.full((200, 220, 3), 180, dtype=np.uint8)
    frame[mask.astype(bool)] = 60
    return frame, analyzer


def read(path):
    return np.array(Image.open(path))


class TestFormat:
    def test_labels_number_particles_as_the_results_row_lists_them(self, analysed):
        frame, analyzer = analysed
        labels = instance_labels(analyzer.regions, frame.shape)
        assert labels.dtype == np.uint16
        assert labels.max() == len(analyzer.regions)
        for number, region in enumerate(analyzer.regions, start=1):
            assert (labels == number).sum() == region.area

    def test_a_saved_frame_reads_back_as_it_was_written(self, analysed, tmp_path):
        frame, analyzer = analysed
        labels = instance_labels(analyzer.regions, frame.shape)
        valid = np.ones(frame.shape[:2], dtype=bool)
        valid[:, :10] = False
        dataset = TrainingSet(tmp_path / DIRNAME)
        dataset.save("A_0001", frame, labels, valid, scale_method="metadata")

        paths = dataset.paths("A_0001")
        image = Image.open(paths["image"])
        assert image.mode == "L"  # identical channels are stored once
        assert np.array_equal(np.array(image), frame[..., 0])

        mask = read(paths["mask"])
        assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 255}
        assert np.array_equal(mask == 255, labels > 0)

        assert np.array_equal(read(paths["labels"]).astype(np.uint16), labels)
        assert np.array_equal(read(paths["valid"]) == 255, valid)

    def test_the_mask_is_named_as_the_classification_pipeline_expects(self, analysed,
                                                                      tmp_path):
        frame, analyzer = analysed
        dataset = TrainingSet(tmp_path / DIRNAME)
        dataset.save("CNT-Fiber-0001", frame, instance_labels(analyzer.regions, frame.shape))
        assert (tmp_path / DIRNAME / "masks" / "CNT-Fiber-0001_mask.png").exists()

    def test_colour_frames_keep_their_colour(self, analysed, tmp_path):
        frame, analyzer = analysed
        frame = frame.copy()
        frame[..., 2] = 7
        dataset = TrainingSet(tmp_path / DIRNAME)
        dataset.save("c", frame, instance_labels(analyzer.regions, frame.shape))
        assert Image.open(dataset.paths("c")["image"]).mode == "RGB"

    def test_mismatched_sizes_are_refused(self, analysed, tmp_path):
        frame, analyzer = analysed
        labels = instance_labels(analyzer.regions, frame.shape)
        with pytest.raises(ValueError):
            TrainingSet(tmp_path).save("x", frame[:-1], labels)
        with pytest.raises(ValueError):
            TrainingSet(tmp_path).save("x", frame, labels, np.ones((5, 5), bool))


class TestIndex:
    def test_one_row_per_frame_and_saving_again_replaces_it(self, analysed, tmp_path):
        frame, analyzer = analysed
        labels = instance_labels(analyzer.regions, frame.shape)
        dataset = TrainingSet(tmp_path / DIRNAME)
        dataset.save("a", frame, labels, scale_method="box_ocr+unconfirmed")
        dataset.save("b", frame, labels)
        dataset.save("a", frame, labels, scale_method="box_ocr")

        rows = dataset.rows()
        assert sorted(r["stem"] for r in rows) == ["a", "b"]
        a = next(r for r in rows if r["stem"] == "a")
        assert a["scale_method"] == "box_ocr"
        assert a["num_particles"] == str(len(DISKS))
        assert a["mask"] == "masks/a_mask.png"
        assert (a["width"], a["height"]) == ("220", "200")
        assert (tmp_path / DIRNAME / "README.md").exists()

    def test_removing_a_frame_removes_its_files_and_row(self, analysed, tmp_path):
        frame, analyzer = analysed
        dataset = TrainingSet(tmp_path / DIRNAME)
        dataset.save("a", frame, instance_labels(analyzer.regions, frame.shape))

        assert dataset.remove("a") is True
        assert dataset.rows() == []
        assert not any(path.exists() for path in dataset.paths("a").values())
        assert dataset.remove("a") is False


class TestTheApp:
    """Saving a frame in the app keeps it as training data; deleting it drops it."""

    @pytest.fixture
    def app(self, analysed, tmp_path):
        pytest.importorskip("gradio")
        from sem_analysis_app.state import state

        frame, analyzer = analysed
        saved = dict(vars(state))
        source = tmp_path / "A_0001.tif"
        Image.fromarray(frame).save(source)
        state.image_paths = [str(source)]
        state.current_index = 0
        state.current_image = np.vstack([frame, np.zeros((24, 220, 3), np.uint8)])
        state.cropped_image = frame
        state.analyzer = analyzer
        state.results_manager = ResultsManager(csv_file=str(tmp_path / "analysis_results.csv"))
        state.scale_calibration = None
        state.save_dataset = True
        yield state, tmp_path
        vars(state).clear()
        vars(state).update(saved)

    def test_saving_a_frame_adds_it_to_the_dataset(self, app):
        from sem_analysis_app.callbacks.results import save_current_results

        state, folder = app
        status, _ = save_current_results()
        assert status.startswith("✅") and DIRNAME in status, status

        rows = TrainingSet(folder / DIRNAME).rows()
        assert [r["stem"] for r in rows] == ["A_0001"]
        row = rows[0]
        assert row["source_file"] == "A_0001.tif"  # a name, never a path
        assert len(row["source_sha256"]) == 64
        assert row["rows_trimmed"] == "24"
        assert row["scale_method"] == "none"

        saved = pd.read_csv(folder / "analysis_results.csv")
        assert saved["num_particles"].iloc[0] == int(row["num_particles"])

    def test_switched_off_it_saves_results_only(self, app):
        from sem_analysis_app.callbacks.results import save_current_results

        state, folder = app
        state.save_dataset = False
        status, _ = save_current_results()
        assert status.startswith("✅")
        assert not (folder / DIRNAME).exists()

    def test_switched_off_a_resave_drops_the_now_stale_copy(self, app):
        # The dataset must never disagree with the row the analyst just saved.
        from sem_analysis_app.callbacks.results import save_current_results

        state, folder = app
        save_current_results()
        state.save_dataset = False
        save_current_results()
        assert TrainingSet(folder / DIRNAME).rows() == []

    def test_deleting_a_result_deletes_its_training_data(self, app):
        from sem_analysis_app.callbacks.results import (delete_result_row,
                                                         save_current_results)

        state, folder = app
        save_current_results()
        *_, message, _gallery = delete_result_row("A_0001.tif")
        assert message.startswith("✅"), message
        assert TrainingSet(folder / DIRNAME).rows() == []
        assert not (folder / DIRNAME / "masks" / "A_0001_mask.png").exists()
