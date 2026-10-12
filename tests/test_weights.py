"""
Downloading SAM weights, from the Setup tab or download_sam_weights.py.

The download has to land where the app looks, or a user downloads the model on
every start; and it must never leave a half-written checkpoint behind, which
would later fail to load with an opaque torch error.
"""

from pathlib import Path

import pytest

from sem_particle_analysis import model as model_module
from sem_particle_analysis import weights


@pytest.fixture
def served(tmp_path, monkeypatch):
    """ViT-B 'served' from a local file, so nothing is fetched from the internet."""
    source = tmp_path / "server" / "sam_vit_b_01ec64.pth"
    source.parent.mkdir()
    source.write_bytes(b"x" * 300_000)
    monkeypatch.setitem(weights.MODELS["vit_b"], "url", source.as_uri())
    return source


class TestWhereWeightsLive:
    def test_downloads_go_where_the_app_looks(self, tmp_path, monkeypatch, served):
        monkeypatch.setenv("SAM_WEIGHTS_DIR", str(tmp_path / "weights"))
        path = weights.download("vit_b")
        assert path.parent == tmp_path / "weights"
        assert path in model_module.discover_checkpoints()

    def test_by_default_that_is_konimeters_own_folder(self, monkeypatch):
        monkeypatch.delenv("SAM_WEIGHTS_DIR", raising=False)
        package = Path(model_module.__file__).resolve().parent
        assert model_module.weights_dir() == package.parent / "sam_weights"

    def test_the_app_folder_is_searched_from_anywhere(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SAM_WEIGHTS_DIR", raising=False)
        assert model_module.weights_dir() in model_module._default_search_dirs()


class TestDownloading:
    def test_writes_the_checkpoint_and_reports_progress(self, tmp_path, served):
        seen = []
        path = weights.download("vit_b", directory=tmp_path / "w",
                                progress=lambda done, total: seen.append((done, total)))
        assert path.read_bytes() == served.read_bytes()
        assert seen and seen[-1][0] == seen[-1][1] == served.stat().st_size

    def test_an_existing_checkpoint_is_not_fetched_again(self, tmp_path, monkeypatch):
        (tmp_path / "sam_vit_b_01ec64.pth").write_bytes(b"already here")
        monkeypatch.setitem(weights.MODELS["vit_b"], "url", "file:///no/such/file")
        path = weights.download("vit_b", directory=tmp_path)
        assert path.read_bytes() == b"already here"

    def test_a_failed_download_leaves_nothing_behind(self, tmp_path, monkeypatch):
        monkeypatch.setitem(weights.MODELS["vit_b"], "url",
                            (tmp_path / "missing.pth").as_uri())
        with pytest.raises(Exception):
            weights.download("vit_b", directory=tmp_path / "w")
        assert list((tmp_path / "w").iterdir()) == []


class TestTheSetupTabButton:
    @pytest.fixture
    def setup_tab(self, monkeypatch):
        pytest.importorskip("gradio")
        from sem_analysis_app.callbacks import setup

        # Loading the real model is the SAM test's business, not this one's.
        monkeypatch.setattr(setup, "initialize_sam",
                            lambda path, progress=None: (f"✅ loaded {Path(path).name}", "load"))
        return setup

    def test_it_downloads_then_loads_and_goes_away(self, setup_tab, tmp_path,
                                                    monkeypatch, served):
        monkeypatch.setenv("SAM_WEIGHTS_DIR", str(tmp_path / "weights"))
        status, checkpoints, button, load_images, load_model = setup_tab.download_model(
            progress=lambda *a, **k: None)
        assert status == "✅ loaded sam_vit_b_01ec64.pth"
        assert checkpoints["value"].endswith("sam_vit_b_01ec64.pth")
        assert button["visible"] is False
        assert load_images == "load"
        assert load_model["variant"] == "primary"

    def test_a_failure_says_so_and_keeps_the_button(self, setup_tab, tmp_path, monkeypatch):
        monkeypatch.setenv("SAM_WEIGHTS_DIR", str(tmp_path / "weights"))
        monkeypatch.setitem(weights.MODELS["vit_b"], "url",
                            (tmp_path / "missing.pth").as_uri())
        status, _checkpoints, button, _load, _model = setup_tab.download_model(
            progress=lambda *a, **k: None)
        assert status.startswith("❌") and "internet connection" in status
        assert button["visible"] is True
