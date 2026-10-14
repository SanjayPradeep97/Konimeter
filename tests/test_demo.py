"""
The demo app, without its models: the images it ships, how a click or the
first guess becomes a mask, the SAM download, and what each button does.

SAM and the classifier are replaced by stand-ins here, so these run anywhere
in seconds; test_demo_models.py runs the real ones.
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demo"


def disc(shape, centre, radius):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - centre[1]) ** 2 + (xx - centre[0]) ** 2 <= radius ** 2


class TestTheShippedImages:
    rows = list(csv.DictReader(open(DEMO / "demo_images.csv")))

    def test_sixty_held_out_images_twenty_per_class(self):
        assert len(self.rows) == 60
        assert Counter(r["category"] for r in self.rows) == {
            "Fiber": 20, "Cluster": 20, "Matrix": 20}
        assert {r["split"] for r in self.rows} == {"test"}

    def test_every_image_has_its_expert_mask_at_the_same_size(self):
        for r in self.rows:
            stem = Path(r["filename"]).stem
            image = Image.open(DEMO / "images" / r["category"] / f"{stem}.png")
            mask = Image.open(DEMO / "masks" / f"{stem}_mask.png")
            assert image.size == mask.size, stem

    def test_the_classifier_heads_ship_with_them(self):
        assert (DEMO / "weights" / "dinov2_probe_heads.pt").is_file()


class TestChoosingAMask:
    """SAM offers three masks per prompt; the demo keeps the largest dark one."""

    def setup_method(self):
        self.gray = np.full((100, 100), 200, np.uint8)
        self.gray[30:70, 30:70] = 60                       # the particle
        self.part = np.zeros((100, 100), bool); self.part[40:60, 40:60] = True
        self.whole = np.zeros((100, 100), bool); self.whole[30:70, 30:70] = True
        self.film = np.ones((100, 100), bool); self.film[:5] = False

    def test_the_whole_particle_rather_than_a_part_of_it(self):
        from prompting import choose_mask

        chosen = choose_mask([self.part, self.whole, self.film], [0.9, 0.5, 0.4], self.gray)
        assert (chosen == self.whole).all()

    def test_sams_own_choice_when_asked_for(self):
        from prompting import MODE_SCORE, choose_mask

        chosen = choose_mask([self.part, self.whole, self.film], [0.9, 0.5, 0.4],
                             self.gray, MODE_SCORE)
        assert (chosen == self.part).all()


class TestTheFirstGuess:
    """Before any click, the demo prompts SAM where the particle probably is."""

    def test_it_lands_inside_a_dark_particle(self):
        from prompting import auto_point

        gray = np.full((300, 400), 200, np.uint8)
        gray[disc(gray.shape, (250, 120), 40)] = 50
        x, y = auto_point(gray)
        assert disc(gray.shape, (250, 120), 40)[y, x]

    def test_a_grid_bar_along_the_edge_is_passed_over(self):
        # A bar across the frame is darker and larger than the particle, but
        # it runs along the frame's edge, as grid bars do.
        from prompting import auto_point

        gray = np.full((300, 400), 200, np.uint8)
        gray[:, 340:] = 20
        gray[disc(gray.shape, (150, 150), 35)] = 60
        x, y = auto_point(gray)
        assert disc(gray.shape, (150, 150), 35)[y, x]

    def test_a_blank_frame_gives_its_middle(self):
        from prompting import auto_point

        assert auto_point(np.full((100, 200), 128, np.uint8)) == (100, 50)


class TestDownloadingSAM:
    def test_it_goes_to_sam_weights_or_where_SAM_WEIGHTS_DIR_says(self, monkeypatch, tmp_path):
        import download_sam_weights as weights

        monkeypatch.delenv("SAM_WEIGHTS_DIR", raising=False)
        assert weights.checkpoint_path() == REPO / "sam_weights" / "sam_vit_b_01ec64.pth"
        monkeypatch.setenv("SAM_WEIGHTS_DIR", str(tmp_path))
        assert weights.checkpoint_path() == tmp_path / "sam_vit_b_01ec64.pth"

    def test_the_file_appears_only_when_complete(self, monkeypatch, tmp_path):
        import download_sam_weights as weights

        def fetch(url, filename, hook):
            assert Path(filename).name.endswith(".part")
            hook(1, 10, 20)
            Path(filename).write_bytes(b"weights")
        monkeypatch.setattr(weights.urllib.request, "urlretrieve", fetch)
        seen = []
        path = weights.download("vit_b", tmp_path, progress=lambda *a: seen.append(a))
        assert path.read_bytes() == b"weights" and seen == [(10, 20)]
        assert not list(tmp_path.glob("*.part"))

    def test_an_interrupted_download_leaves_nothing(self, monkeypatch, tmp_path):
        import download_sam_weights as weights

        def fetch(url, filename, hook):
            Path(filename).write_bytes(b"half")
            raise OSError("connection reset")
        monkeypatch.setattr(weights.urllib.request, "urlretrieve", fetch)
        with pytest.raises(OSError):
            weights.download("vit_b", tmp_path)
        assert not list(tmp_path.iterdir())

    def test_a_downloaded_checkpoint_is_not_fetched_again(self, monkeypatch, tmp_path):
        import download_sam_weights as weights

        (tmp_path / "sam_vit_b_01ec64.pth").write_bytes(b"already here")
        monkeypatch.setattr(weights.urllib.request, "urlretrieve",
                            lambda *a: pytest.fail("downloaded again"))
        assert weights.download("vit_b", tmp_path).read_bytes() == b"already here"


# ---------------------------------------------------------------- the page

pytest.importorskip("gradio")


class FakeSAM:
    """Returns a disc round the last positive point, as three candidates."""
    def __init__(self):
        self.prompts = []

    def set_image(self, rgb):
        self.shape = rgb.shape[:2]

    def predict(self, coords, labels):
        self.prompts.append((coords.tolist(), labels.tolist()))
        x, y = coords[labels == 1][-1] if (labels == 1).any() else coords[-1]
        mask = disc(self.shape, (x, y), 30)
        return np.stack([mask, mask, mask]), np.array([0.5, 0.4, 0.3]), None


class FakeClassifier:
    def predict(self, gray, mask):
        return "Fiber", np.array([0.7, 0.1, 0.1, 0.1])


@pytest.fixture
def app(monkeypatch):
    import demo_app

    monkeypatch.setattr(demo_app.Models, "sam", FakeSAM())
    monkeypatch.setattr(demo_app.Models, "clf", FakeClassifier())
    monkeypatch.setattr(demo_app.Models, "image", None)
    return demo_app


class Select:
    """What Gradio hands a click handler."""
    def __init__(self, x, y):
        self.index = (x, y)


def opened(app, stem="CNT-Fiber-0014"):
    _image, _report, session = app.open_image(stem, app.Session(), progress=lambda *a, **k: None)
    return session


class TestThePage:
    def test_it_builds_without_any_model(self):
        import demo_app

        assert demo_app.build_app() is not None

    def test_without_sam_it_offers_the_download(self, monkeypatch, tmp_path):
        import demo_app

        monkeypatch.setenv("SAM_WEIGHTS_DIR", str(tmp_path))
        monkeypatch.delenv("SAM_CHECKPOINT", raising=False)
        panel, _ = demo_app.models_status()
        assert panel["visible"] is True

    def test_with_sam_it_does_not(self, monkeypatch, tmp_path):
        import demo_app

        (tmp_path / "sam_vit_b_01ec64.pth").write_bytes(b"x")
        monkeypatch.setenv("SAM_WEIGHTS_DIR", str(tmp_path))
        monkeypatch.delenv("SAM_CHECKPOINT", raising=False)
        panel, _ = demo_app.models_status()
        assert panel["visible"] is False

    def test_choosing_an_image_before_the_download_says_what_to_do(self, monkeypatch,
                                                                    tmp_path):
        import demo_app

        monkeypatch.setattr(demo_app.Models, "sam", None)
        monkeypatch.setenv("SAM_WEIGHTS_DIR", str(tmp_path))
        monkeypatch.delenv("SAM_CHECKPOINT", raising=False)
        image, report, _ = demo_app.open_image("CNT-Fiber-0014", demo_app.Session(),
                                               progress=lambda *a, **k: None)
        assert image is None and "Download the SAM model first" in report

    def test_next_goes_round_the_list(self):
        import demo_app

        assert demo_app.next_image(None) == demo_app.CHOICES[0]
        assert demo_app.next_image(demo_app.CHOICES[-1]) == demo_app.CHOICES[0]


class TestCorrectingTheMask:
    def test_opening_an_image_segments_and_classifies_it(self, app):
        s = opened(app)
        assert s.first_guess and len(s.points) == 1
        assert s.mask.any() and s.pred == "Fiber"
        assert "first guess" in app.report(s) and "Expert label" in app.report(s)

    def test_the_automatic_prompt_is_never_drawn(self, app):
        s = opened(app)
        assert s.clicks == []
        app.click(app.NEGATIVE, s, Select(100, 100))
        assert [label for _point, label in s.clicks] == [0]

    def test_a_click_on_the_particle_replaces_the_first_guess(self, app):
        s = opened(app)
        app.click(app.POSITIVE, s, Select(100, 100))
        assert len(s.points) == 1 and not s.first_guess and not s.seeded
        assert len(s.clicks) == 1 and "from 1 click" in app.report(s)
        scale = s.gray.shape[1] / app.DISPLAY_W
        assert s.points[0] == pytest.approx([100 * scale, 100 * scale])

    def test_a_background_click_trims_the_first_guess(self, app):
        s = opened(app)
        app.click(app.NEGATIVE, s, Select(100, 100))
        assert s.labels == [1, 0] and s.seeded

    def test_later_clicks_add_up(self, app):
        s = opened(app)
        app.click(app.POSITIVE, s, Select(100, 100))
        app.click(app.POSITIVE, s, Select(200, 200))
        app.click(app.NEGATIVE, s, Select(300, 300))
        assert s.labels == [1, 1, 0]

    def test_undo_steps_back_to_the_first_guess(self, app):
        s = opened(app)
        guess = list(s.points)
        app.click(app.POSITIVE, s, Select(100, 100))
        app.undo(s)
        assert s.points == guess and s.first_guess

    def test_reset_makes_the_first_guess_again(self, app):
        s = opened(app)
        app.click(app.POSITIVE, s, Select(100, 100))
        app.reset(s)
        assert s.first_guess and len(s.points) == 1 and not s.history

    def test_a_mask_brighter_than_its_surroundings_is_called_film(self, app):
        s = opened(app)
        s.mask = s.gray > np.percentile(s.gray, 90)
        assert "support film" in app.report(s)
