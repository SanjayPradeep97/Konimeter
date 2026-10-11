"""
Batch CLI argument contract.

The defaults here end up in every published number and in the run.json
provenance record, so a silent change to one is a silent change to results.
"""

from pathlib import Path

import pytest

from sem_particle_analysis import cli


@pytest.fixture
def parser():
    return cli.build_parser()


class TestDefaults:
    def test_defaults_match_the_documented_pipeline(self, parser):
        args = parser.parse_args(["images"])
        assert args.input == Path("images")
        assert args.output == Path("analysis_output")
        assert args.model_type is None  # chosen by choose_model()
        assert args.scale_method == "auto"
        assert args.scale_nm_per_px is None
        assert args.min_size == 30
        assert args.border_buffer == 4
        assert args.clear_edges is False
        assert args.no_plots is False

    def test_crop_percent_defaults_to_measuring_the_databar(self, parser):
        # None means "measure it"; a fixed percentage is fragile in both
        # directions, so it must not become the default by accident.
        assert parser.parse_args(["images"]).crop_percent is None

    def test_crop_percent_zero_keeps_the_full_frame(self, parser):
        assert parser.parse_args(["images", "--crop-percent", "0"]).crop_percent == 0.0


class TestOverrides:
    def test_output_directory(self, parser):
        args = parser.parse_args(["images", "-o", "out"])
        assert args.output == Path("out")

    def test_fixed_scale(self, parser):
        args = parser.parse_args(["images", "--scale-nm-per-px", "2.5"])
        assert args.scale_nm_per_px == pytest.approx(2.5)

    def test_clear_edges_is_a_flag(self, parser):
        assert parser.parse_args(["images", "--clear-edges"]).clear_edges is True

    def test_min_size(self, parser):
        assert parser.parse_args(["images", "--min-size", "120"]).min_size == 120

    @pytest.mark.parametrize("model", ["vit_b", "vit_h", "vit_l"])
    def test_accepts_every_sam_variant(self, parser, model):
        assert parser.parse_args(["images", "--model-type", model]).model_type == model

    @pytest.mark.parametrize("method", ["auto", "metadata", "ocr"])
    def test_accepts_every_scale_method(self, parser, method):
        assert parser.parse_args(["images", "--scale-method", method]).scale_method == method


class TestRejections:
    def test_requires_an_input(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_rejects_an_unknown_model(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["images", "--model-type", "vit_xl"])

    def test_rejects_an_unknown_scale_method(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["images", "--scale-method", "guess"])


class TestMainErrorPaths:
    def test_missing_input_exits_non_zero(self, tmp_path, capsys):
        assert cli.main([str(tmp_path / "nope"), "-o", str(tmp_path / "out")]) == 2
        assert "no such file or folder" in capsys.readouterr().err

    def test_empty_folder_exits_non_zero(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert cli.main([str(empty), "-o", str(tmp_path / "out")]) == 2
        assert "no images found" in capsys.readouterr().err


class TestChoosingTheModel:
    VIT_B = Path("weights") / "sam_vit_b_01ec64.pth"
    VIT_H = Path("weights") / "sam_vit_h_4b8939.pth"

    def test_a_checkpoint_alone_is_loaded_as_what_its_name_says(self):
        # Loading ViT-B weights as ViT-H fails inside torch with a shape error.
        assert cli.choose_model(self.VIT_B, None) == (self.VIT_B, "vit_b", None)

    def test_a_type_that_contradicts_the_checkpoint_is_refused(self):
        checkpoint, model_type, problem = cli.choose_model(self.VIT_B, "vit_h")
        assert checkpoint is None and "does not match" in problem

    def test_an_unrecognisable_name_needs_the_type_spelled_out(self):
        _, _, problem = cli.choose_model(Path("finetuned.pth"), None)
        assert "--model-type" in problem
        assert cli.choose_model(Path("finetuned.pth"), "vit_l")[1] == "vit_l"

    def test_with_neither_it_takes_the_apps_default(self, monkeypatch):
        decoder_only = Path("weights") / "decoder.pth"
        monkeypatch.setattr(cli, "discover_checkpoints",
                            lambda: [self.VIT_B, self.VIT_H, decoder_only])
        assert cli.choose_model(None, None) == (self.VIT_B, "vit_b", None)

    def test_with_nothing_downloaded_there_is_nothing_to_load(self, monkeypatch):
        monkeypatch.setattr(cli, "discover_checkpoints", lambda: [])
        assert cli.choose_model(None, None) == (None, None, None)

    def test_missing_weights_say_which_to_download(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "discover_checkpoints", lambda: [])
        folder = tmp_path / "images"
        folder.mkdir()
        (folder / "a.png").write_bytes(b"")
        assert cli.main([str(folder), "-o", str(tmp_path / "out")]) == 2
        assert "download_sam_weights.py --model vit_b" in capsys.readouterr().err


class TestDistributionPlots:
    @staticmethod
    def particles(units):
        import pandas as pd
        return pd.DataFrame({"unit": units, "diameter": range(1, len(units) + 1),
                             "area": range(1, len(units) + 1)})

    def test_one_unit_keeps_the_plain_file_names(self, tmp_path):
        written = cli.write_distributions(self.particles(["nm"] * 4), tmp_path)
        assert written == ["size_distribution.png", "size_distribution.pdf"]

    def test_nanometres_and_pixels_are_never_pooled(self, tmp_path):
        written = cli.write_distributions(
            self.particles(["nm", "nm", "pixels", "nm"]), tmp_path)
        assert sorted(written) == sorted([
            "size_distribution_nm.png", "size_distribution_nm.pdf",
            "size_distribution_pixels.png", "size_distribution_pixels.pdf"])

    def test_nothing_measured_draws_nothing(self, tmp_path):
        assert cli.write_distributions(self.particles([]), tmp_path) == []


class TestProvenanceHelpers:
    def test_sha256_of_a_known_file(self, tmp_path):
        # Hashing the inputs is what makes a published result traceable.
        path = tmp_path / "f.bin"
        path.write_bytes(b"abc")
        expected = ("ba7816bf8f01cfea414140de5dae2223"
                    "b00361a396177a9cb410ff61f20015ad")
        assert cli.sha256_file(path) == expected

    def test_sha256_reads_in_chunks(self, tmp_path):
        # Checkpoints are multi-GB; the hash must not load one into memory.
        path = tmp_path / "big.bin"
        path.write_bytes(b"x" * 100_000)
        assert cli.sha256_file(path, chunk_size=1024) == cli.sha256_file(path)

    def test_versions_reports_the_running_python(self):
        versions = cli._versions()
        assert versions["python"]
        assert "platform" in versions
        for library in ("numpy", "pandas", "skimage"):
            assert library in versions


class TestResolveScale:
    class _Args:
        scale_nm_per_px = None
        scale_method = "auto"

    def test_a_fixed_scale_short_circuits_detection(self):
        args = self._Args()
        args.scale_nm_per_px = 2.5

        class Boom:
            def detect_scale(self, *a, **k):
                raise AssertionError("detection must not run")

        nm_per_px, info = cli.resolve_scale(Boom(), None, "x.tif", args)
        assert nm_per_px == pytest.approx(2.5)
        assert info["method"] == "manual"

    def test_detection_failure_falls_back_to_pixels_rather_than_raising(self):
        # A missing scale must not abort a batch; the image is still measurable
        # in pixels, and run.json records that scale failed.
        class Broken:
            def detect_scale(self, *a, **k):
                raise ValueError("no bar")

        nm_per_px, info = cli.resolve_scale(Broken(), None, "x.tif", self._Args())
        assert nm_per_px is None
        assert info["method"] == "failed"
        assert "no bar" in info["error"]

    def test_a_successful_detection_is_reported_with_provenance(self):
        class Works:
            def detect_scale(self, *a, **k):
                return {"conversion": 2.5, "method": "metadata", "scale_nm": 2.5,
                        "pixel_length": 1, "ocr_text": None, "warning": None}

        nm_per_px, info = cli.resolve_scale(Works(), None, "x.tif", self._Args())
        assert nm_per_px == pytest.approx(2.5)
        assert info["method"] == "metadata"
