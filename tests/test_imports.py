"""
Import-time dependency boundaries.

The measurement path must not drag in the optional heavy stack. Importing
``sem_particle_analysis`` used to pull EasyOCR through ScaleDetector, so a
checkout without OCR installed could not import the analyzer, the segmenter or
even the pure-arithmetic scale calibration module.

The facts are gathered by one child interpreter rather than one per assertion:
importing the package pulls in torch, which costs several seconds a time.
"""

import importlib.util
import json
import subprocess
import sys
import textwrap

import pytest


PROBE = """
    import json, sys

    report = {}

    import sem_particle_analysis as package
    report["exports"] = sorted(package.__all__)
    report["pulled_in"] = sorted(
        name for name in ("easyocr", "gradio") if name in sys.modules
    )

    report["standalone"] = {}
    for module in ("analysis", "scale_calibration", "segmentation", "utils", "_compat"):
        try:
            __import__(f"sem_particle_analysis.{module}")
            report["standalone"][module] = "ok"
        except Exception as exc:
            report["standalone"][module] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report))
"""


@pytest.fixture(scope="module")
def probe():
    """Import the package in a clean interpreter and report what happened.

    The interpreter is pointed at the repository root, as conftest.py does for
    the in-process tests, so the probe works from a plain checkout as well as
    from an editable install."""
    import os
    from pathlib import Path
    repo_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(PROBE)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert result.returncode == 0, (
        f"importing sem_particle_analysis failed:\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestOptionalDependenciesStayOptional:
    def test_importing_the_package_pulls_in_neither_easyocr_nor_gradio(self, probe):
        # EasyOCR costs seconds of model loading and is needed only to read a
        # printed scale bar; gradio is only for the web app. Metadata-based
        # scale, segmentation and measurement must all work without either.
        assert probe["pulled_in"] == []

    def test_the_public_api_is_importable(self, probe):
        assert {"ParticleAnalyzer", "ParticleSegmenter", "ScaleDetector",
                "ResultsManager", "SAMModel", "OCRUnavailableError",
                "discover_checkpoints", "infer_model_type"} <= set(probe["exports"])

    @pytest.mark.parametrize(
        "module", ["analysis", "scale_calibration", "segmentation", "utils", "_compat"]
    )
    def test_measurement_modules_import_standalone(self, probe, module):
        # Each must be importable on its own, not only via the package __init__.
        assert probe["standalone"][module] == "ok"


class TestOCRUnavailableError:
    def test_is_a_value_error_so_auto_mode_still_falls_back(self):
        # detect_scale(method='auto') catches ValueError around the OCR attempt
        # and falls through to metadata. A missing library must take that path
        # rather than crashing the run.
        from sem_particle_analysis import OCRUnavailableError

        assert issubclass(OCRUnavailableError, ValueError)

    def test_constructing_a_detector_builds_no_reader(self):
        from sem_particle_analysis import ScaleDetector

        assert ScaleDetector(use_gpu=False)._reader is None

    @pytest.mark.skipif(
        importlib.util.find_spec("easyocr") is not None,
        reason="EasyOCR is installed, so the reader builds normally",
    )
    def test_reader_access_explains_how_to_fix_it(self):
        from sem_particle_analysis import OCRUnavailableError, ScaleDetector

        with pytest.raises(OCRUnavailableError, match="easyocr"):
            ScaleDetector(use_gpu=False).reader
