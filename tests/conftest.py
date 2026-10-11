import getpass
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Test helpers live alongside the tests rather than in the installed package.
sys.path.insert(0, str(Path(__file__).parent))
# The library under test, taken from this checkout rather than from whatever is
# installed. Both apps import sem_particle_analysis by name, and a development
# install resolves that name to the directory pip was pointed at — which inside
# a git worktree is the *other* checkout. Without this the suite silently
# exercises code that is not the code being changed, and a new argument to a
# library function comes back as "unexpected keyword argument".
sys.path.insert(0, str(Path(__file__).parent.parent))


def _ensure_usable_temp_root():
    """
    Make sure pytest has somewhere writable to put ``tmp_path`` directories.

    pytest creates per-run temp directories under ``<TEMP>/pytest-of-<user>``. On
    Windows that directory is sometimes left behind by a run under a different or
    elevated identity, with ACLs the current user cannot read, rename or delete
    without elevation. Every test taking ``tmp_path`` then errors at setup with
    ``PermissionError: [WinError 5]``, and no amount of cleaning the working tree
    helps.

    Rather than require a manual, elevated fix before the suite can run at all,
    probe the default root and redirect to a fresh one if it is unusable.
    ``PYTEST_DEBUG_TEMPROOT`` names the *parent*; pytest creates its own
    ``pytest-of-<user>`` inside it.

    Runs at conftest import, which is before the temp factory first resolves a
    path. An explicit ``PYTEST_DEBUG_TEMPROOT`` is always left alone.
    """
    if os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        return

    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - getuser can fail on odd systems
        return

    root = Path(tempfile.gettempdir()) / f"pytest-of-{user}"
    if not root.exists():
        return

    probe = root / ".write-probe"
    try:
        probe.mkdir(exist_ok=True)
        probe.rmdir()
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "pytest-temproot"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - nothing writable anywhere
            return
        os.environ["PYTEST_DEBUG_TEMPROOT"] = str(fallback)


_ensure_usable_temp_root()

# Where download_sam_weights.py puts SAM weights. SAM_CHECKPOINT names a file
# elsewhere.
_WEIGHT_DIRS = [
    Path(__file__).parent.parent / "sam_weights",
]
_WEIGHT_FILES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def find_sam_checkpoint(model_type="vit_b"):
    """Return a path to local SAM weights, or None if they aren't downloaded."""
    env = os.environ.get("SAM_CHECKPOINT")
    if env and Path(env).exists():
        return Path(env)
    for directory in _WEIGHT_DIRS:
        candidate = directory / _WEIGHT_FILES[model_type]
        if candidate.exists():
            return candidate
    return None


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs SAM weights or OCR models; deselect with -m 'not slow'")


@pytest.fixture(scope="session")
def scale_detector():
    """
    A single ScaleDetector for the whole session.

    EasyOCR takes several seconds to spin up its models, so building one per test
    would dominate the runtime.
    """
    pytest.importorskip("easyocr")
    from sem_particle_analysis import ScaleDetector

    return ScaleDetector(use_gpu=False)


@pytest.fixture(scope="session")
def sam_checkpoint():
    path = find_sam_checkpoint("vit_b")
    if path is None:
        pytest.skip("SAM weights not available; run python download_sam_weights.py")
    return path
