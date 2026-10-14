"""Shared set-up for the demo's tests."""
import getpass
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# The demo's modules are imported the way demo_app.py imports them.
for path in (REPO / "demo", REPO, REPO / "classification"):
    sys.path.insert(0, str(path))

# PyTorch and NumPy each ship an OpenMP runtime; on Windows they otherwise
# refuse to share a process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: needs the SAM checkpoint and DINOv2's weights; "
                   "deselect with -m 'not slow'")


@pytest.fixture(scope="session")
def sam_checkpoint():
    import download_sam_weights as weights

    named = os.environ.get("SAM_CHECKPOINT")
    path = Path(named) if named else weights.checkpoint_path("vit_b")
    if not path.is_file():
        pytest.skip("SAM ViT-B not downloaded; run python download_sam_weights.py --model vit_b")
    return path
