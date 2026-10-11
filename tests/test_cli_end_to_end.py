"""
The batch pipeline end to end, with real SAM weights.

Everything else in the suite stops short of the model, so this is the one test
that a checkout can actually segment: a synthetic frame of dark disks of known
size goes through konimeter-batch, and the particles, the measurements and the
provenance record that come out are checked.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sem_particle_analysis import cli

pytestmark = pytest.mark.slow

SCALE_NM_PER_PX = 2.0
DISKS = ((120, 130, 30), (330, 110, 24), (240, 300, 36), (110, 400, 26), (400, 380, 32))


@pytest.fixture(scope="module")
def run(tmp_path_factory, sam_checkpoint):
    folder = tmp_path_factory.mktemp("images")
    rng = np.random.default_rng(0)
    frame = np.clip(rng.normal(190, 6, size=(512, 512)), 0, 255)
    yy, xx = np.mgrid[0:512, 0:512]
    for cx, cy, r in DISKS:
        frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 60
    Image.fromarray(frame.astype(np.uint8)).convert("RGB").save(folder / "disks.png")

    out = tmp_path_factory.mktemp("out")
    code = cli.main([str(folder), "-o", str(out), "--checkpoint", str(sam_checkpoint),
                     "--scale-nm-per-px", str(SCALE_NM_PER_PX),
                     "--modality", "TEM", "--particles", "dark",
                     "--device", "cpu"])
    return code, out


def test_the_run_succeeds_and_writes_every_output(run):
    code, out = run
    assert code == 0
    for name in ("particles.csv", "per_image_summary.csv", "run.json",
                 "size_distribution.png", "size_distribution.pdf"):
        assert (out / name).exists(), name


def test_every_disk_is_found_at_its_true_size(run):
    _, out = run
    particles = pd.read_csv(out / "particles.csv")
    assert len(particles) == len(DISKS)
    assert set(particles["unit"]) == {"nm"}

    found = sorted(particles["diameter"])
    truth = sorted(2 * r * SCALE_NM_PER_PX for _, _, r in DISKS)
    assert found == pytest.approx(truth, rel=0.08)


def test_the_run_record_identifies_model_and_code(run):
    _, out = run
    record = json.loads((out / "run.json").read_text())
    assert record["model"]["type"] == "vit_b"
    assert len(record["model"]["sha256"]) == 64
    # The checkout's own commit, so a result can be traced to the code.
    if (Path(cli.__file__).resolve().parents[1] / ".git").exists():
        assert record["git_revision"] and len(record["git_revision"]) == 40
