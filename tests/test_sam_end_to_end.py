"""
The Segment & Refine tab end to end, with real SAM weights.

Everything else in the suite stops short of the model, so this is the one test
that runs SAM: a synthetic TEM-like frame of dark disks of known size goes
through the tab's own callbacks, as a click on Segment and then on the first
candidate would send it, and the particles that come out are checked.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("gradio")

SCALE_NM_PER_PX = 2.0
DISKS = ((120, 130, 30), (330, 110, 24), (240, 300, 36), (110, 400, 26), (400, 380, 32))


def no_progress(*args, **kwargs):
    pass


@pytest.fixture(scope="module")
def segmented(sam_checkpoint):
    from sem_particle_analysis import ParticleSegmenter, SAMModel
    from sem_particle_analysis import modality as mod
    from sem_analysis_app.callbacks.segmentation import (segment_with_sam,
                                                         select_mask_and_analyze)
    from sem_analysis_app.state import state

    rng = np.random.default_rng(0)
    frame = np.clip(rng.normal(190, 6, size=(512, 512)), 0, 255)
    yy, xx = np.mgrid[0:512, 0:512]
    for cx, cy, r in DISKS:
        frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 60
    frame = np.repeat(frame.astype(np.uint8)[..., None], 3, axis=2)

    saved = dict(vars(state))
    state.sam_model = SAMModel(str(sam_checkpoint), model_type="vit_b", device="cpu")
    state.segmenter = ParticleSegmenter(state.sam_model)
    state.current_image = frame
    state.cropped_image = frame
    state.analysable_region = None
    state.modality = mod.Modality(kind=mod.TEM)
    state.particle_choice = "auto"
    state.scale_info = {"conversion": SCALE_NM_PER_PX}

    _viz, status, _choices = segment_with_sam(progress=no_progress)
    *_, analysis_status, _results, _stats = select_mask_and_analyze("Option 1",
                                                                   progress=no_progress)
    yield state, status, analysis_status
    vars(state).clear()
    vars(state).update(saved)


def test_sam_offers_candidates(segmented):
    state, status, _ = segmented
    assert status.startswith("✅"), status
    assert state.candidates


def test_the_first_candidate_finds_every_disk_at_its_true_size(segmented):
    state, _, analysis_status = segmented
    assert analysis_status.startswith("✅"), analysis_status
    measurements = state.analyzer.get_measurements(in_nm=True)
    assert measurements["num_particles"] == len(DISKS)
    assert measurements["unit"] == "nm"

    found = sorted(measurements["diameters"])
    truth = sorted(2 * r * SCALE_NM_PER_PX for _, _, r in DISKS)
    assert found == pytest.approx(truth, rel=0.08)
