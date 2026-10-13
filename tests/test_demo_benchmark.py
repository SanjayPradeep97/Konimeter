"""
SAM's first suggestion against expert masks, on the 60 NIOSH demo micrographs.

The demo images and their expert masks ship on the publication branch only, so
this skips anywhere else. It is the check on the candidate ranking's defaults:
the size ceiling, the databar crop and which way round particles are expected.
Each of those once cost real images without any test noticing:

- at a 35% size ceiling, all 17 frames whose particle fills more of the frame
  than that were offered nothing, or the background;
- plain shading at the bottom of four frames was cropped off as a databar;
- treating the expected polarity as a filter left frames with no candidate.

A suggestion counts as right when it overlaps the expert mask by at least half
(intersection over union >= 0.5). The thresholds sit a little below what the
current code scores, so a change that makes things worse fails here.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

DEMO = Path(__file__).resolve().parents[1] / "demo"
IMAGES = sorted((DEMO / "images").glob("*/*.png"))

if not IMAGES:
    pytest.skip("the demo images are on the publication branch", allow_module_level=True)


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


@pytest.fixture(scope="module")
def scores(sam_checkpoint):
    """For each image: IoU of the first and best of three suggestions, and how
    many were offered, with polarity left on auto and with it set to darker."""
    from PIL import Image
    from sem_particle_analysis import ParticleSegmenter, ScaleDetector, SAMModel
    from sem_particle_analysis import modality, region
    from sem_particle_analysis.utils import load_image

    segmenter = ParticleSegmenter(SAMModel(str(sam_checkpoint), model_type="vit_b"))
    detector = ScaleDetector(use_gpu=False)

    results = {"auto": [], "dark": [], "cropped": 0}
    for path in IMAGES:
        truth = np.array(Image.open(DEMO / "masks" / f"{path.stem}_mask.png")
                         .convert("L")) > 127
        image = load_image(str(path))

        # As the Scale & Frame tab prepares a frame.
        databar = detector.detect_databar(image)
        if databar["has_databar"]:
            results["cropped"] += 1
            image = image[: image.shape[0] - databar["databar_height"]]
            truth = truth[: image.shape[0]]
        kind = modality.detect(image)
        analysable, _ = region.analysable_region(image)
        exclude = ~analysable if analysable is not None else None

        masks, _ = segmenter.segment_image(image)
        for label, dark in (("auto", kind.dark_particles), ("dark", True)):
            ranked = segmenter.rank_candidates(image, masks, top_k=3,
                                               dark_features=dark, exclude=exclude)
            overlaps = [iou(c["mask"], truth) for c in ranked]
            results[label].append((overlaps[0] if overlaps else 0.0,
                                   max(overlaps, default=0.0), len(ranked)))
    return results


def right_first(rows):
    return sum(first >= 0.5 for first, _, _ in rows)


def right_among_three(rows):
    return sum(best >= 0.5 for _, best, _ in rows)


def offered_nothing(rows):
    return sum(n == 0 for _, _, n in rows)


def test_no_demo_frame_loses_a_strip_to_a_false_databar(scores):
    assert scores["cropped"] == 0


def test_particles_set_to_darker(scores):
    rows = scores["dark"]
    assert right_first(rows) >= 45, right_first(rows)
    assert right_among_three(rows) >= 51, right_among_three(rows)
    assert offered_nothing(rows) <= 2


def test_particles_left_on_auto(scores):
    # Most demo frames say nothing about their instrument, so auto falls back
    # to whichever side stands out more.
    rows = scores["auto"]
    assert right_first(rows) >= 37, right_first(rows)
    assert right_among_three(rows) >= 49, right_among_three(rows)
    assert offered_nothing(rows) <= 2
