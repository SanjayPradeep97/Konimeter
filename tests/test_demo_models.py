"""
The demo with its real models, on all sixty shipped images: the numbers
demo/README.md reports.

Needs the SAM ViT-B checkpoint (python download_sam_weights.py --model vit_b),
DINOv2's weights (fetched by timm on first use) and the cnt-vfm environment.
A few minutes on a laptop.
"""
import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.slow

DEMO = Path(__file__).resolve().parents[1] / "demo"


@pytest.fixture(scope="module")
def models(sam_checkpoint):
    pytest.importorskip("timm")
    pytest.importorskip("segment_anything")
    from classify import CNTClassifier
    from sam_model import SAMModel

    return SAMModel(sam_checkpoint), CNTClassifier(verbose=False)


def sixty():
    for row in csv.DictReader(open(DEMO / "demo_images.csv")):
        stem = Path(row["filename"]).stem
        gray = np.asarray(Image.open(DEMO / "images" / row["category"] / f"{stem}.png")
                          .convert("L"))
        mask = np.asarray(Image.open(DEMO / "masks" / f"{stem}_mask.png").convert("L")) > 127
        yield stem, gray, mask, row["category"]


def test_every_expert_mask_is_classified_correctly(models):
    _sam, clf = models
    wrong = [stem for stem, gray, mask, truth in sixty() if clf.predict(gray, mask)[0] != truth]
    assert wrong == []


def test_the_first_guess_alone_classifies_most_of_them(models):
    # 54 of 60 when this was written: the misses are faint fibres and frames
    # where another object is the darkest thing in view. Clicks fix them.
    from prompting import auto_point, choose_mask

    sam, clf = models
    wrong = []
    for stem, gray, _mask, truth in sixty():
        sam.set_image(np.stack([gray] * 3, -1))
        masks, scores, _ = sam.predict(np.array([auto_point(gray)], float), np.array([1]))
        chosen = choose_mask(masks, scores, gray)
        if not chosen.any() or clf.predict(gray, chosen)[0] != truth:
            wrong.append(stem)
    print(f"first guess: {60 - len(wrong)}/60 correct; wrong: {wrong}")
    assert len(wrong) <= 7
