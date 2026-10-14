# Demo: segment a particle, see it classified

A small Gradio page that joins the two halves of the paper end to end. You open
a TEM micrograph, SAM outlines the particle, and the paper's classifier
(frozen DINOv2 ViT-B/14 hypercolumns pooled **inside the mask**, then an MLP)
predicts Fiber / Cluster / Matrix / MatrixSurface. If
the outline is wrong, you click to correct it, and the label follows the mask.
Sixty micrographs and their expert masks ship with the repository.

It is deliberately not the full segmentation tool (that is Konimeter, on the
repository's `main` branch): no scale bars, no measurement, no results file.
Just mask and label.

## Start it

Follow *Run the demo* in the [README](../README.md): create the `cnt-vfm`
environment once, then

```
conda activate cnt-vfm
python demo/demo_app.py
```

or double-click `demo\run_demo.bat` on Windows (`demo/run_demo.sh` on macOS).
The page opens at http://127.0.0.1:7861. The first time, it shows **Download
SAM ViT-B**: the checkpoint (358 MB) is saved in `sam_weights/` and found there
from then on. DINOv2's weights (330 MB) are fetched by `timm` from the Hugging
Face hub the first time a micrograph is opened. Without a GPU, opening a
micrograph takes a few seconds and each click about two.

## Using it

1. **Choose a micrograph**, or press **Next** to step through them. The list
   is in a fixed shuffled order.
2. **SAM makes a first guess.** Before any click, SAM is prompted at the
   darkest large object away from the frame edge (the prompt is not shown),
   and the largest dark one of the three masks it returns is classified
   straight away. The prediction, the four class probabilities and the
   expert's label appear beside the image.
3. **If the outline is wrong, click.** With **Particle** selected, a click on
   the particle replaces the first guess and segments from your click; with
   **Background**, a click marks something the mask should leave out. After
   that, clicks add up: each one re-runs SAM on all of them and re-classifies.
4. **Undo** steps back one click, and **Reset** returns to the first guess.

When a mask is brighter than its surroundings the page says it is probably
support film rather than presenting a label for a patch of background as if
it meant something.

## What is in here

| file | purpose |
|---|---|
| `run_demo.bat`, `run_demo.sh` | one-command start |
| `demo_app.py` | the page |
| `prompting.py` | the first guess (`auto_point`) and which of SAM's three masks is kept (`choose_mask`) |
| `sam_model.py` | SAM ViT-B on the best device available (NVIDIA GPU, Apple silicon, CPU) |
| `classify.py` | `CNTClassifier`: one image + one mask -> label and probabilities, using the same `encoder_bench.build_encoder` / `pool_maps` code as the benchmark |
| `weights/dinov2_probe_heads.pt` | the five fold heads of the pre-specified configuration (40 MB, fp16) with their feature scalers |
| `train_demo_head.py` | regenerates the heads from the feature cache (needs `CNT_BASE`) and refuses to save unless their test predictions equal `results/per_image_probe.json` image for image |
| `images/<class>/*.png` | the 60 micrographs, lossless PNG conversions of the Dataverse TIFFs |
| `masks/*_mask.png` | their expert masks |
| `demo_images.csv`, `select_demo_images.py` | which images, and the rule that chose them |
| `fetch_demo_images.py` | optional: fetch the original TIFFs from Dataverse and verify the PNGs are identical |
| `ATTRIBUTION.md` | licence and credit for the images |

The page is tested in `tests/test_demo.py` (no weights needed) and, with SAM
and the classifier, on all sixty images in `tests/test_demo_models.py`, which
also checks that the shipped classifier labels every expert mask correctly.

## What the classifier is, exactly

`weights/dinov2_probe_heads.pt` holds the **five fold heads** of the
pre-specified configuration, produced by `train_demo_head.py` through the same
`probe_fit` code as the benchmark: each trained on 90 % of its four training
folds and early-stopped on the other 10 %, never on its held-out fold or on the
test set. Averaged as a softmax ensemble they are the `test_ensemble` variant
the paper reports: **92.7 % (166/179)** on the held-out test set. The training
script checks that the ensemble's 179 test predictions are identical to the
ones stored in [`results/per_image_probe.json`](../results/per_image_probe.json)
before it will write the file, so the demo runs the paper's model and not a
look-alike.

## Which of SAM's three masks is kept

SAM returns three candidate masks per prompt, roughly a part, the object and
its surroundings, with a confidence score for each. On TEM micrographs the
confidence score often prefers a single fibre inside a bundle, or a large
patch of support film. Because a CNT particle is darker than the film, the
demo instead keeps the **largest candidate that covers less than 60 % of the
frame and is at least 30 % as dark, relative to its surroundings, as the
darkest candidate**. The alternative, SAM's own highest score, can be chosen
on the page, under *Which of SAM's three masks is kept*, for comparison.

Driving the page programmatically with one positive click per image, placed
at the deepest interior point of the expert mask and then displaced by 8 and
15 px in eight directions to imitate an imprecise click (60 images, 17 clicks
each; the script that drove these runs is not part of this branch):

| mask rule | click exactly inside | 8 px off | 15 px off | median IoU with the expert mask |
|---|---|---|---|---|
| largest dark candidate (default) | **58 / 60** (96.7 %) | 94.2 % | 94.0 % | 0.86 |
| SAM's highest score | 49 / 60 (81.7 %) | 80.6 % | 81.9 % | 0.81 |
| expert mask itself | 60 / 60 | | | 1.00 |

The two single-click misses with an exact click are thin fibres on a busy
background (`CNT-Fiber-0343`, `CNT-Fiber-0358`), where every SAM candidate is
either a fragment or a patch of film; a second click, or a negative click on
the film, fixes them. Moving each click to the darkest pixel nearby was also
tried and made no measurable difference, so clicks are used exactly as made.
When a click lands on the film the page says so (the mask is brighter than its
surroundings) instead of presenting a label for a patch of background as if it
meant something. That sensitivity is the point of the demo: the label is a
function of the mask, and the mask is a function of the clicks.

**With no click at all**, the first guess gives the right class on **54 of the
60** images (`tests/test_demo_models.py` measures it); keeping SAM's
highest-scoring mask instead would give 44. The six it gets wrong
are two faint fibres (`CNT-Fiber-0338`, `CNT-Fiber-0358`), three clusters where
a denser knot or a separate dark particle is prompted instead of the open
network (`CNT-Cluster-0051`, `-0055`, `-0446`), and one frame where the
neighbouring particle is (`CNT-Matrix-0101`). The first guess is a starting
point; the clicks are for the rest.

## Why these sixty images

They are **held-out test images**, chosen by `select_demo_images.py` as the
test images that the largest number of the 118 evaluated configurations
classified correctly (57 of the 60 were classified correctly by every
configuration, the other three by all but one) and that are present in the public Dataverse record. They are
the *unambiguous* cases on purpose: the demo shows the mechanism working, it
does not estimate accuracy. The accuracy estimate is the paper's, on the whole
test set. MatrixSurface is omitted because its particles (fibres protruding from
a large carbonaceous particle) are the hardest to isolate with a single click.
