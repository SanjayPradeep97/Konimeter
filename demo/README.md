# Demo: segment a particle, see it classified

A small Gradio page that joins the two halves of the paper end to end. You click
on a particle in a TEM micrograph, SAM proposes a mask from that click, and the
paper's classifier (frozen DINOv2 ViT-B/14 hypercolumns pooled **inside the
mask**, then the five-fold MLP ensemble) predicts Fiber / Cluster / Matrix /
MatrixSurface. Sixty micrographs and their expert masks ship with the
repository, so nothing has to be downloaded except the SAM checkpoint.

It is deliberately not the full analysis application (that is
[`sem_analysis_app/`](../docs/SEGMENTATION_TOOL.md)): no scale bars, no
measurement, no batch mode. Just click, mask, label.

## Start it

```bat
conda env create -f environment.yml        :: once, at the repo root (environment-cpu.yml without CUDA)
demo\run_demo.bat                          :: Windows
```

```bash
conda env create -f environment.yml        # once, at the repo root; on macOS or
                                           # without an NVIDIA GPU, environment-cpu.yml
./demo/run_demo.sh                         # Linux / macOS
```

The launcher activates the `cnt-vfm` environment, downloads the SAM ViT-B
checkpoint (358 MB) into `sam_weights/` the first time, and opens
http://127.0.0.1:7861. DINOv2 weights (330 MB) are fetched by `timm` from the
Hugging Face hub on first use. Without a GPU each click takes a few seconds.

## Using it

1. Pick a micrograph. The list is in a fixed shuffled order so the class is
   not given away; the true (expert) label is shown after each prediction.
2. Click on a particle. A green dot is a positive point; switch the radio
   button to add a red negative point on something the mask should not
   include. Every click reruns SAM on all points so far, keeps one of its three
   candidate masks (see below), and classifies it.
3. **Show the expert mask** replaces the SAM mask with the mask from the
   paper's dataset for the same image and classifies that. It is the reference
   answer: the shipped ensemble labels all sixty expert masks correctly.
4. *Undo last point* and *reset* do what they say.

The report under the image gives the predicted class, the expert label, the
four class probabilities, the mask size and the timing of each stage.

## What is in here

| file | purpose |
|---|---|
| `run_demo.bat`, `run_demo.sh` | one-command start (env, checkpoint, page) |
| `demo_app.py` | the page |
| `classify.py` | `CNTClassifier`: one image + one mask -> label and probabilities, using the same `encoder_bench.build_encoder` / `pool_maps` code as the benchmark |
| `weights/dinov2_probe_heads.pt` | the five fold heads of the pre-specified configuration (40 MB, fp16) with their feature scalers |
| `train_demo_head.py` | regenerates the heads from the feature cache (needs `CNT_BASE`) and refuses to save unless their test predictions equal `results/per_image_probe.json` image for image |
| `images/<class>/*.png` | the 60 micrographs, lossless PNG conversions of the Dataverse TIFFs |
| `masks/*_mask.png` | their expert masks |
| `demo_images.csv`, `select_demo_images.py` | which images, and the rule that chose them |
| `fetch_demo_images.py` | optional: fetch the original TIFFs from Dataverse and verify the PNGs are identical |
| `ATTRIBUTION.md` | licence and credit for the images |

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
darkest candidate**. The alternative, SAM's own highest score, is available
in the radio button for comparison.

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

## Why these sixty images

They are **held-out test images**, chosen by `select_demo_images.py` as the
test images that the largest number of the 118 evaluated configurations
classified correctly (57 of the 60 were classified correctly by every
configuration, the other three by all but one) and that are present in the public Dataverse record. They are
the *unambiguous* cases on purpose: the demo shows the mechanism working, it
does not estimate accuracy. The accuracy estimate is the paper's, on the whole
test set. MatrixSurface is omitted because its particles (fibres protruding from
a large carbonaceous particle) are the hardest to isolate with a single click.
