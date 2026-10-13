# The segmentation tool (Konimeter)

Segmentation and measurement of particles in electron micrographs, built on
Meta's Segment Anything Model, and developed further as
[Konimeter](https://github.com/SanjayPradeep97/Konimeter) on the main branch.
It is an application that runs in the browser, for working through images one at
a time: SAM proposes the particles and the analyst checks and corrects them. It
is built on the `sem_particle_analysis` library.

This is the tool that produced Table 1 of the paper and every mask in the
classification dataset.

## Install

From the repository root, with [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
and Git installed (Python 3.10 to 3.12 work):

```bash
conda create -n konimeter python=3.11
conda activate konimeter

# PyTorch first. NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# or Apple Silicon, or no GPU:
pip install torch torchvision

pip install -e .
```

PyTorch goes in before the package: EasyOCR depends on it, and if it is not
already there pip fetches the default build, which on Windows is CPU-only. The
environment is kept separate from the classification half because EasyOCR pins
its own PyTorch.

The first time the app starts, its Setup tab offers to download the SAM model
(ViT-B, 358 MB) into `sam_weights/`, where it looks on every start. The paper
used ViT-H (2.4 GB); `python download_sam_weights.py --model vit_h` adds it to
the app's list. ViT-B is several times faster and the default. With no `--model` the script asks. A GPU
segments an image in one to three seconds, a CPU in ten to thirty.

## Run the application

```bash
konimeter                        # in the activated environment
python -m sem_analysis_app       # the same
run_app.bat                      # Windows, activates the conda env first
```

It serves `http://127.0.0.1:7860`. `--port` and `--host` change where it
listens. The app keeps a single process-wide state object, so it is built for
one analyst at a time; two browser tabs pointed at the same server share one
session.

The tabs, in the order they are used:

| Tab | What happens there |
| --- | --- |
| ⚙️ Setup | Load the SAM model (the first time, download it with the button there), say whether the images are from an SEM or a TEM, then paste the path of a folder of images. Results go to `analysis_results.csv` in that folder, and opening the folder again carries on with it |
| 🖼️ Gallery | Pick an image. Ticked images already have saved results |
| 📏 Scale & Frame | Establish the scale (below), and check what the app decided about the instrument and the analysable area |
| 🔬 Segment & Refine | Segment with SAM, choose among its candidate masks, then correct them: Remove, Add, Merge, or Redraw a particle with include and exclude clicks. Every click and every applied change can be undone. *Particles are* (darker / brighter than the background) sets which candidates come first |
| 💾 Results | The measurements per image, and the whole folder's results: delete rows, remove duplicates, export |
| 📊 Plots | Size distributions across everything saved, on a log or linear size axis. On log, with the geometric mean and geometric standard deviation |

The masks of the classification dataset were made on the Segment & Refine tab.

Saving a frame also keeps it, with its corrected mask, in `konimeter_dataset/`
beside the results file: `images/<stem>.png`, `masks/<stem>_mask.png` in the
same 0/255 format as the paper's masks, a 16-bit instance-label image whose
label *i* is the *i*-th particle of the frame's results row, the analysable-area
mask, and an `index.csv` recording where each frame came from. Analysing a
folder therefore also builds training data for the classifier. A switch on the
Setup tab turns it off.

Module layout:

```
sem_analysis_app/
    __main__.py       entry point (python -m sem_analysis_app)
    ui.py             tab layout and event wiring
    state.py          shared application state
    static/           the scale box and magnifier, client-side
    callbacks/        event handlers, one module per tab
    visualization.py  overlays and figures
```

## Establishing scale

Scale gets its own tab, because every measurement is a pixel count multiplied by
this one number. Three tiers, tried in order of trustworthiness:

| Tier | Method | When it applies |
| --- | --- | --- |
| 1 | Pixel size read from the file's own metadata | Automatic on load. Exact, nothing to check. Read from FEI/Thermo, Zeiss, Hitachi and TESCAN TIFFs, AMT TEM cameras, and the `.txt` a JEOL SEM saves beside each image. A TIFF's resolution tag is ignored: programs fill it with a print setting |
| 2 | Read the printed scale bar inside a box you draw | Runs automatically first; drag the box if it got it wrong |
| 3 | Click each end of the bar and type its length: the outer edge of a solid bar, the middle of each end tick of a \|—\| bar | When the bar is unreadable. A magnifier follows the cursor so you land exactly |

Tier 2's box is manipulated directly on the image: drag anywhere to draw one,
drag a corner to resize, drag the middle to move. Tier 3 shows a 7x loupe with a
crosshair marking precisely where a click will land.

A tier-2 reading is marked unconfirmed until you press **Confirm scale**; tier 1
and tier 3 are trusted outright, since neither involves a machine reading a
glyph. Tier 2 finds the label first and measures the bar beside it, in the
label's colour, between the centres of its end ticks where it has them. The value
it read is shown beside **Confirm scale**: if OCR misread it, type the right one
and the measured bar is kept; the same misreading of a bar the same length is
corrected the same way from then on, and every scale confirmed lets later
images at that magnification through without asking. The result is shown as "X nm/px · how it was obtained", and it is what the
Segment & Refine tab and the results CSV use. Nothing downstream re-detects scale or
overrides what you set here.

Calibration needs only OCR, so you can sort out scale before loading a SAM
checkpoint.

## SEM and TEM in one workflow

On the Setup tab, say which **Microscope** the folder came from. It tells the app
which way round the particles are: brighter than the background in SEM, darker in
TEM bright field. That matters most for a large particle, where the particle and
the background around it are equally plausible outlines.

It is filled in when the files say: FEI, Zeiss, Hitachi and TESCAN TIFFs, JEOL
TEM files (by the tags JEOL's software writes), the `.txt` a JEOL SEM saves beside
each image, and AMT TEM cameras; if the choice and the files disagree, the app
says so. **Particles are** on the Segment & Refine tab overrides it for an unusual
image. The other side's outlines are always still offered, after. The demo images
in `demo/images/` carry no metadata: choose TEM for them.

An information strip (databar) below the image, as most SEMs write, is found and
cut off, including Zeiss's, which sits a few rows above the bottom edge; it must
begin with a straight, sharp edge, so shading or a grid bar at the bottom of a
frame is never taken for one. A scale bar burned into the image itself, as on
most TEMs, is left out of measurement instead: just the bar and its label's line.

`tests/test_demo_benchmark.py` scores SAM's first suggestion against the expert
masks of the 60 demo images (overlap of at least half counts as right): 48 right
with TEM chosen and about 40 with no microscope chosen, with the right outline
among the three offered for 54 and 52.

**Analysable region.** Micrographs routinely contain large areas that are not
specimen: the black corners left by a circular aperture, a specimen grid bar
blocking the beam, a burned-in scale bar. Every one of them out-contrasts the
particles, so they are excluded before anything is measured. Left in, they are
what gets measured. On real frames these produced five "particles" of 119 µm
that were the corner wedges, and 221 spurious particles that were a grid bar.
The share excluded is shown on the Scale & Frame tab.

## Tests

```bash
pytest                      # everything, including a run of SAM itself
pytest -m "not slow"        # skip tests needing model weights or OCR
```

The fast suite takes under a minute and needs neither SAM weights nor EasyOCR;
tests that do are marked `slow` and skip themselves with a reason when their
dependency is missing. Measurements are checked against shapes of exactly known
size. `tests/synthetic.py` builds micrographs whose scale bar length, printed
label and particle mask are all ground truth, so a regression shows up as a
wrong number rather than merely a changed one.
