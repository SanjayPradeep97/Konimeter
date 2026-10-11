# The segmentation tool (Konimeter)

Segmentation and measurement of particles in electron micrographs, built on
Meta's Segment Anything Model, and developed further as
[Konimeter](https://github.com/SanjayPradeep97/Konimeter) on the main branch.
There are two ways in: a Gradio application for working through images one at a
time, and `konimeter-batch` for running a folder headless. Both sit on the `sem_particle_analysis` library, which can also be
imported directly.

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

Then fetch a SAM checkpoint into `sam_weights/`:

```bash
python download_sam_weights.py --model vit_b          # 358 MB, what the app loads first
python download_sam_weights.py --model vit_b vit_h    # also ViT-H, 2.4 GB
```

The paper used ViT-H, which can be picked in the app once downloaded; ViT-B is
several times faster and the default. With no `--model` the script asks. A GPU
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
| ⚙️ Setup | Load a SAM checkpoint, then paste the path of a folder of images. Results go to `analysis_results.csv` in that folder, and opening the folder again carries on with it. A results file elsewhere can be named instead |
| 🖼️ Gallery | Pick an image. Ticked images already have saved results |
| 📏 Scale & Frame | Establish the scale (below), and check what the app decided about the instrument and the analysable area |
| 🔬 Segment & Refine | Segment with SAM, choose among its candidate masks, then correct them: Remove, Add, Merge, or Redraw a particle with include and exclude clicks. Every click and every applied change can be undone |
| 💾 Results | The measurements per image, and the whole folder's results: delete rows, remove duplicates, export |
| 📊 Plots | Size distributions across everything saved |

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
| 1 | Pixel size read from the file's own metadata | Automatic on load. Exact, nothing to check |
| 2 | Read the printed scale bar inside a box you draw | Runs automatically first; drag the box if it got it wrong |
| 3 | Click both ends of the bar and type its length | When the bar is unreadable. A magnifier follows the cursor so you land on the exact pixel |

Tier 2's box is manipulated directly on the image: drag anywhere to draw one,
drag a corner to resize, drag the middle to move. Tier 3 shows a 7x loupe with a
crosshair marking precisely which pixel a click will land on.

A tier-2 reading is marked unconfirmed until you press **Confirm scale**; tier 1
and tier 3 are trusted outright, since neither involves a machine reading a
glyph. The result is shown as "X nm/px · how it was obtained", and it is what the
Segment & Refine tab and the results CSV use. Nothing downstream re-detects scale or
overrides what you set here.

Calibration needs only OCR, so you can sort out scale before loading a SAM
checkpoint.

## SEM and TEM in one workflow

The two need opposite handling, and the app works out which it is looking at from
the file rather than asking you to remember:

| | SEM | TEM |
| --- | --- | --- |
| Instrument read from | FEI / Zeiss / Hitachi / TESCAN tags | JEOL mode tag, else frame shape |
| Databar | detected and trimmed | none, frame kept whole |
| Scale bar | in the databar, which gets cropped | burned into the frame, so its patch is excluded from measurement |
| Particles are | **brighter** than the substrate | **darker** than the support film |

Both decisions are shown on the Scale & Frame tab and can be overridden there, or with
`--modality` and `--particles` in batch mode.

**Analysable region.** Micrographs routinely contain large areas that are not
specimen: the black corners left by a circular aperture, a specimen grid bar
blocking the beam, a burned-in scale bar. Every one of them out-contrasts the
particles, so they are excluded before anything is measured. Left in, they are
what gets measured. On real frames these produced five "particles" of 119 µm
that were the corner wedges, and 221 spurious particles that were a grid bar.
The share excluded is shown on the Scale & Frame tab and recorded per image in
`run.json`.

**Magnification.** Low-magnification overviews are navigation frames; their
particles are a few pixels across and counting them adds noise.
`--max-nm-per-px` skips them, and skipped frames are recorded as skipped rather
than failed.

## Batch analysis without the GUI

For a dataset you intend to publish, run the pipeline headless. Every run writes
a `run.json` recording the image hashes, model and checkpoint hash, the scale and
how it was obtained, all parameters, library versions and the git revision, so a
result can be traced back to exactly what produced it.

```bash
konimeter-batch path/to/images -o path/to/output --clear-edges
```

`sem-analyze`, the command's earlier name, still works.

Outputs: `particles.csv` (one row per particle), `per_image_summary.csv`,
`size_distribution.png/.pdf` at 300 dpi, and `run.json`. Input may be `.tif`,
`.tiff`, `.png`, `.jpg` or `.jpeg`; prefer original TIFFs, because the scale
detector can read pixel size straight from vendor metadata, which is exact.

| Flag | Effect |
| --- | --- |
| `-o DIR` | Where the outputs go (default `analysis_output/`) |
| `--checkpoint PATH` | Use this weights file instead of looking in `sam_weights/` |
| `--model-type vit_b\|vit_l\|vit_h` | The SAM architecture. Default: read from the checkpoint's name, else the ViT-B weights in `sam_weights/`, as in the app |
| `--device cpu\|mps\|cuda` | Force a device (default: the GPU if there is one) |
| `--scale-nm-per-px X` | Fix the scale instead of detecting it per image |
| `--scale-method metadata\|ocr\|auto` | How to establish scale (default `auto`) |
| `--clear-edges` | Drop particles touching the frame; they are only partly imaged |
| `--min-size N` | Ignore particles below N pixels (default 30) |
| `--border-buffer N` | Clear an N-pixel strip along the frame edge before measuring (default 4) |
| `--crop-percent P` | Override databar removal (default: measure it) |
| `--modality SEM\|TEM` | Force the instrument kind (default: read it from the file) |
| `--particles bright\|dark` | Force particle polarity (default: follow the modality) |
| `--max-nm-per-px X` | Skip frames coarser than this, magnifications too low to resolve particles |
| `--min-nm-per-px X` | Skip frames finer than this |
| `--no-plots`, `--no-checkpoint-hash` | Skip the plots; skip hashing the weights into `run.json` |

When some frames had no readable scale and were measured in pixels, there is
one size distribution per unit (`size_distribution_nm.*`,
`size_distribution_pixels.*`) rather than one that mixes the two.

Batch mode uses the automatic pipeline only, with no interactive refinement, and
picks its mask candidate by heuristic. The chosen mask and the rejected
candidates are recorded in `run.json`.

## Use as a library

```python
from sem_particle_analysis import SAMModel, ScaleDetector, ParticleSegmenter, ParticleAnalyzer
from sem_particle_analysis.utils import load_image

image = load_image("micrograph.tif")
# Pixel size from the file's metadata, else read off the printed scale bar.
scale = ScaleDetector().detect_scale(image, file_path="micrograph.tif")

segmenter = ParticleSegmenter(SAMModel("sam_weights/sam_vit_b_01ec64.pth", model_type="vit_b"))
segmenter.segment_image(image)
segmenter.select_mask()                                   # highest-scoring candidate

analyzer = ParticleAnalyzer(conversion_factor=scale["conversion"])
n, regions = analyzer.analyze_mask(segmenter.get_binary_mask(invert=True))
print(n, analyzer.get_measurements(in_nm=True))
```

`ParticleAnalyzer` also offers `clear_edge_particles`, `delete_particles` and
`merge_particles` for correcting a mask, and `ParticleSegmenter.refine_with_sam`
takes point prompts. `ResultsManager` accumulates per-image measurements and
writes the CSV.

## What needs EasyOCR and Gradio

`pip install -e .` installs both, but the library imports neither until it is
needed, so it also works in an environment that lacks them:

| Dependency | Needed for | Without it |
| --- | --- | --- |
| `easyocr` | Reading a printed scale bar (tier 2) | Scale from metadata or by hand still works; OCR raises `OCRUnavailableError` explaining the fix |
| `gradio` | The web app | The library and `konimeter-batch` are unaffected |

Importing `sem_particle_analysis` pulls in neither. EasyOCR's models load on
first OCR use rather than when a `ScaleDetector` is constructed, so a
metadata-only run never pays for them.

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
