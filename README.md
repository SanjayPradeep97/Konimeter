# SAM-SEM Segmentation Toolkit

> **AI-powered segmentation and analysis of particles in electron microscopy images using Meta's Segment Anything Model (SAM)**

This toolkit provides both a **Python package** for programmatic access and a **web-based GUI** for interactive analysis.

---

## 🚀 Quick Start

### 1. Setup Environment

You need [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
and [Git](https://git-scm.com/downloads) — the SAM package installs straight from
its GitHub repository.

```bash
git clone https://github.com/SanjayPradeep97/SAM-SEM-Segmentation.git
cd SAM-SEM-Segmentation

conda create -n SEM_analysis python=3.11
conda activate SEM_analysis
```

Install PyTorch **before** the package. EasyOCR depends on it, and if it is not
already there pip fetches the default build — which on Windows is CPU-only, and
SAM then runs ten times slower without any error to say why.

```bash
# NVIDIA GPU (RTX 20, 30, 40 and 50 series, with a current driver):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon, or no GPU:
pip install torch torchvision
```

Then the package and everything else:

```bash
pip install -e .
```

### 2. Download SAM Weights

```bash
python download_sam_weights.py
```

Choose **3 (ViT-B, 358 MB)** unless you have a reason not to: both apps load it
by default, and it is fast enough to use interactively on a laptop. ViT-H (2.4 GB)
segments slightly better and is picked from the dropdown when present. Weights go
to `sam_weights/`.

### 3. Run the Apps

There are two, for two stages of the work:

| App | For | Windows | Any platform | Opens at |
| --- | --- | --- | --- | --- |
| Analysis | Segmenting images one at a time, from scratch | `run_app.bat` | `python -m sem_analysis_app` | http://127.0.0.1:7860 |
| Review | Correcting and signing off a folder a batch run pre-analysed | `run_review_app.bat` | `python -m sem_review_app` | http://127.0.0.1:7870 |

The `.bat` launchers find conda wherever it is installed and activate
`SEM_analysis`; set `SEM_ENV` first to use another environment. Either app takes
`--port` and `--host`; the review app also takes `--folder` to open a folder on
startup, and moves to the next free port if 7870 is taken.

On Windows, outside the launchers, set `KMP_DUPLICATE_LIB_OK=TRUE` first —
PyTorch and NumPy each ship an OpenMP runtime and otherwise refuse to share a
process.

### 4. Establishing Scale

Scale gets its own tab, because every measurement is a pixel count multiplied by
this one number. Three tiers, tried in order of trustworthiness:

| Tier | Method | When it applies |
| --- | --- | --- |
| 1 | Pixel size read from the file's own metadata | Automatic on load. Exact — nothing to check |
| 2 | Read the printed scale bar inside a box you draw | Runs automatically first; drag the box if it got it wrong |
| 3 | Click both ends of the bar and type its length | When the bar is unreadable. A magnifier follows the cursor so you land on the exact pixel |

Tier 2's box is manipulated directly on the image: drag anywhere to draw one,
drag a corner to resize, drag the middle to move. Tier 3 shows a 7x loupe with a
crosshair marking precisely which pixel a click will land on.

A tier-2 reading is marked unconfirmed until you press **Confirm scale**; tier 1
and tier 3 are trusted outright, since neither involves a machine reading a
glyph. The result is shown as "X nm/px · how it was obtained", and it is what the
Processing tab and the results CSV use. Nothing downstream re-detects scale or
overrides what you set here.

Calibration needs only OCR, so you can sort out scale before loading a SAM
checkpoint.

### 5. SEM and TEM in One Workflow

The two need opposite handling, and the app works out which it is looking at from
the file rather than asking you to remember:

| | SEM | TEM |
| --- | --- | --- |
| Instrument read from | FEI / Zeiss / Hitachi / TESCAN tags | JEOL mode tag, else frame shape |
| Databar | detected and trimmed | none — frame kept whole |
| Scale bar | in the databar, which gets cropped | burned into the frame, so its patch is excluded from measurement |
| Particles are | **brighter** than the substrate | **darker** than the support film |

Both decisions are shown on the Scale tab and can be overridden there, or with
`--modality` and `--particles` in batch mode.

**Analysable region.** Micrographs routinely contain large areas that are not
specimen — the black corners left by a circular aperture, a specimen grid bar
blocking the beam, a burned-in scale bar. Every one of them out-contrasts the
particles, so they are excluded before anything is measured. Left in, they are
what gets measured: on real frames these produced five "particles" of 119 µm
that were the corner wedges, and 221 spurious particles that were a grid bar.
The share excluded is shown on the Scale tab and recorded per image in
`run.json`.

**Magnification.** Low-magnification overviews are navigation frames; their
particles are a few pixels across and counting them adds noise. `--max-nm-per-px`
skips them, and skipped frames are recorded as skipped rather than failed.

### 6. Batch Analysis Without the GUI

For a dataset you intend to publish, run the pipeline headless. Every run writes
a `run.json` recording the image hashes, model and checkpoint hash, the scale and
how it was obtained, all parameters, library versions and the git revision — so a
result can be traced back to exactly what produced it.

```bash
sem-analyze data/raw/sample-a -o data/processed/sample-a --clear-edges
```

Outputs: `particles.csv` (one row per particle), `per_image_summary.csv`,
`size_distribution.png/.pdf` at 300 dpi, and `run.json`.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--scale-nm-per-px X` | Fix the scale instead of detecting it per image |
| `--scale-method metadata\|ocr\|auto` | How to establish scale (default `auto`) |
| `--clear-edges` | Drop particles touching the frame; they are only partly imaged |
| `--min-size N` | Ignore particles below N pixels (default 30) |
| `--crop-percent P` | Override databar removal (default: measure it) |
| `--model-type vit_b` | Faster, lower quality than the default `vit_h` |
| `--modality SEM\|TEM` | Force the instrument kind (default: read it from the file) |
| `--particles bright\|dark` | Force particle polarity (default: follow the modality) |
| `--max-nm-per-px X` | Skip frames coarser than this — magnifications too low to resolve particles |
| `--min-nm-per-px X` | Skip frames finer than this |

Note that batch mode uses the automatic pipeline only — no interactive
refinement — and picks its mask candidate by heuristic. The chosen mask and the
rejected candidates are recorded in `run.json`.

### 7. Reviewing a Pre-analysed Folder

Automatic segmentation gets most frames nearly right and a few badly wrong, so
for counts you intend to publish it is a first draft. The review app opens a
folder of masks somebody (or something) has already made and gives you the same
Remove / Add / Merge / Redraw tools to correct them, one frame at a time.

It expects this layout:

```
C1_analysis/
├── raw/                    one PNG per frame
├── mask/                   one black-and-white PNG per frame, same name
├── overlay/                optional; the thumbnails shown in the Frames tab
└── analysis_results.csv    optional; each frame's scale (file_name, nm_per_px,
                            scale_method). Without it, set the scale in the app.
```

Point it at the folder with **Open folder** or `--folder`. Each frame opens with
its scale from `analysis_results.csv`, which the 📏 Scale panel lets you check
against the bar and confirm, or replace by clicking the bar's two ends.

What it writes, beside the originals rather than over them:

| File | Contents |
| --- | --- |
| `reviewed_results.csv` | One row per signed-off frame, same columns as the analysis app's results; saving a frame again replaces its row |
| `mask_reviewed/` | Your corrected mask for every frame you changed |
| `overlay_reviewed/` | The frame with your corrected mask drawn over it |

Corrections are written when you leave a frame, not only when you save it, so
closing the app — or losing power — does not lose them. Reopening a frame shows
your version; **Reload from analysis** puts the original back.

### 8. Use as Python Package

```python
from sem_particle_analysis import SAMModel, ParticleAnalyzer

# Your code here
```

### 9. Run the Tests

```bash
pytest                      # everything
pytest -m "not slow"        # skip tests needing model weights or OCR
```

The fast suite takes a few seconds and needs neither SAM weights nor EasyOCR;
tests that do are marked `slow` and skip themselves with a reason when their
dependency is missing. Measurements are checked against shapes of exactly known
size — `tests/synthetic.py` builds micrographs whose scale bar length, printed
label and particle mask are all ground truth — so a regression shows up as a
wrong number rather than merely a changed one.

### Optional dependencies

Only two parts of the toolkit need the heavy optional stack:

| Dependency | Needed for | Without it |
| --- | --- | --- |
| `easyocr` | Reading a printed scale bar (tier 2) | Scale from metadata or by hand still works; OCR raises `OCRUnavailableError` explaining the fix |
| `gradio` | The web app | The library and `sem-analyze` are unaffected |

Importing `sem_particle_analysis` pulls in neither. EasyOCR's models load on
first OCR use rather than when a `ScaleDetector` is constructed, so a
metadata-only run never pays for them.

---

## 📦 What's Included

### 1. **Gradio Web Application** (`sem_analysis_app/`)

A beautiful, production-ready web interface with:

- **🤖 AI-Powered Segmentation**: Automatic particle detection using SAM
- **📏 Auto Scale Detection**: OCR-based scale bar recognition and calibration
- **✏️ Interactive Refinement**: Add, delete, merge particles; point-based refinement with live preview
- **📊 Real-time Analysis**: Particle measurements with histograms and statistics
- **💾 Batch Processing**: Process multiple images with session tracking
- **📈 Results Management**: CSV export, duplicate removal, row deletion
- **↩️ Undo/Redo**: Click-level undo for refinement operations
- **🎯 Advanced Features**: Edge particle removal, particle number toggle, size filtering

Module layout:

```
sem_analysis_app/
├── __main__.py       entry point (python -m sem_analysis_app)
├── ui.py             tab layout and event wiring
├── state.py          shared application state
├── callbacks/        event handlers, one module per tab
└── visualization.py  overlays and figures
```

The app keeps a single process-wide state object, so it is built for one analyst
at a time; two browser tabs pointed at the same server share one session.

See [`sem_analysis_app/README.md`](sem_analysis_app/README.md) for detailed usage instructions.

### 2. **Review App** (`sem_review_app/`)

A second interface for the correction pass over a pre-analysed folder — see
[Reviewing a Pre-analysed Folder](#7-reviewing-a-pre-analysed-folder). It reuses
the analysis app's refinement tools rather than reimplementing them, so a fix in
one is a fix in both.

### 3. **Python Package** (`sem_particle_analysis/`)

A clean, modular Python library for programmatic access:

- Scale detection and image preprocessing
- SAM-based particle segmentation
- Particle analysis and measurements
- Results export to CSV
- Interactive Jupyter notebook widgets (legacy)

See [`sem_particle_analysis/README.md`](sem_particle_analysis/README.md) for API documentation.

---

## 🎯 Key Features

### Automatic Scale Detection
- OCR-based scale bar detection using EasyOCR
- Support for both horizontal and vertical scale bars
- Manual override option for non-standard scales

### Advanced Particle Refinement
- **Delete Mode**: Click particles to remove false positives
- **Add Mode**: Click to add missed particles
- **Merge Mode**: Combine touching particles
- **Point Refine Mode**: Iterative refinement with positive/negative points
- Undo individual clicks before applying changes
- Real-time visualization with live previews

### Comprehensive Analysis
- Particle count and size distribution
- Area measurements (pixels and nm²)
- Equivalent diameter calculations
- Summary statistics (mean, median, std, min, max)
- Aggregate statistics across all images in session

### Results Management
- Auto-save to CSV after each image
- Session-wide tracking and export
- Duplicate detection and removal
- Individual row deletion
- State persistence across sessions

---

## 💻 System Requirements

- **Python**: 3.11 or higher
- **GPU**: Optional but recommended
  - NVIDIA GPU with CUDA support (RTX 5080, 4090, 3090, etc.)
  - Apple Silicon (M1/M2/M3/M4) with MPS support
  - CPU fallback available (slower)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: ~3GB for SAM model + your images

### Performance Expectations
- **RTX 5080 (CUDA)**: 1-3 seconds per image
- **Apple Silicon (MPS)**: 2-5 seconds per image
- **CPU**: 10-30 seconds per image

---

## 📚 Documentation

- **Web App Guide**: [`sem_analysis_app/README.md`](sem_analysis_app/README.md)
- **Python API**: [`sem_particle_analysis/README.md`](sem_particle_analysis/README.md)

---

## 🔧 Supported Image Formats

- `.tif`, `.tiff`
- `.png`
- `.jpg`, `.jpeg`

---

## 📊 Output Data

Results are exported as CSV files containing:
- Filename
- Particle count
- Individual particle areas (pixels and nm²)
- Equivalent diameters (pixels and nm)
- Easy integration with Excel, Python, R, etc.

---

## 🎓 Citation

This tool uses Meta's Segment Anything Model (SAM):

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 🙏 Acknowledgments

Built with:
- [Segment Anything Model](https://segment-anything.com/) by Meta AI
- [Gradio](https://gradio.app/) for the web interface
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for scale detection
- [scikit-image](https://scikit-image.org/) for image processing
