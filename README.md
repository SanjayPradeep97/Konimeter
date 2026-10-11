# SAM-SEM Segmentation

Segment and measure particles in SEM and TEM micrographs with Meta's
[Segment Anything Model](https://segment-anything.com/) (SAM): a web app for
interactive analysis, and a command-line tool for running the same pipeline over
a folder.

For every image you get a particle count and, per particle, the area and the
equivalent circular diameter, in pixels and in nanometres.

---

## Install

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

### SAM weights

```bash
python download_sam_weights.py
```

Choose **3 (ViT-B, 358 MB)** unless you have a reason not to: the app loads it by
default, and it is fast enough to use interactively on a laptop. ViT-H (2.4 GB)
segments slightly better and can be picked from the app's dropdown when present.
Weights go to `sam_weights/`.

The app uses an NVIDIA GPU (CUDA) or Apple Silicon (MPS) when it finds one, and
the CPU otherwise.

---

## Run the app

| | |
| --- | --- |
| Windows | double-click `run_app.bat` |
| Any platform | `python -m sem_analysis_app` |

Then open http://127.0.0.1:7860. `--port` and `--host` change where it listens.

`run_app.bat` finds conda wherever it is installed and activates `SEM_analysis`;
set `SEM_ENV` first to use another environment. On Windows outside the launcher,
set `KMP_DUPLICATE_LIB_OK=TRUE` before starting — PyTorch and NumPy each ship an
OpenMP runtime and otherwise refuse to share a process.

### The tabs, in the order you use them

| Tab | What happens there |
| --- | --- |
| ⚙️ Setup | Load a SAM checkpoint and choose a folder of images. Results go to `analysis_results.csv` in that folder; an existing one can be resumed |
| 🖼️ Gallery | Pick an image. Ticked images already have saved results |
| 📏 Scale & Frame | Establish the scale, and check what the app decided about the image (see below) |
| 🔬 Segment & Refine | Segment with SAM, choose among its candidate masks, then correct them: Remove, Add, Merge, or Redraw a particle with include/exclude clicks. Every click and every applied change can be undone |
| 💾 Results | The measurements per image, and the whole folder's results: delete rows, remove duplicates, export |
| 📊 Plots | Size distributions across everything saved |

The app keeps one process-wide session, so it is built for one analyst at a time;
two browser tabs pointed at the same server share it.

### Scale

Every measurement is a pixel count multiplied by one number, so scale gets its own
tab. Three methods, tried in order of trustworthiness:

| | Method | When |
| --- | --- | --- |
| 1 | Pixel size read from the file's own metadata | Automatic on load. Exact — nothing to check |
| 2 | Read the printed scale bar inside a box | Runs automatically when there is no metadata. Drag to draw the box, a corner to resize it, the middle to move it |
| 3 | Click both ends of the bar and type its length | When the bar cannot be read. A 7x magnifier follows the cursor so you land on the exact pixel |

A reading from the printed bar is marked unconfirmed until you press **Confirm
scale**, and the measured span is drawn on the bar so you can see what was read.
Metadata and hand measurement are trusted outright, since neither involves a
machine reading a glyph. The result is shown as "X nm/px · how it was obtained",
and is what segmentation and the results file use; how the scale was obtained is
recorded beside every row in the `scale_method` column.

Scale needs only OCR, not SAM, so it can be sorted out before loading a checkpoint.

### SEM and TEM

The two need opposite handling, and the app works out which it is looking at from
the file:

| | SEM | TEM |
| --- | --- | --- |
| Instrument read from | FEI / Zeiss / Hitachi / TESCAN tags | JEOL mode tag, else frame shape |
| Information strip (databar) | detected and cropped off | none — frame kept whole |
| Scale bar | in the databar | burned into the frame, so its patch is excluded from measurement |
| Particles are | **brighter** than the substrate | **darker** than the support film |

Both decisions are shown on the Scale & Frame tab and can be overridden there, or
with `--modality` and `--particles` in batch mode.

Micrographs often contain large areas that are not specimen — the black corners
left by a circular aperture, a grid bar blocking the beam, a burned-in scale bar.
Each of them out-contrasts the particles, so they are excluded before anything is
measured. The share excluded is shown on the Scale & Frame tab.

Prefer **original TIFFs straight from the microscope**: once an image has been
converted to PNG or JPEG, or cropped in another tool, its pixel-size metadata is
gone and scale has to come from the printed bar or from a hand measurement.

---

## Batch analysis

The same pipeline without the interface, for a whole folder. Every run writes a
`run.json` recording the image hashes, model and checkpoint hash, the scale and
how it was obtained, all parameters, library versions and the git revision, so a
result can be traced back to exactly what produced it.

```bash
sem-analyze path/to/images -o path/to/output --model-type vit_b --clear-edges
```

Outputs: `particles.csv` (one row per particle), `per_image_summary.csv`,
`size_distribution.png/.pdf` at 300 dpi, and `run.json`.

| Flag | Effect |
| --- | --- |
| `--model-type vit_b\|vit_l\|vit_h` | Which SAM weights to use (default `vit_h`) |
| `--checkpoint PATH` | Use this weights file instead of looking in `sam_weights/` |
| `--scale-nm-per-px X` | Fix the scale instead of establishing it per image |
| `--scale-method metadata\|ocr\|auto` | How to establish scale (default `auto`) |
| `--clear-edges` | Drop particles touching the frame; they are only partly imaged |
| `--min-size N` | Ignore particles below N pixels (default 30) |
| `--crop-percent P` | Override databar removal (default: measure it) |
| `--modality SEM\|TEM` | Force the instrument kind (default: read it from the file) |
| `--particles bright\|dark` | Force particle polarity (default: follow the modality) |
| `--max-nm-per-px X` | Skip frames coarser than this — magnifications too low to resolve particles |
| `--min-nm-per-px X` | Skip frames finer than this |

Batch mode is the automatic pipeline only, with no interactive correction, and
picks its mask candidate by heuristic; the chosen mask and the rejected candidates
are recorded in `run.json`. Check its output before relying on the counts.

---

## Output

`analysis_results.csv` has one row per image:

| Column | Contents |
| --- | --- |
| `file_name` | the image |
| `num_particles` | particle count |
| `nm_per_px` | the scale used |
| `scale_method` | how that scale was obtained, e.g. `metadata`, `box_ocr`, `two_points`; `+unconfirmed` if nobody confirmed it |
| `particle_areas_px`, `particle_areas_nm2` | each particle's area |
| `equiv_diameters_px`, `equiv_diameters_nm` | each particle's equivalent circular diameter |

Supported images: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`.

---

## Tests

```bash
pip install pytest
pytest                      # everything
pytest -m "not slow"        # skip the tests needing SAM weights or EasyOCR
```

Measurements are checked against shapes of exactly known size —
`tests/synthetic.py` builds micrographs whose scale bar length, printed label and
particle mask are all ground truth — so a regression shows up as a wrong number
rather than merely a changed one.

Importing `sem_particle_analysis` pulls in neither EasyOCR nor Gradio: without
EasyOCR, scale from metadata or by hand still works, and the library and
`sem-analyze` do not need Gradio.

---

## Layout

```
sem_analysis_app/        the web app (python -m sem_analysis_app)
sem_particle_analysis/   the library and the sem-analyze command
tests/                   the test suite
download_sam_weights.py  fetches SAM checkpoints into sam_weights/
run_app.bat              Windows launcher
```

---

## Citation

This tool uses Meta's Segment Anything Model:

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

## License

MIT — see [LICENSE](LICENSE).

Built with [Segment Anything](https://segment-anything.com/),
[Gradio](https://gradio.app/), [EasyOCR](https://github.com/JaidedAI/EasyOCR) and
[scikit-image](https://scikit-image.org/).
