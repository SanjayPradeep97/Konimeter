# SAM-SEM Segmentation

[![tests](https://github.com/SanjayPradeep97/SAM-SEM-Segmentation/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/SanjayPradeep97/SAM-SEM-Segmentation/actions/workflows/tests.yml)

Segment and measure particles in SEM and TEM micrographs with Meta's
[Segment Anything Model](https://segment-anything.com/) (SAM): a web app for
interactive analysis, and a command-line tool for running the same pipeline over
a folder.

For every image you get a particle count and, per particle, the area and the
equivalent circular diameter, in pixels and in nanometres.

> **Reviewing the paper?** The code, results and reproduction scripts for
> *Quantification and Classification of Carbon Nanotubes in Electron Micrographs
> using Vision Foundation Models* are on the
> [`publication`](https://github.com/SanjayPradeep97/SAM-SEM-Segmentation/tree/publication)
> branch: `git clone -b publication https://github.com/SanjayPradeep97/SAM-SEM-Segmentation.git`

---

## Install

You need [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
and [Git](https://git-scm.com/downloads) — the SAM package installs straight from
its GitHub repository. Python 3.10 to 3.12 work; the commands below use 3.11.
Run them in a terminal (on Windows, the Anaconda Prompt).

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
python download_sam_weights.py --model vit_b
```

ViT-B (358 MB) is what the app loads by default, and is fast enough to use
interactively on a laptop. ViT-H (2.4 GB) segments slightly better and can be
picked from the app's dropdown once downloaded: `--model vit_b vit_h` fetches
both. With no `--model` the script asks. Weights go to `sam_weights/`; to keep
them somewhere else, set `SAM_WEIGHTS_DIR` to that folder.

The app uses an NVIDIA GPU (CUDA) or Apple Silicon (MPS) when it finds one, and
the CPU otherwise. The Setup tab says which, once the model is loaded.

The first time a scale bar is read, EasyOCR downloads its text models (about
100 MB) to `~/.EasyOCR`; after that, everything runs offline.

---

## Run the app

| | |
| --- | --- |
| Windows | double-click `run_app.bat` |
| Any platform | `python -m sem_analysis_app` |

Then open http://127.0.0.1:7860. `--port` and `--host` change where it listens.
`--share` also serves the app through a public Gradio link; since the app opens
folders by path, anyone with that link can open folders on your computer, so
share it only with people you trust.

`run_app.bat` finds conda in the usual Anaconda, Miniconda and Miniforge install
folders, or the one an Anaconda Prompt was started from, and activates
`SEM_analysis`; set `SEM_ENV` first to use another environment. On Windows outside the launcher,
set `KMP_DUPLICATE_LIB_OK=TRUE` before starting — PyTorch and NumPy each ship an
OpenMP runtime and otherwise refuse to share a process.

### The tabs, in the order you use them

| Tab | What happens there |
| --- | --- |
| ⚙️ Setup | Load a SAM checkpoint, then paste the path of a folder of images. Results go to `analysis_results.csv` in that folder, and opening the folder again carries on with it. A results file elsewhere can be named instead |
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
sem-analyze path/to/images -o path/to/output --clear-edges
```

Outputs: `particles.csv` (one row per particle), `per_image_summary.csv`,
`size_distribution.png/.pdf` at 300 dpi, and `run.json`. When some frames had no
readable scale and were measured in pixels, there is one distribution per unit
(`size_distribution_nm.*`, `size_distribution_pixels.*`) rather than one that
mixes the two.

| Flag | Effect |
| --- | --- |
| `-o DIR` | Where the outputs go (default `analysis_output/`) |
| `--checkpoint PATH` | Use this weights file instead of looking in `sam_weights/` |
| `--model-type vit_b\|vit_l\|vit_h` | The SAM architecture. Default: read from the checkpoint's name, else the ViT-B weights in `sam_weights/`, as in the app |
| `--device cpu\|mps\|cuda` | Force a device (default: the GPU if there is one) |
| `--scale-nm-per-px X` | Fix the scale instead of establishing it per image |
| `--scale-method metadata\|ocr\|auto` | How to establish scale (default `auto`) |
| `--clear-edges` | Drop particles touching the frame; they are only partly imaged |
| `--min-size N` | Ignore particles below N pixels (default 30) |
| `--border-buffer N` | Clear an N-pixel strip along the frame edge before measuring (default 4) |
| `--crop-percent P` | Override databar removal (default: measure it) |
| `--modality SEM\|TEM` | Force the instrument kind (default: read it from the file) |
| `--particles bright\|dark` | Force particle polarity (default: follow the modality) |
| `--max-nm-per-px X` | Skip frames coarser than this — magnifications too low to resolve particles |
| `--min-nm-per-px X` | Skip frames finer than this |
| `--no-plots` | Write the CSVs and `run.json` only |
| `--no-checkpoint-hash` | Don't hash the weights file into `run.json` (saves a few seconds) |

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

Supported images: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, in any letter case,
directly inside the folder (subfolders are not searched).

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| The Setup tab says the model is on `cpu`, but there is an NVIDIA GPU | PyTorch is the CPU-only build. In the environment, run `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| `OMP: Error #15 ... already initialized` on Windows | Start the app with `run_app.bat`, or `set KMP_DUPLICATE_LIB_OK=TRUE` first |
| The app will not start because the port is in use | Another copy is running; close it, or start with `--port 7861` |
| "No images in …" | The images must be directly inside the folder, with one of the extensions above |
| "Cannot write results to …" | The folder is read-only (a network share, say). Open **Save results somewhere else** on the Setup tab and give a writable folder |
| `ModuleNotFoundError` for `sem_particle_analysis` or `sem_analysis_app` | The environment is not active, or `pip install -e .` was not run in it |

Anything else: please [open an issue](https://github.com/SanjayPradeep97/SAM-SEM-Segmentation/issues)
with the message and what you were doing.

---

## Tests

```bash
pip install pytest
pytest                      # everything, including a run of SAM itself
pytest -m "not slow"        # skip the tests that need SAM weights or EasyOCR models
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
run_app.bat              Windows launcher (scripts/activate_env.bat finds conda)
data/                    a place for your own images; nothing in it is tracked
```

---

## Citation

If you use this tool, please cite the paper it was built for (under review;
[`CITATION.cff`](CITATION.cff) has the details, and GitHub's "Cite this
repository" button formats it):

> S. Pradeep, C. Wang, M. M. Dahm, J. D. Eldredge and C. S. J. Tsai.
> Quantification and Classification of Carbon Nanotubes in Electron Micrographs
> using Vision Foundation Models. *Scientific Reports* (under review), 2026.

It builds on Meta's Segment Anything Model:

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
