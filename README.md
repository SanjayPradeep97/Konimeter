# Konimeter

[![tests](https://github.com/SanjayPradeep97/Konimeter/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/SanjayPradeep97/Konimeter/actions/workflows/tests.yml)

Segment, count and measure particles in SEM and TEM micrographs with Meta's
[Segment Anything Model](https://segment-anything.com/) (SAM), in an app that
runs in your browser: SAM proposes the particles, and you check and correct them.
It has been used for carbon nanotubes, microplastics and wildfire particulate
matter, and works for anything else that can be imaged as particles.

For every image you get a particle count and, per particle, the area and the
equivalent circular diameter, in pixels and in nanometres.

*The name:* a konimeter (Greek *konis*, dust) was the instrument that measured
dust exposure in mines by catching particles on a glass slide, to be counted
under a microscope. This is the same job, done by a model instead of an eye.
The project was called SAM-SEM-Segmentation until October 2026; old links
redirect here.

> **Reviewing the paper?** The code, results and reproduction scripts for
> *Quantification and Classification of Carbon Nanotubes in Electron Micrographs
> using Vision Foundation Models* are on the
> [`publication`](https://github.com/SanjayPradeep97/Konimeter/tree/publication)
> branch: `git clone -b publication https://github.com/SanjayPradeep97/Konimeter.git`

---

## Install

You need [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
and [Git](https://git-scm.com/downloads) — the SAM package installs straight from
its GitHub repository. Python 3.10 to 3.12 work; the commands below use 3.11.
Run them in a terminal (on Windows, the Anaconda Prompt).

```bash
git clone https://github.com/SanjayPradeep97/Konimeter.git
cd Konimeter

conda create -n konimeter python=3.11
conda activate konimeter
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

### The SAM model

The first time you start the app, its Setup tab offers **Download the SAM
model** (ViT-B, 358 MB). It is saved in the `sam_weights/` folder inside your
Konimeter folder, which is where the app looks every time it starts, so you
download it once. ViT-B is fast enough to use interactively on a laptop.

ViT-H (2.4 GB) segments slightly better. To add it to the app's list, run
`python download_sam_weights.py --model vit_h` in the Konimeter folder. To keep
weights somewhere else, set `SAM_WEIGHTS_DIR` to that folder.

The app uses an NVIDIA GPU (CUDA) or Apple Silicon (MPS) when it finds one, and
the CPU otherwise. The Setup tab says which, once the model is loaded.

The first time a scale bar is read, EasyOCR downloads its text models (about
100 MB) to `~/.EasyOCR`; after that, everything runs offline.

---

## Run the app

| | |
| --- | --- |
| Windows | double-click `run_app.bat` |
| Any platform | `konimeter`, in the activated environment |

Then open http://127.0.0.1:7860. `--port` and `--host` change where it listens.
`--share` also serves the app through a public Gradio link; since the app opens
folders by path, anyone with that link can open folders on your computer, so
share it only with people you trust.

`run_app.bat` finds conda in the usual Anaconda, Miniconda and Miniforge install
folders, or the one an Anaconda Prompt was started from, and activates
the `konimeter` environment (or `SEM_analysis`, its name before the rename); set
`SEM_ENV` first to use another environment. On Windows outside the launcher,
set `KMP_DUPLICATE_LIB_OK=TRUE` before starting — PyTorch and NumPy each ship an
OpenMP runtime and otherwise refuse to share a process.

### The tabs, in the order you use them

| Tab | What happens there |
| --- | --- |
| ⚙️ Setup | Load the SAM model (the first time, download it with the button there), then paste the path of a folder of images. Results go to `analysis_results.csv` in that folder, and opening the folder again carries on with it |
| 🖼️ Gallery | Pick an image. Ticked images already have saved results |
| 📏 Scale & Frame | Establish the scale, and check what the app decided about the image (see below) |
| 🔬 Segment & Refine | Segment with SAM, choose among its candidate masks, then correct them: Remove, Add, Merge, or Redraw a particle with include/exclude clicks. Every click and every applied change can be undone. *Particles are* (darker / brighter than the background) sets which candidates come first |
| 💾 Results | The measurements per image, and the whole folder's results: delete rows, remove duplicates, export |
| 📊 Plots | Size distributions across everything saved, on a log or linear size axis. On log, with the geometric mean and geometric standard deviation |

The app keeps one process-wide session, so it is built for one analyst at a time;
two browser tabs pointed at the same server share it.

### Scale

Every measurement is a pixel count multiplied by one number, so scale gets its own
tab. Three methods, tried in order of trustworthiness:

| | Method | When |
| --- | --- | --- |
| 1 | Pixel size read from the file's own metadata | Automatic on load. Exact — nothing to check. The one exception: a pixel size found only in a TIFF's generic resolution tag, usually a print setting, waits for you to confirm it |
| 2 | Read the printed scale bar inside a box | Runs automatically when there is no metadata. Drag to draw the box, a corner to resize it, the middle to move it |
| 3 | Click the outer edge of each end of the bar and type its length | When the bar cannot be read. A 7x magnifier follows the cursor so you land exactly on the edge |

A reading from the printed bar is marked unconfirmed until you press **Confirm
scale**, and the measured span is drawn on the bar so you can see what was read.
Metadata and hand measurement are trusted outright, since neither involves a
machine reading a glyph. The result is shown as "X nm/px · how it was obtained",
and is what segmentation and the results file use; how the scale was obtained is
recorded beside every row in the `scale_method` column.

### Darker or brighter particles

SAM proposes outlines for both what stands out dark and what stands out bright,
and the app has to put the right one first. With small particles that is easy;
with a large one, the particle and the background around it are equally
plausible, and only knowing which way round the image is settles it.

That is what **Particles are** on the Segment & Refine tab says: *darker* than
the background (TEM bright field, the usual case), *brighter* (most SEM images), or
*Auto*. Auto goes by the file: FEI and JEOL files name the instrument, Zeiss,
Hitachi and TESCAN files are SEM, and a square frame with no metadata is taken to
be from a TEM camera. Otherwise Auto puts first whichever side stands out more.
The choice only orders the candidates — the other side is still offered after
them — and it is kept for every image until you change it, so set it once per
folder. The Scale & Frame tab shows what the file said and what is expected.

An information strip (databar) below the image, as most SEMs write, is found and
cut off. A scale bar burned into the image itself, as on most TEMs, is left out
of measurement instead.

Micrographs often contain large areas that are not specimen — the black corners
left by a circular aperture, a grid bar blocking the beam, a burned-in scale bar.
Each of them out-contrasts the particles, so they are excluded before anything is
measured. The share excluded is shown on the Scale & Frame tab.

Prefer **original TIFFs straight from the microscope**: once an image has been
converted to PNG or JPEG, or cropped in another tool, its pixel-size metadata is
gone and scale has to come from the printed bar or from a hand measurement.

---

## Output

`analysis_results.csv` has one row per image:

| Column | Contents |
| --- | --- |
| `file_name` | the image |
| `num_particles` | particle count |
| `nm_per_px` | the scale used |
| `scale_method` | how that scale was obtained: `metadata`, `tiff_resolution`, `box_ocr`, `two_points` or `none`; `+unconfirmed` if nobody confirmed it |
| `particle_areas_px`, `particle_areas_nm2` | each particle's area |
| `equiv_diameters_px`, `equiv_diameters_nm` | each particle's equivalent circular diameter |

Supported images: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, in any letter case,
directly inside the folder (subfolders are not searched).

### Training data

Every frame you save has been segmented by SAM and then checked, and corrected
where needed, by you, so its mask is a label. The app keeps each one in
`konimeter_dataset/`, beside the results file, so analysing a folder also builds
a labelled dataset for training a classifier or fine-tuning the segmentation
model:

| Path | Contents |
| --- | --- |
| `images/<name>.png` | the frame exactly as segmented, with any databar trimmed off |
| `masks/<name>_mask.png` | 8-bit, 255 = particle: the format the paper's classification pipeline reads |
| `labels/<name>_labels.png` | 16-bit, one value per particle; value *i* is the *i*-th particle in the frame's row of `analysis_results.csv` |
| `valid/<name>_valid.png` | 255 where the frame is specimen; 0 where it was left out of analysis (aperture vignette, grid bar, burned-in scale bar), so those pixels count as neither particle nor background |
| `index.csv` | one row per frame: the source file's name and SHA-256, size, scale and how it was set, instrument kind, particle polarity, SAM model, and when it was saved |

Saving a frame again replaces its entry; deleting its row on the Results tab
deletes it. The index records file names, never paths, so the folder can be
shared without revealing where it lived. Nothing leaves your computer; to turn
it off, untick **Also keep each saved image and its mask as training data** on
the Setup tab.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| The Setup tab says the model is on `cpu`, but there is an NVIDIA GPU | PyTorch is the CPU-only build. In the environment, run `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| `OMP: Error #15 ... already initialized` on Windows | Start the app with `run_app.bat`, or `set KMP_DUPLICATE_LIB_OK=TRUE` first |
| The app will not start because the port is in use | Another copy is running; close it, or start with `--port 7861` |
| "No images in …" | The images must be directly inside the folder, with one of the extensions above |
| "Results cannot be saved in …" | The folder is read-only (a network share, say). Copy the images to a folder you can write to, and open that one |
| `konimeter: command not found`, or `ModuleNotFoundError` for `sem_particle_analysis` or `sem_analysis_app` | The environment is not active, or `pip install -e .` was not run in it |

Anything else: please [open an issue](https://github.com/SanjayPradeep97/Konimeter/issues)
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

---

## Layout

```
sem_analysis_app/        the web app (konimeter, or python -m sem_analysis_app)
sem_particle_analysis/   the segmentation and measurement library the app is built on
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
