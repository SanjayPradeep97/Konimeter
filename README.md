# Quantification and Classification of Carbon Nanotubes in Electron Micrographs using Vision Foundation Models

[![tests](https://github.com/SanjayPradeep97/Konimeter/actions/workflows/tests.yml/badge.svg?branch=publication)](https://github.com/SanjayPradeep97/Konimeter/actions/workflows/tests.yml?query=branch%3Apublication)

Code, results and reproduction scripts for the paper of that title (Pradeep,
Wang, Dahm, Eldredge & Tsai; *Scientific Reports*, under review). This branch
(`publication`) is written for someone who has the paper open and wants to
check a number. Every reported value can be recomputed from the files here;
the expensive ones can also be recomputed from the raw images.

The project does two things:

1. **Segmentation and quantification.** An interactive tool built on Meta's
   Segment Anything Model: read the scale bar (OCR or metadata), click on
   particles to get masks, post-process them, and measure them. This is the
   tool used for Table 1 and to produce every mask in the classification
   dataset. It is maintained, with its scale-bar reading and measurement, as
   **Konimeter** on the repository's
   [`main`](https://github.com/SanjayPradeep97/Konimeter) branch. The images on
   this branch carry no scale bar, so the demo below uses only its
   segmentation: SAM, prompted by clicks.
2. **Classification.** A frozen DINOv2 ViT-B/14 whose hypercolumn features are
   pooled only inside the particle's mask, followed by a small MLP, assigns each
   segmented particle to Fiber, Cluster, Matrix or MatrixSurface. The
   evaluation protocol, the eight-encoder benchmark, five fine-tuned baselines,
   the Luo et al. re-implementation and every statistical test are in
   [`classification/`](classification/) and [`baseline/`](baseline/).

A [demo](demo/README.md) joins the two on 60 included micrographs: open one,
SAM makes a first guess at the particle, click to correct it if needed, and the
classifier labels it. `demo/run_demo.bat` (or `.sh`) is the one-command start;
the page downloads the SAM ViT-B checkpoint itself the first time.

## Repository layout

| path | what it holds |
|---|---|
| `classification/` | protocol, feature extraction, probes, fine-tuned baselines, statistics, figure and table scripts, tests |
| `baseline/` | the Luo et al. (2021) VGG-16 + VLAD re-implementation and its results |
| `results/` | the run the paper reports: every configuration's metrics and **per-image predictions**, `analysis.json`, the epoch sweep, regenerated figures |
| `splits/` | `dataset_splits.pkl`, the 1,785-image partition every number uses, the same as a CSV, and what each split contains |
| `demo/` | the segment-and-classify demonstration, with 60 held-out micrographs, their expert masks and the paper's classifier heads |
| `tests/` | the demo's tests |
| `download_sam_weights.py` | fetches the SAM ViT-B checkpoint, for the demo and Figure 7 |
| `reproduce.bat`, `reproduce.sh` | the one-command reproduction (Windows / Linux-macOS) |

## Install

```bash
git clone -b publication https://github.com/SanjayPradeep97/Konimeter.git
cd Konimeter
```

(The repository's default branch, `main`, holds the segmentation tool,
Konimeter, for general use, developed further since the paper was submitted;
everything for the paper is on this branch. Until October 2026 the repository
was called SAM-SEM-Segmentation, and links to that name redirect here.)

One conda environment, `cnt-vfm`, runs everything on this branch:

```bash
conda env create -f environment.yml          # NVIDIA GPU (CUDA 12.8 wheels)
conda env create -f environment-cpu.yml      # no CUDA: macOS, or CPU-only machines
conda activate cnt-vfm
python classification/check_env.py
```

The CUDA file pins the `+cu128` PyTorch build so an RTX 50-series card works
(an unpinned `torch` resolves to a CPU-only wheel); the CPU file takes PyTorch,
torchvision and XGBoost from conda-forge, so that they share one OpenMP runtime
(from PyPI, the two runtimes crash the Luo baseline on macOS). The demo, the
statistics, the tables and the tests run on CPU; re-extracting features and
re-running the benchmark want a GPU.

## Run the demo

```bash
conda activate cnt-vfm
./demo/run_demo.sh               # Linux / macOS
demo\run_demo.bat                # Windows
```

The page opens at http://127.0.0.1:7861. The first time, press **Download SAM
ViT-B** (358 MB, saved in `sam_weights/`); DINOv2's weights (330 MB) follow
automatically. Then choose a micrograph: SAM outlines the particle and the
classifier labels it, with the expert's label beside it. If the outline is
wrong, click on the particle, or on the background to leave it out.
[`demo/README.md`](demo/README.md) has the details and the measured numbers.

## Data

The TEM images come from the NIOSH collection deposited on Harvard Dataverse,
<https://doi.org/10.7910/DVN/5O0SF7> (*Dataset of TEM Images for Carbon
Nanomaterial Classification*, 5,323 files, licence CC BY-NC 4.0). The paper
uses 1,785 images; `splits/dataset_splits.csv` lists which, and in which
partition.

**Coverage of the public deposit.** 1,705 of the 1,785 are in
the Dataverse record under the same file names, pixel for pixel (checked on a
sample of 12 across all four classes). The remaining 80 (64 train, 8
validation, 8 test; listed in `splits/not_on_dataverse.csv`) are from the same
NIOSH collection but are not in the current Dataverse version;
`splits/dataverse_files.txt` is the record's file list as retrieved on
2026-09-07. Those 80 images and all 1,785 masks are distributed with the
paper's data archive (see the paper's Code availability statement), so the
benchmark is reproducible from the archive but not from Dataverse alone.
`python classification/check_data.py` reports what a local data root is
missing.

**Sixty of them are included**, with their expert masks, under
[`demo/images/`](demo/images/) and [`demo/masks/`](demo/masks/), so the
segment-then-classify pipeline can be tried without downloading anything else
(see [`demo/ATTRIBUTION.md`](demo/ATTRIBUTION.md) for the licence terms). All
sixty are in the Dataverse record; `demo/fetch_demo_images.py --download
--verify` fetches the originals and confirms the shipped PNGs are identical.
The full dataset is needed only to re-run the benchmark itself.

Set `CNT_BASE` to a directory laid out as

```
<CNT_BASE>/
    NIOSH Dataset/CNT-Fiber/CNT-Fiber-0001.tif ...       the Dataverse download
    NIOSH Dataset/CNT-Cluster/ ...   CNT-Matrix/ ...   CNT-MatrixSurface/ ...
    NIOSH Dataset/Masks/masks/CNT-Fiber-0001_mask.png ... the segmentation masks
```

Feature caches (`Encoder Benchmark/`, about 1.5 GB) and download caches
(`_cache/`) are created underneath it. Nothing in the code refers to any other
location; `classification/cnt_paths.py` is the single place paths are resolved.

**Masks.** The 1,785 masks were made with the segmentation tool (Konimeter,
on `main`). The 60 that belong to the demo images are in `demo/masks/`; the
full set is distributed with the paper's data archive rather than in this
branch. A mask can be made again with the demo, which makes masks live, or
with Konimeter. The classification pipeline refuses to run if a mask is
missing, and says so.

The SEM images of Table 1 were acquired under a collaboration that does not
permit redistribution.

## The one-command reproduction

```bat
conda activate cnt-vfm
set CNT_BASE=<data root>
reproduce.bat
```

```bash
conda activate cnt-vfm
export CNT_BASE=<data root>
./reproduce.sh                  # Linux / macOS
```

What it does, and how long it takes on an RTX 5080:

| step | what | time |
|---|---|---|
| 0 | environment check; fetch the SAM ViT-B checkpoint for Figure 7 if missing (358 MB, once); assert `encoder_bench.py` and `paper_results.py` agree on every feature-cache name | 20 s |
| 1–2 | extract features for 8 encoders (benchmark geometry) and DINOv2 (manuscript geometry) | ~10 min first time, seconds afterwards |
| 3 | 108 frozen probes, 10 fine-tuned baselines, statistics | ~3 h; **skipped while `results/analysis.json` exists** (`--force` recomputes) |
| 4 | epoch-budget sweep | 3 min |
| 5 | Figures 4, 7, 8 into `results/figures/` | 2 min |
| 6 | LaTeX bodies of Tables 2 and 3, printed | 1 s |
| 7 | `verify_manuscript.py`: every headline number re-derived and checked | 1 s |

Because `results/` ships with the run the paper reports, a first run finishes in
the time feature extraction takes, and step 7 passes against the shipped
files. Delete or `--force` to recompute from the images. Details, including
how to run stage 3 in pieces, are in
[`classification/REPRODUCE.md`](classification/REPRODUCE.md). The Luo
baseline is its own command (`baseline/run_all.py`, about 2 h).

Without a GPU or the images, the statistics still recompute from the shipped
per-image predictions:

```bash
python classification/paper_results.py --stages analyse     # rewrites results/analysis.json
python classification/make_tables.py
python classification/verify_manuscript.py
```

## Where each number, figure and table comes from

| in the paper | produced by | from |
|---|---|---|
| Table 1 (segmentation Dice/IoU, clicks) | the segmentation tool (Konimeter, on `main`), on the validation images of the original submission; not regenerated by `reproduce.bat` (the SEM images cannot be redistributed) | |
| Table 2 (selected configurations) | `classification/make_tables.py` | `results/results_probes.csv` |
| Table 3 (baselines and *p* values) | `classification/make_tables.py` | `results/results_probes.csv`, `results/results_finetune.csv`, `results/analysis.json`, `baseline/results/luo_results_*.json` |
| Figs. 1, 2, 3, 5 (image mosaics, architecture diagram, misclassified examples) | figures from the original submission, assembled by hand or with notebooks that are not part of this branch | |
| Fig. 4 (test confusion matrix) | `classification/figures/make_confusion.py` | `results/per_image_probe.json` |
| Fig. 6 (DINOv2 activation mosaic) | figure from the original submission, produced with notebooks that are not part of this branch; unchanged in revision | |
| Fig. 7 (SAM activation mosaic) | `classification/figures/make_sam_mosaic.py` | four micrographs + masks, SAM ViT-B checkpoint |
| the 1.05 / 0.97 ratios in the Fig. 7 caption | `classification/figures/neck_inversion.py`, run by hand; not part of `reproduce.bat` | the same four micrographs, which are training images, so their masks come from the data archive rather than `demo/masks/` |
| Fig. 8 (t-SNE of the test set) | `classification/figures/make_tsne.py` | DINOv2 manuscript-geometry feature cache |
| Fig. 9 (composite multi-particle demo) | figure from the original submission, produced with the segmentation tool and notebooks that are not part of this branch | |
| headline 89.5 % ± 1.0 % CV, 92.7 % (166/179) test, CI 88.0–95.7 % | `classification/paper_results.py --stages analyse` | `results/analysis.json` → `headline` |
| "108 configurations, 80.8 % to 89.7 %, mean 86.6 % ± 2.0 %" | `verify_manuscript.py` derives them | `results/results_probes.csv` |
| mask effect: 54/54 pairs, +2.48 pp, sign test, *t* = 17.8 | `classification/analyse.py::mask_effect` | `results/analysis.json` → `mask_effect` |
| every *p* value (exact McNemar, Holm over 117 tests) | `classification/analyse.py::compare_all` | `results/analysis.json` → `comparisons` |
| fine-tuned baselines | `paper_results.py --stages finetune` | `results/results_finetune.csv`, `results/per_image_finetune.json` |
| the 4.3–9.0 pp margin over them | `verify_manuscript.py` derives it | `results/analysis.json`, `results/results_finetune.csv` |
| Luo et al. row | `baseline/run_all.py`, selected by `classification/luo_results.py` | `baseline/results/` |
| Supplementary Notes S2 (layer ablation), S3 (MLP sweep), S5 (CLS token) | notebooks from the original submission, not part of this branch and not re-run in revision | |
| Supplementary Table S3 (all 108 configurations) | `results/results_probes.csv` directly | |
| epoch-budget check | `classification/epoch_sweep.py` | `results/epoch_sweep.json` |

`classification/verify_manuscript.py` prints every derived value in the form the
paper prints it; run it and compare.

## What the protocol guarantees

`classification/paper_protocol.py` is the single evaluation routine every method
goes through: frozen probes, fine-tuned networks and the VLAD baseline alike.

* **The test set is scored once, for a configuration fixed in advance.**
  `PRIMARY` is declared as a literal before any result exists (DINOv2, 518 px,
  taps [1,3,6,9,11], mask-guided avg+max pooling, MLP). No table can promote
  another configuration on the strength of a test number; other rows in
  Table 3 show each method's best configuration *by cross-validated accuracy*.
* **Early stopping uses an inner split only.** Each fold model trains on 90 %
  of its training folds and stops on the remaining 10 %; the held-out fold and
  the test set are never a stopping monitor. `_guarded()` enforces this on
  every call; `mutants.py` re-injects seven bugs of this kind and shows that
  `test_protocol.py` catches each one.
* **Scalers are fit on training rows only.**
* **Caches are keyed by split** (and by geometry, pyramid input size, a
  random-weights flag and any checkpoint override), so features of one split
  can never be loaded for another; `check_cache_agreement.py` proves both
  scripts build the same names.
* **Comparisons are made on 1,606 out-of-fold predictions**, not on the
  179-image test set, which cannot resolve differences of a few points; the
  test set is reported once with a Wilson interval.

## Tests

```bash
pytest -m "not slow"                        # the demo, without model weights, in seconds
pytest                                       # + the demo with SAM and the classifier on all 60 images (a few minutes)
cd classification
python test_protocol.py                      # 42 checks on the protocol (needs one feature cache)
python mutants.py                            # re-injects 7 audited bugs; all must be caught (needs the same cache)
python test_yolo_adapter.py                  # no GPU or images needed
python test_cache.py
python check_data.py                         # is every image and mask of the split present under CNT_BASE?
```

The full `pytest` run checks that the shipped classifier labels all 60 expert
masks correctly. The demo also has its own checks: `demo/train_demo_head.py`
refuses to write heads whose test predictions differ from the shipped
per-image predictions, and `demo/fetch_demo_images.py --download --verify`
confirms the shipped PNGs are pixel-identical to the Dataverse originals.

## Citation

See [`CITATION.cff`](CITATION.cff). The segmentation tool builds on Segment
Anything (Kirillov et al., 2023); the classifier on DINOv2 (Oquab et al.,
2023).

The code is MIT-licensed ([`LICENSE`](LICENSE)). The micrographs and masks
under `demo/` are not: they are CC BY-NC 4.0, non-commercial use only, and the
images belong to their depositors ([`demo/ATTRIBUTION.md`](demo/ATTRIBUTION.md)).
