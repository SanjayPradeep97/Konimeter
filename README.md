# Quantification and Classification of Carbon Nanotubes in Electron Micrographs using Vision Foundation Models

[![tests](https://github.com/SanjayPradeep97/Konimeter/actions/workflows/tests.yml/badge.svg?branch=publication)](https://github.com/SanjayPradeep97/Konimeter/actions/workflows/tests.yml?query=branch%3Apublication)

Code, results and a demo for the paper of that title (Pradeep, Wang, Dahm,
Eldredge & Tsai; *Scientific Reports*, under review).

**This branch is a short demo of the classification.** Open one of 60 TEM
micrographs from the paper's held-out test set: SAM outlines the particle, and
the paper's classifier labels it Fiber, Cluster, Matrix or MatrixSurface, with
the expert's label beside it. If the outline is wrong, click to correct it.

> **The full segmentation app** described in the paper, with scale-bar reading
> and particle measurement, is **Konimeter**, on this repository's
> [`main`](https://github.com/SanjayPradeep97/Konimeter) branch.

## Run the demo on Windows or macOS

**1. Install Miniconda**, if you do not have conda already:
<https://www.anaconda.com/docs/getting-started/miniconda/install>.

**2. Download this branch.** Download
[the ZIP](https://github.com/SanjayPradeep97/Konimeter/archive/refs/heads/publication.zip)
(under 100 MB) and unzip it, or clone it:

```
git clone -b publication https://github.com/SanjayPradeep97/Konimeter.git
```

**3. Create the environment** (once; 10–20 minutes, a few GB). Open
*Anaconda Prompt* on Windows or *Terminal* on macOS and go to the folder
(`Konimeter-publication` if unzipped, `Konimeter` if cloned). Then, on macOS
or on Windows without an NVIDIA GPU:

```
conda env create -f environment-cpu.yml
```

or, on Windows with an NVIDIA GPU:

```
conda env create -f environment.yml
```

**4. Start the demo**, in the same window:

```
conda activate cnt-vfm
python demo/demo_app.py
```

It opens in your browser at <http://127.0.0.1:7861>. On Windows you can
instead double-click `demo\run_demo.bat`.

**5. Use it.**

1. The first time, press **Download SAM ViT-B** (358 MB, kept in
   `sam_weights/`). Opening the first micrograph also fetches DINOv2's weights
   (330 MB).
2. Choose a micrograph, or press **Next**. SAM outlines the particle and the
   prediction appears with its class probabilities and the expert's label.
3. If the outline is wrong, click on the image: with **Particle** selected,
   on the particle; with **Background**, on what to leave out. **Undo** and
   **Reset** step back.

Without a GPU each click takes a few seconds. More about the demo, and how
well SAM's first outline does, is in [`demo/README.md`](demo/README.md).

## Checking the paper's numbers

```
conda activate cnt-vfm
python classification/verify_manuscript.py
```

re-derives every headline number from the shipped results and compares it with
the manuscript, in seconds, with no images needed. The full reproduction from
the raw images, the data, and where each number, table and figure comes from
are in [`classification/REPRODUCE.md`](classification/REPRODUCE.md).

## What is here

| folder | contents |
|---|---|
| `demo/` | the demo: 60 held-out test micrographs, their expert masks, the classifier |
| `classification/` | evaluation protocol, feature extraction, classifiers, statistics, figures and tables |
| `baseline/` | the Luo et al. (2021) re-implementation and its results |
| `results/` | the run the paper reports, with every per-image prediction |
| `splits/` | the 1,785-image train / validation / test partition |
| `tests/` | the demo's tests |

## Citation and licence

See [`CITATION.cff`](CITATION.cff). The images are from the NIOSH dataset on
Harvard Dataverse, <https://doi.org/10.7910/DVN/5O0SF7>. The code is MIT
([`LICENSE`](LICENSE)); the micrographs and masks under `demo/` are CC BY-NC
4.0, non-commercial use only ([`demo/ATTRIBUTION.md`](demo/ATTRIBUTION.md)).
