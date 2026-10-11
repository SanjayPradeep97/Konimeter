# Data

Put your SEM/TEM micrographs here. Image files in this directory are **not**
tracked by git (see `data/.gitignore`) — only this README and directory
structure are.

## Suggested layout

```
data/
├── raw/                 # untouched micrographs straight off the instrument
│   └── <sample-id>/     # one folder per sample or deposition run
├── processed/           # analysis outputs (CSV + provenance sidecars)
└── figures/             # publication figures
```

Keeping one folder per sample matters because the app opens every image in the
folder you give it, and keeps one `analysis_results.csv` per folder.
Saved frames are also kept, with their masks, in that folder's
`konimeter_dataset/` (see the README, "Training data").

## Formats

`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`.

Prefer **original TIFFs straight from the microscope**. The scale detector can
read pixel size directly from vendor TIFF metadata (FEI/Helios, Zeiss, JEOL,
Hitachi, TESCAN), which is exact. Once an image has been converted to PNG/JPEG
or cropped in another tool, that metadata is gone and scale has to come from
OCR of the scale bar or from manual entry — both of which are less reliable.
