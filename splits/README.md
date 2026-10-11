# The split every reported number uses

`dataset_splits.pkl` is the 1,785-image partition behind every figure and table.
`dataset_splits.csv` is the same partition as plain text.

| split | n | Fiber | Cluster | Matrix | MatrixSurface |
|---|---|---|---|---|---|
| train | 1,428 | 344 | 347 | 367 | 370 |
| val | 178 | 43 | 43 | 46 | 46 |
| test | 179 | 43 | 44 | 46 | 46 |
| **total** | **1,785** | 430 | 434 | 459 | 462 |
| dev (train+val) | 1,606 | 387 | 390 | 413 | 416 |

Stratified 80/10/10; each class contributes 10.0–10.1 % of its images to test.
The classes are not balanced, 430 to 462 per class, a 7.4 % spread. Fiber is the
smallest class: 412 of its 430 images are among the 442 fiber images in the
Dataverse record, and the other 18 are from the same collection but not in the
record (`not_on_dataverse.csv`).

## How the protocol uses it

`dev`, the 1,606 train and validation images together, is where all model
development happens. It is cut into five stratified folds, every dev image is
predicted exactly once out-of-fold, and accuracy over those 1,606 out-of-fold
predictions is the primary metric.

`test`, 179 images, is held out. It is scored once per reported variant with a
Wilson 95 % interval and is never used for training, early stopping or model
selection.

The `val` rows are not a separate evaluation set under this protocol. The
train/val boundary inside dev is ignored and the folds are redrawn with
`StratifiedKFold(5, shuffle=True, random_state=42)`.

## What is in the pickle

`{train_df, val_df, test_df}`, each a list of dicts with the keys `image_path`,
`mask_path`, `category`, `category_id`, `filename` and `base_name`. The two path
fields hold absolute paths from the machine the split was made on and are
ignored by every script here: paths are rebuilt from the category and filename
by `classification/cnt_paths.py`, which resolves them under `CNT_BASE` as

```
<CNT_BASE>/NIOSH Dataset/CNT-<class>/<name>.tif          images
<CNT_BASE>/NIOSH Dataset/Masks/masks/<name>_mask.png     masks
```

`dataverse_files.txt` is the file list of the public Harvard Dataverse record as
retrieved on 2026-09-07, and `not_on_dataverse.csv` lists the 80 split images
that record does not contain. `python classification/check_data.py` reports what
a local data root is missing.
