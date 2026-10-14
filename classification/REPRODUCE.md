# Reproducing every number in the manuscript

One command, from a checkout of this repository plus the dataset:

```bat
conda activate cnt-vfm
set CNT_BASE=<directory containing "NIOSH Dataset">
reproduce.bat                       :: at the repository root
```

On macOS, `export CNT_BASE=...` and `./reproduce.sh` do the same.

It refuses to continue past any failing step, and the last step fails loudly
if the results files disagree with the manuscript's numbers.

## The data

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
[`demo/images/`](../demo/images/) and [`demo/masks/`](../demo/masks/), so the
segment-then-classify pipeline can be tried without downloading anything else
(see [`demo/ATTRIBUTION.md`](../demo/ATTRIBUTION.md) for the licence terms). All
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

## What it does

| step | script | cost (RTX 5080) | output |
|------|--------|------|--------|
| 0 | `check_env.py`, `download_sam_weights.py --model vit_b`, `encoder_bench.py --dry-run`, `check_cache_agreement.py` | 20 s, plus the 358 MB checkpoint once | environment usable; the SAM ViT-B checkpoint step 5 needs is in `sam_weights/`; both scripts agree, byte for byte, on all 40 feature-cache filenames (default and every variant) |
| 1 | `encoder_bench.py --group benchmark` | ~9 min, or seconds if cached | 8 encoders at the matched 32x32 benchmark geometry, `<CNT_BASE>/Encoder Benchmark/feats_*_dataset_splits.npz` |
| 2 | `encoder_bench.py --encoders dinov2_b14 --target-grid 37 --taps 1,3,6,9,11` | 30 s, or seconds if cached | DINOv2 at the manuscript geometry |
| 3 | `paper_results.py --stages probes finetune analyse` | ~3 h; **skipped while `results/analysis.json` exists** | 108 probes, 10 fine-tuned baselines, `analysis.json`, per-image predictions |
| 4 | `epoch_sweep.py` | 3 min | epoch-budget ablation, decided on CV (`results/epoch_sweep.json`) |
| 5 | `figures/make_confusion.py`, `make_tsne.py`, `make_sam_mosaic.py` | 2 min | Figures 4, 7 and 8 into `results/figures/` |
| 6 | `make_tables.py` | 1 s | LaTeX bodies of Tables 2 and 3, printed |
| 7 | `verify_manuscript.py` | 1 s | every headline value re-derived and checked |

The shipped `results/` directory is the run the manuscript reports, so a first
`reproduce.bat` on a machine with the dataset finishes in the time it takes to
extract features. `reproduce.bat --force` moves the shipped results aside and
recomputes stage 3 from scratch.

Stage 3 alone can also be run in pieces:

```bat
cd classification
python paper_results.py --smoke --stages probes finetune analyse     :: ~10 min end-to-end check
python paper_results.py --stages finetune                            :: most of the ~3 h
python paper_results.py --stages probes
python paper_results.py --stages analyse                             :: no GPU, no data root needed
```

A `--smoke` run writes to `results/smoke/`, never over the shipped results.

The Luo et al. baseline is run separately (`baseline/run_all.py`, ~2 h) and
its results are read from `baseline/results/`; stage 6 and 7 use them.

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

## Step 7 without the manuscript source

The `.tex` files are not part of this repository. With `CNT_TEX` set to a
directory containing `main.tex` and `supplementary.tex`, `verify_manuscript.py`
renders every derived value exactly as the paper prints it and asserts the
string is present. Without it, the same derived values are compared with
`manuscript_numbers.json`, a snapshot written by `verify_manuscript.py
--snapshot` at the moment the `.tex` check last passed. Either way the script
prints the full list of derived values, so a reader with the paper open can
compare them line by line.

## The guarantees these scripts enforce

**The test set is scored once, for a configuration fixed in advance.**
`paper_protocol.py` declares `PRIMARY` as a literal before any result exists,
and `_guarded()` wraps every adapter call so no fold model can see the test set,
or its own evaluation fold, as a training or early-stopping set.

**Early stopping only ever sees an inner split.** Each fold model trains on 90 %
of its four training folds and stops on the other 10 %; the fifth fold is
predicted once, out-of-fold, and never influences stopping.

**Scalers are fit on training rows only.** `probe_fit.py` fits the
`StandardScaler` on `train_idx` and nothing else; `test_protocol.py` spies on
`StandardScaler.fit` to prove it.

**Caches are keyed by everything that changes them.** Feature-cache names carry
the encoder, geometry, split-file stem, pyramid input size, a random-weights
flag and any checkpoint override (`encoder_bench.cache_name`). A smoke-test
cache (`ENCBENCH_NO_PRETRAINED=1`) is named `_RANDOMWEIGHTS` and can never be
picked up by a real run. `paper_results.cache_path` builds the same names
independently and `check_cache_agreement.py` asserts they match for every
variant.

**The two scripts cannot disagree about which features they load.** See above;
a mismatch aborts before any hours-long stage starts.

**Selection is on cross-validation, never on the test column.** `make_tables.py`
states every selection rule in one place; the fine-tuned masked-vs-raw choice
and the Luo normalisation variant are both chosen on CV.

**The manuscript cannot drift from the results.** `verify_manuscript.py`
recomputes each headline value and fails if the manuscript (or its snapshot)
disagrees, and it re-derives the confusion matrix from the per-image
predictions rather than trusting the figure.

**Bugs of this kind are caught.** `mutants.py` re-injects seven audited leaks
of the test set or the evaluation fold, one at a time, and shows that
`test_protocol.py` catches each one.

**Comparisons are made on 1,606 out-of-fold predictions,** not on the
179-image test set, which cannot resolve differences of a few points; the test
set is reported once, with a Wilson interval.

## Two geometries, deliberately

| label | config | used for |
|---|---|---|
| `g37_L1-3-6-9-11` | 518x518, taps [1,3,6,9,11], 37x37 tokens, 3,840-D per pooling | **the headline row**, the configuration Methods describes and Supplementary Note S2 justifies |
| `g32_t5` | 32x32 token grid, five evenly spaced taps | the cross-encoder comparison: the same layer *strategy* for every model, so none benefits from a layer set tuned for it |

`PRIMARY` names its geometry, so if the manuscript-geometry features were never
extracted the analysis aborts instead of quietly reporting the benchmark number.

## The `refit_unreliable` flag

Every results row has three test-set variants: `test_ensemble` (mean softmax of
the five fold models; **the number the paper reports**), `test_refit` (one
fresh model on all 1,606 dev images for `median(fold best epoch)` epochs) and
`test_single` (a legacy single model on 90 % of dev).

For four of the ten fine-tuned rows the folds early-stopped within the first
three epochs, because the inner 10 % validation split is small and the loss
bottoms out during learning-rate warm-up. The transferred budget is then 1–3
epochs, the cosine schedule never leaves warm-up, and the refit is not a
trained model: ViT-B/16 masked 60.9 %, ViT-B/16 raw 73.7 %, ConvNeXt-V2 raw
36.3 % (ConvNeXt-V2 masked, also flagged, reached 87.7 %). The fold ensemble is
unaffected because each fold model keeps its own best checkpoint.

Rather than ship those numbers bare, `paper_protocol.refit_unreliable()` flags
any row whose epoch budget is below `REFIT_MIN_BUDGET = 4`; every row in the
shipped CSVs carries the value that function returns, and new runs write it
natively. Eighteen probe rows are flagged by the same rule;
their refits are fine (probes have no warm-up schedule), which is why the flag
means "do not read this value" rather than "this value is wrong". No flagged
value is quoted anywhere in the manuscript.

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
