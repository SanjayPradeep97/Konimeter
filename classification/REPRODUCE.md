# Reproducing every number in the manuscript

One command, from a checkout of this repository plus the dataset:

```bat
conda activate cnt-vfm
set CNT_BASE=<directory containing "NIOSH Dataset">
reproduce.bat                       :: at the repository root
```

On Linux or macOS, `export CNT_BASE=...` and `./reproduce.sh` do the same.

It refuses to continue past any failing step, and the last step fails loudly
if the results files disagree with the manuscript's numbers.

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
