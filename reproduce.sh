#!/usr/bin/env bash
# ============================================================================
#  reproduce.sh -- Linux/macOS twin of reproduce.bat.  Rebuild every number,
#  table and figure in the manuscript from the raw dataset and the split
#  file, then verify that the results files still say what the manuscript
#  says.  See reproduce.bat for the stage-by-stage description.
#
#  Needs:    conda env cnt-vfm active (environment.yml or environment-cpu.yml)
#            CNT_BASE  the data root (README.md, "Data")
#  Optional: CNT_TEX   directory holding main.tex / supplementary.tex
#            --force   recompute stage 3 (~3 h on a GPU) instead of using the
#                      shipped results
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS="$REPO/classification"
SPLITS="$REPO/splits/dataset_splits.pkl"
OUT="$REPO/results"
export PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE

if [ -z "${CNT_BASE:-}" ]; then
  echo 'CNT_BASE is not set. Point it at the directory holding "NIOSH Dataset/" -- see README.md.' >&2
  exit 1
fi
if [ ! -d "$CNT_BASE/NIOSH Dataset" ]; then
  echo "\"$CNT_BASE/NIOSH Dataset\" does not exist. See README.md, \"Data\"." >&2
  exit 1
fi
BENCHD="$CNT_BASE/Encoder Benchmark"

fail() { echo; echo "============ FAILED at the step above ============"; exit 1; }
cd "$CLS"

echo "==== 0. preflight ========================================================="
python -u check_env.py || fail
# Figure 7 runs SAM ViT-B; fetched once (358 MB), skipped when present.
python -u "$REPO/download_sam_weights.py" --model vit_b || fail
python -u encoder_bench.py --stage features --group benchmark --splits "$SPLITS" --out-dir "$BENCHD" --dry-run || fail
python -u check_cache_agreement.py --splits "$SPLITS" || fail

echo "==== 1-2. features (skipped per encoder when the cache exists) ============="
python -u encoder_bench.py --stage features --group benchmark --splits "$SPLITS" --out-dir "$BENCHD" || fail
python -u encoder_bench.py --stage features --encoders dinov2_b14 --target-grid 37 --taps 1,3,6,9,11 --splits "$SPLITS" --out-dir "$BENCHD" || fail

echo "==== 3. probes + fine-tuned baselines + analysis ==========================="
if [ "${1:-}" = "--force" ] && [ -f "$OUT/analysis.json" ]; then
  echo "    --force: moving the shipped results aside to results/_previous/"
  mkdir -p "$OUT/_previous"
  for f in analysis.json results_probes.csv results_finetune.csv per_image_probe.json per_image_finetune.json; do
    mv -f "$OUT/$f" "$OUT/_previous/" 2>/dev/null || true
  done
fi
if [ -f "$OUT/analysis.json" ]; then
  echo "    results/analysis.json exists -- skipping (pass --force to recompute; ~3 h)."
else
  python -u paper_results.py --splits "$SPLITS" --out "$OUT" --stages probes finetune analyse || fail
fi

echo "==== 4. epoch-budget sweep ================================================"
python -u epoch_sweep.py "$SPLITS" "$OUT" || fail

echo "==== 5. figures ==========================================================="
python -u figures/make_confusion.py "$OUT" "$OUT/figures" || fail
python -u figures/make_tsne.py "$SPLITS" "$OUT/figures" || fail
python -u figures/make_sam_mosaic.py "$OUT/figures" || fail

echo "==== 6. tables ============================================================"
python -u make_tables.py "$OUT" || fail

echo "==== 7. verify the manuscript numbers ====================================="
if [ -z "${CNT_TEX:-}" ]; then
  python -u verify_manuscript.py "$OUT" || fail
else
  python -u verify_manuscript.py "$OUT" --tex "$CNT_TEX" || fail
fi

echo
echo "============ REPRODUCED AND VERIFIED ============"
