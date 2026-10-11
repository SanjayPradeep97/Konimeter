"""Held-out test confusion matrix (single panel).

An earlier whole-dataset confusion matrix was computed on data the model
trained on; this one reports the held-out test set only.

Counts are read from the pre-specified configuration's per-image predictions,
so the figure cannot drift from the number reported in the text.

Aesthetics match the earlier confusion_matrix_test.png: seaborn
'Blues', row-normalised, two-decimal annotations, colour scale pinned to [0,1],
white cell separators.

usage: python make_confusion.py [<results dir>] [<output dir>]
       defaults: <repo>/results  and  <repo>/results/figures
"""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # classification/
import cnt_paths

RES = sys.argv[1] if len(sys.argv) > 1 else str(cnt_paths.results_dir())
IMG = sys.argv[2] if len(sys.argv) > 2 else str(cnt_paths.results_dir() / "figures")
os.makedirs(IMG, exist_ok=True)

CLASSES = ["Fiber", "Cluster", "Matrix", "MatrixSurface"]
KEY = "dinov2_b14[g37_L1-3-6-9-11]|masked|avg+max|mlp"   # == paper_protocol.PRIMARY

recs = {e["key"]: e for e in json.load(open(os.path.join(RES, "per_image_probe.json")))}
if KEY not in recs:
    raise SystemExit(f"PRIMARY key {KEY} absent from per_image_probe.json")
e = recs[KEY]
y = np.array(e["y_test"]); p = np.array(e["pred_ensemble"])
cm = np.zeros((4, 4), int)
for t, q in zip(y, p):
    cm[t, q] += 1
cmn = cm / cm.sum(1, keepdims=True)

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(8.2, 7.1), dpi=300)
sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            xticklabels=CLASSES, yticklabels=CLASSES, square=True,
            linewidths=1.5, linecolor="white",
            annot_kws={"size": 15}, cbar_kws={"shrink": 0.98, "pad": 0.02}, ax=ax)
ax.set_xlabel("Predicted", fontsize=17, labelpad=10)
ax.set_ylabel("True", fontsize=17, labelpad=10)
ax.tick_params(axis="x", labelsize=13, length=0, pad=6)
ax.tick_params(axis="y", labelsize=14, length=0, pad=6, rotation=0)
cb = ax.collections[0].colorbar
cb.ax.tick_params(labelsize=13, length=3); cb.outline.set_visible(False)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout()
out = os.path.join(IMG, "confusion_matrix_test.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", out)
print(f"  accuracy {100*np.trace(cm)/cm.sum():.2f}%  ({np.trace(cm)}/{cm.sum()})")
for i, c in enumerate(CLASSES):
    print(f"  {c:14} recall {100*cm[i,i]/cm[i].sum():5.1f}%   n={cm[i].sum()}")
