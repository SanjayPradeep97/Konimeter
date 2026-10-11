"""t-SNE of the classifier's representation, HELD-OUT TEST IMAGES ONLY.

An earlier version embedded all images, including those the classifier was
trained on, so the visible separation was partly memorisation.  This version
embeds only the held-out test images.

Aesthetics match the earlier TSNE_evolution.png: palette
saddlebrown / seagreen / steelblue / mediumpurple at alpha 0.8 (recovered by
sampling that PNG), same panel titles, same bold (a)/(b)/(c) headings,
frame on, ticks off.

The embedded model is one single held-out model, reconstructed with the same
seed and the same fit/stop split paper_results.py used.  It is the
representation that is being visualised, not the fold ensemble the manuscript
quotes for accuracy; its own test accuracy is printed for the record.

usage: python make_tsne.py [<splits pkl>] [<output dir>]
       defaults: <repo>/splits/dataset_splits.pkl  and  <repo>/results/figures
       needs the DINOv2 manuscript-geometry feature cache under CNT_BASE
"""
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # classification/
import cnt_paths
import paper_protocol as P
import paper_results as PR
import probe_fit as PF

SPLITS = Path(sys.argv[1]) if len(sys.argv) > 1 else cnt_paths.splits_file()
IMG = Path(sys.argv[2]) if len(sys.argv) > 2 else cnt_paths.results_dir() / "figures"

PR.SPLIT_STEM = SPLITS.stem                    # keeps the cache name in step
CACHE = PR.cache_path(P.PRIMARY["encoder"], PR.GEOM_PAPER)
if not CACHE.exists():
    raise SystemExit(f"no feature cache at {CACHE}\n  build it with encoder_bench.py first")

CLASSES = ["Fiber", "Cluster", "Matrix", "MatrixSurface"]
COLORS = ["saddlebrown", "seagreen", "steelblue", "mediumpurple"]
SEED = 42

z = np.load(CACHE, allow_pickle=True)
y, split = z["y"], z["split"]
di, ti = P.dev_test_indices(split)
X = z[f"{P.PRIMARY['mask']}|{P.PRIMARY['pooling']}"].astype(np.float32)
Xd, Xt, yd, yt = X[di], X[ti], y[di], y[ti]
print(f"cache {CACHE.name}\ndev {Xd.shape}  test {Xt.shape}")

# --- rebuild one single held-out model exactly as paper_protocol.run() does ---
i_fit, i_es = P.inner_split(np.arange(len(yd)), yd, SEED)
sc = StandardScaler().fit(Xd[i_fit])
dev = torch.device("cpu")
Xtr = torch.tensor(sc.transform(Xd[i_fit]), device=dev)
ytr = torch.tensor(yd[i_fit], device=dev, dtype=torch.long)
Xes = torch.tensor(sc.transform(Xd[i_es]), device=dev)
yes = torch.tensor(yd[i_es], device=dev, dtype=torch.long)

model = PF.build_head(X.shape[1], P.PRIMARY["classifier"], SEED).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()
g = torch.Generator().manual_seed(SEED)
best, wait, best_state, best_ep = float("inf"), 0, None, 100
for ep in range(1, 101):
    model.train()
    perm = torch.randperm(len(i_fit), generator=g).to(dev)
    for i in range(0, len(i_fit), 64):
        idx = perm[i:i + 64]
        if len(idx) < 2: continue
        opt.zero_grad(); crit(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad(): vl = crit(model(Xes), yes).item()
    if vl < best - 1e-5:
        best, wait, best_ep = vl, 0, ep
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= 10: break
model.load_state_dict(best_state); model.eval()

Xt_s = torch.tensor(sc.transform(Xt), device=dev)
with torch.no_grad():
    h1 = model[3](model[2](model[1](model[0](Xt_s))))
    h2 = model[7](model[6](model[5](model[4](h1))))
    logits = model[8](h2)
pred = logits.argmax(1).numpy()
print(f"reconstructed single model: best epoch {best_ep}, test accuracy "
      f"{100*(pred==yt).mean():.2f}%  ({(pred==yt).sum()}/{len(yt)})")

panels = [(Xt, f"(a) Raw DINOv2 Features ({Xt.shape[1]}-dim)"),
          (h1.numpy(), "(b) After MLP Layer 1 (512-dim)"),
          (h2.numpy(), "(c) After MLP Layer 2 (128-dim)")]

fig, axes = plt.subplots(1, 3, figsize=(20.6, 7.05), dpi=300)
for ax, (F, title) in zip(axes, panels):
    emb = TSNE(n_components=2, random_state=SEED, init="pca",
               perplexity=30, max_iter=1500).fit_transform(
                   StandardScaler().fit_transform(F))
    for c in range(4):
        m = yt == c
        ax.scatter(emb[m, 0], emb[m, 1], s=42, c=COLORS[c], alpha=0.8,
                   linewidths=0, label=CLASSES[c])
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    ax.set_xlabel("t-SNE 1", fontsize=15, labelpad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.0); s.set_color("black")
axes[0].set_ylabel("t-SNE 2", fontsize=15, labelpad=10)
leg = axes[0].legend(loc="upper right", fontsize=13, frameon=True,
                     framealpha=1.0, edgecolor="0.8", markerscale=1.4,
                     borderpad=0.6, labelspacing=0.5)
leg.get_frame().set_linewidth(0.8)
fig.tight_layout()
IMG.mkdir(parents=True, exist_ok=True)
out = IMG / "TSNE_evolution.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", out)
