"""Does the pre-specified probe want more epochs?

Decision metric is CROSS-VALIDATED (out-of-fold) accuracy.  The test column is
printed as a diagnostic only: choosing the epoch budget by that column is
exactly the mechanism that once inflated the reported accuracy, so it is
labelled and must not be used for selection.

Each budget trains for exactly that many epochs with early stopping OFF, over
3 seeds x 5 folds, plus a refit on all 1,606 dev images.

usage: python epoch_sweep.py [<splits pkl>] [<results dir>]
"""
import numpy as np, torch, json, sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnt_paths
import paper_protocol as P
import paper_results as PR
from probe_fit import make_probe_fit

SPLITS = Path(sys.argv[1]) if len(sys.argv) > 1 else cnt_paths.splits_file()
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else cnt_paths.results_dir()
PR.SPLIT_STEM = SPLITS.stem
BUDGETS = [3, 6, 11, 15, 20, 30, 50, 75, 100, 150]
SEEDS   = [42, 43, 44]

dev_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc, geom = P.PRIMARY["encoder"], PR.GEOM_PAPER
z = np.load(PR.cache_path(enc, geom), allow_pickle=True)
y_all, split = z["y"], z["split"]
di, ti = P.dev_test_indices(split)
yd, yt = y_all[di], y_all[ti]
X = z[f"{P.PRIMARY['mask']}|{P.PRIMARY['pooling']}"]
Xd, Xt = X[di], X[ti]
print(f"PRIMARY {enc} {geom[0]} {P.PRIMARY['mask']}/{P.PRIMARY['pooling']}/{P.PRIMARY['classifier']}")
print(f"dev {len(yd)}   test {len(yt)}   dim {Xd.shape[1]}   device {dev_}\n")

fit = make_probe_fit(Xd, yd, Xt, P.PRIMARY["classifier"], dev_)
print(f"{'epochs':>7} | {'CV / out-of-fold':>26} | {'test (DIAGNOSTIC - not for selection)':>38}")
print(f"{'':>7} | {'mean':>7} {'sd':>6} {'best-seed':>10} | {'mean':>7} {'sd':>6} {'max':>7}")
print("-"*80)
out = []
for E in BUDGETS:
    oofs, tests = [], []
    for s in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        oof = np.zeros(len(yd), dtype=int)
        for tr, va in skf.split(np.zeros(len(yd)), yd):
            probs, _ = fit(tr, None, [va], fixed_epochs=E, seed=s)
            oof[va] = probs[0].argmax(1)
        oofs.append(100.0 * (oof == yd).mean())
        probs, _ = fit(np.arange(len(yd)), None, ["test"], fixed_epochs=E, seed=s)
        tests.append(100.0 * (probs[0].argmax(1) == yt).mean())
    row = dict(epochs=E, cv_mean=float(np.mean(oofs)), cv_sd=float(np.std(oofs)),
               cv_best=float(max(oofs)), test_mean=float(np.mean(tests)),
               test_sd=float(np.std(tests)), test_max=float(max(tests)))
    out.append(row)
    print(f"{E:7d} | {row['cv_mean']:7.2f} {row['cv_sd']:6.2f} {row['cv_best']:10.2f} | "
          f"{row['test_mean']:7.2f} {row['test_sd']:6.2f} {row['test_max']:7.2f}")

best = max(out, key=lambda r: r["cv_mean"])
print(f"\nCV-selected budget: {best['epochs']} epochs  (CV {best['cv_mean']:.2f}%)")
print(f"Protocol's budget : 11 epochs")
OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / "epoch_sweep.json", "w"), indent=2)
print(f"\nwrote {OUT / 'epoch_sweep.json'}")
