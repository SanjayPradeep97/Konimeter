"""
analyse.py -- paired significance tests and confidence intervals.

WHY OUT-OF-FOLD AND NOT THE TEST SET
------------------------------------
The advantage over Luo et al. is small, so a state-of-the-art claim needs a
PAIRED test, and it needs enough samples to resolve the gap.  On a ~180-image
test set, exact McNemar needs a ~5-6 pp difference before p<0.05:

    b=9,  c=3   (3.3 pp)  p = 0.146
    b=12, c=4   (4.4 pp)  p = 0.077
    b=15, c=5   (5.6 pp)  p = 0.041

Essentially no comparison in this paper is resolvable at n=179.  The same test
on the 1,606 out-of-fold predictions resolves 1.2-1.9 pp.  So every
method-vs-method claim is made on OOF; the test set is reported once, for the
pre-declared PRIMARY configuration only, as a Wilson interval.

WHY NOT A PAIRED t-TEST OVER THE 5 FOLDS
----------------------------------------
Fold training sets overlap by 3/5, so fold-wise differences are strongly
positively correlated and the naive paired t-test is badly anti-conservative.
Simulated Type-I error at nominal alpha=0.05 for two IDENTICAL methods:
rho=0.3 -> 0.194, rho=0.5 -> 0.319, rho=0.7 -> 0.476.  Where a fold-wise test is
reported at all it uses the Nadeau-Bengio correction, whose variance multiplier
here is 1/k + n_test/n_train = 1/5 + 324/1296 = 0.450 against the naive 0.200 --
the uncorrected standard error is understated by 1.50x.
"""
from __future__ import annotations
import numpy as np


def mcnemar_exact(pred_a, pred_b, y):
    """Exact (binomial) McNemar on paired predictions.  Returns b, c, p."""
    a_ok = np.asarray(pred_a) == np.asarray(y)
    b_ok = np.asarray(pred_b) == np.asarray(y)
    b = int(np.sum(a_ok & ~b_ok))       # a right, b wrong
    c = int(np.sum(~a_ok & b_ok))       # a wrong, b right
    n = b + c
    if n == 0:
        return b, c, 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1))
    p = min(1.0, 2.0 * tail / (2 ** n))
    return b, c, p


def holm(pvals, labels):
    """Holm-Bonferroni over the family of comparisons actually reported."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return {labels[i]: float(adj[i]) for i in range(m)}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def mask_effect(rows):
    """Paired masked-vs-uniform comparison, reported SEPARATELY per family.

    The two families are not the same intervention and must never be pooled:

      frozen     mask-guided token pooling -- the encoder is frozen and features
                 are pooled only over tokens the mask covers.  The image itself
                 is untouched, so nothing but the pooling support changes.
      finetuned  the image is cropped to the mask bounding box and the
                 background zeroed.  That ALSO changes effective magnification
                 per image, so a difference here confounds masking with zoom.

    Pooling them produced a single "57/59, p=6e-15" headline that hid the fact
    that the effect is confined to the frozen family.  Matched within a family
    on (method, geometry, pooling, classifier), so the only thing differing
    inside a pair is the mask setting.  Sign test: no distributional assumption.
    """
    from collections import defaultdict
    from math import comb

    out = {}
    fams = defaultdict(list)
    for r in rows:
        fams[r.get("family", "frozen")].append(r)

    for fam, frows in fams.items():
        pairs = defaultdict(dict)
        for r in frows:
            pairs[(r["method"], r.get("geometry", ""), r["pooling"],
                   r["classifier"])][r["mask"]] = r
        d, per_enc = [], defaultdict(list)
        for (enc, _geom, _, _), v in pairs.items():
            if "masked" in v and "nomask" in v:
                delta = v["masked"]["oof_acc"] - v["nomask"]["oof_acc"]
                d.append(delta); per_enc[enc].append(delta)
        if not d:
            continue
        d = np.array(d); n = len(d); pos = int((d > 0).sum())
        p_sign = min(1.0, 2 * sum(comb(n, i) for i in range(pos, n + 1)) / 2 ** n)
        t = (float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
             if n > 1 and d.std(ddof=1) > 0 else float("nan"))
        out[fam] = dict(n_pairs=n, n_favouring_masked=pos,
                        mean_delta=float(d.mean()), min_delta=float(d.min()),
                        max_delta=float(d.max()), sign_test_p=float(p_sign),
                        paired_t=t, df=n - 1,
                        per_encoder={k: float(np.mean(v)) for k, v in per_enc.items()})
    return out


def headline(rows, primary):
    """Locate the pre-declared PRIMARY row.  Refuse to emit a table without it."""
    want = dict(method=primary["encoder"], geometry=primary["geometry"],
                mask=primary["mask"], pooling=primary["pooling"],
                classifier=primary["classifier"])
    hits = [r for r in rows if all(r.get(k) == v for k, v in want.items())]
    if not hits:
        raise SystemExit(
            "PRIMARY configuration is absent from the results.\n"
            f"  wanted: {want}\n"
            "  The headline is fixed in advance and cannot be reassigned to "
            "whichever configuration happened to score best.\n"
            "  If `geometry` is what is missing, the manuscript-geometry\n"
            "  features were never extracted:\n"
            "      python encoder_bench.py --stage features --encoders dinov2_b14 \\\n"
            "          --target-grid 37 --taps 1,3,6,9,11")
    if len(hits) > 1:
        raise SystemExit(f"PRIMARY is ambiguous: {len(hits)} matching rows")
    return hits[0]


def compare_all(oof_preds, y_dev, primary_key, min_pairs=1):
    """Every method against PRIMARY, on the paired OOF predictions (1,606 here)."""
    base = oof_preds[primary_key]
    labels, ps, out = [], [], []
    for key, pred in oof_preds.items():
        if key == primary_key:
            continue
        b, c, p = mcnemar_exact(base, pred, y_dev)
        acc_a = 100 * (base == y_dev).mean()
        acc_b = 100 * (pred == y_dev).mean()
        out.append(dict(method=key, acc=acc_b, delta=acc_a - acc_b,
                        b=b, c=c, p_raw=p))
        labels.append(key); ps.append(p)
    adj = holm(ps, labels) if ps else {}
    for r in out:
        r["p_holm"] = adj.get(r["method"], float("nan"))
    out.sort(key=lambda r: -r["acc"])
    return out, len(ps)
