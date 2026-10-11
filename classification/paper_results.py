#!/usr/bin/env python
"""
paper_results.py -- regenerate EVERY quantitative result in the manuscript
under one protocol.

    python paper_results.py --smoke      # ~10 min on a GPU, proves the pipeline runs
    python paper_results.py --plan       # print the work list and exit
    python paper_results.py              # the real run (~3 h on a GPU)

Every method -- frozen probes, fine-tuned CNNs, and the Luo et al.
re-implementation -- is evaluated by paper_protocol.run(), on the same
development images (1,606 for dataset_splits.pkl), the same five folds, and the
same held-out test images (179).
The test set is scored once per reported variant and never influences training,
early stopping, or model selection.

STAGES
  probes     8 frozen encoders (DINOv2 at two geometries, so 9 encoder-geometry
             pairs) x {masked, uniform} x {avg, max, avg+max} x {linear, mlp}
             = 108 configurations.  Features are cached, so this stage trains
             probe heads only.
  finetune   5 architectures x {masked, raw} end-to-end fine-tuned.
  analyse    aggregate, run the statistical tests, emit the manuscript tables.

The Luo et al. baseline is not a stage here: it is run by baseline/run_all.py,
and its results are read from baseline/results/.

Results land in <out>/results_<stage>.csv, per-image predictions in
<out>/per_image_<stage>.json, and the statistics in <out>/analysis.json.  <out>
is results/, except that a --smoke run writes to results/smoke/, so it never
replaces the results the paper reports.  Every run recomputes what it is asked
for; reproduce.sh / reproduce.bat are what skip stage 3 when results exist.
"""
from __future__ import annotations

import argparse, json, sys, time, os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_protocol as P
import cnt_paths

# The data root comes from CNT_BASE (see cnt_paths.py).  It is required for
# every stage except `analyse`, which only reads the results files.
BASE = cnt_paths.base(required=False)
BENCH = cnt_paths.bench_dir() if BASE is not None else None

# Pin every download cache before timm / huggingface_hub / ultralytics load.
# HuggingFace reads its cache location at import time, so this cannot move later.
import cache_setup
if BASE is not None:
    cache_setup.setup(BASE, verbose=False)

# The reported roster.  Every one of these has cached features.
ENCODERS = ["dinov2_b14", "sam_b16", "dinov3_b16", "dinov2_reg",
            "sam2_hiera", "convnextv2_b", "resnet50", "supvit_b16"]

# usam_b (micro-SAM) is deliberately NOT in the default roster.  Without
# USAM_CHECKPOINT it loads samvit_base_patch16.sa1b -- byte-identical to
# sam_b16 -- so a default run would extract features for every image and
# report a duplicate of SAM under a "micro-SAM" label.  It stays available
# via `--encoders ... usam_b` once the checkpoint is set.  It is not reported:
# micro-SAM is trained for biological microscopy, and its own documentation
# directs other users to SAM, which is in the roster.
OPTIONAL_ENCODERS = ["usam_b"]
MASKS = ["masked", "nomask"]
POOLS = ["avg", "max", "avg+max"]
HEADS = ["linear", "mlp"]

FINETUNE = ["resnet50", "vit_b16", "convnextv2", "yolo11", "yolo26"]
INPUTS = ["masked", "raw"]


def log(m=""):
    print(m, flush=True)


def banner(m):
    log("\n" + "=" * 78); log(m); log("=" * 78)


# (label, grid, taps)  taps=int -> evenly spaced; taps=list -> explicit layers
GEOM_BENCH = ("g32_t5", 32, 5)
GEOM_PAPER = ("g37_L1-3-6-9-11", 37, [1, 3, 6, 9, 11])


SPLIT_STEM = os.environ.get("CNT_SPLIT_STEM", cnt_paths.splits_file().stem)
PYRAMID_IMG = 512          # encoder_bench's default; must match for cache names


def cache_path(enc, geom=GEOM_BENCH):
    """Feature-cache path for (encoder, geometry) under the current split.

    Built INDEPENDENTLY of encoder_bench.cache_name() on purpose;
    check_cache_agreement.py asserts the two agree.  The key covers the
    geometry, the split file, the pyramid input size, a random-weights flag and
    any checkpoint override, so a smoke-test or foreign cache is never reused.
    """
    _, grid, taps = geom
    tag = ("L" + "-".join(map(str, taps))) if isinstance(taps, (list, tuple)) \
          else f"t{taps}"
    suffix = "" if SPLIT_STEM == "balanced_dataset_splits" else f"_{SPLIT_STEM}"
    if PYRAMID_IMG != 512:
        suffix += f"_p{PYRAMID_IMG}"
    ckpt = os.environ.get("USAM_CHECKPOINT") if enc == "usam_b" else None
    if ckpt:
        import encoder_bench
        suffix += "_ckpt" + encoder_bench.checkpoint_digest(ckpt)
    if os.environ.get("ENCBENCH_NO_PRETRAINED") == "1":
        suffix += "_RANDOMWEIGHTS"
    bench = BENCH if BENCH is not None else cnt_paths.bench_dir()
    return bench / f"feats_{enc}_g{grid}_{tag}{suffix}.npz"


def geometries_for(enc):
    """Every encoder is benchmarked at the matched geometry.  The manuscript's
    own encoder is ALSO run at the geometry the paper describes, which is what
    the headline row comes from."""
    g = [GEOM_BENCH]
    if enc == P.PRIMARY["encoder"]:
        g.append(GEOM_PAPER)
    return g


# ---------------------------------------------------------------------------
class Progress:
    def __init__(self, total, label="work"):
        self.total, self.done, self.t0, self.label = total, 0, time.time(), label

    def tick(self, n=1):
        self.done += n
        el = time.time() - self.t0
        rate = self.done / el if el > 0 else 0
        eta = (self.total - self.done) / rate if rate > 0 else 0
        pct = 100 * self.done / self.total if self.total else 100
        bar = "#" * int(pct / 2.5) + "." * (40 - int(pct / 2.5))
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {self.done}/{self.total}  "
                         f"elapsed {el/60:5.1f}m  eta {eta/60:5.1f}m   ")
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n"); sys.stdout.flush()
        return time.time() - self.t0


# ---------------------------------------------------------------------------
def stage_probes(args, device, encoders, pools, heads, masks, out_csv):
    """Frozen-encoder probes on cached features."""
    import torch
    from probe_fit import make_probe_fit

    if "usam_b" in encoders and not os.environ.get("USAM_CHECKPOINT"):
        raise SystemExit(
            "ABORT: usam_b requested but USAM_CHECKPOINT is unset.\n"
            "  Without it, usam_b loads samvit_base_patch16.sa1b -- byte-identical\n"
            "  to sam_b16 -- and would be reported as a 'micro-SAM' result.\n"
            "  Set the variable, or drop usam_b from --encoders.")

    todo, missing = [], []
    for enc in encoders:
        for geom in geometries_for(enc):
            if not cache_path(enc, geom).exists():
                missing.append(f"{enc} @ {geom[0]}"); continue
            for mk in masks:
                for pl in pools:
                    for hd in heads:
                        todo.append((enc, geom, mk, pl, hd))
    if missing:
        hint = ("  python encoder_bench.py --stage features --encoders <enc>\n"
                "  ...and for the manuscript geometry, add:  "
                "--target-grid 37 --taps 1,3,6,9,11\n")
        raise SystemExit(
            "ABORT: no cached features for: " + ", ".join(missing) + "\n"
            "  Build them first:\n" + hint +
            "  (A partial run that exits 0 is the most dangerous failure mode "
            "on a deadline, so this refuses to continue.)")

    log(f"  {len(todo)} configurations x {P.n_fits(args.folds)} fits = "
        f"{len(todo)*P.n_fits(args.folds)} probe trainings")
    prog = Progress(len(todo) * P.n_fits(args.folds), "probe fits")
    rows, per_image = [], []

    cur_key, z = None, None
    for enc, geom, mk, pl, hd in todo:
        if (enc, geom[0]) != cur_key:
            z = np.load(cache_path(enc, geom), allow_pickle=True)
            cur_key = (enc, geom[0])
            y_all, split = z["y"], z["split"]
            dev, test = P.dev_test_indices(split)
            yd, yt = y_all[dev], y_all[test]
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
        X = z[f"{mk}|{pl}"]
        fit = make_probe_fit(X[dev], yd, X[test], hd, device,
                             epochs=args.epochs, patience=args.patience)
        r = P.run(fit, yd, yt, n_folds=args.folds, seed=args.seed,
                  label=f"{enc}[{geom[0]}]/{mk}/{pl}/{hd}", log=lambda *a: None,
                  tick=prog.tick)
        row = dict(stage="probe", method=enc, geometry=geom[0], family="frozen",
                   mask=mk, pooling=pl, classifier=hd, dim=int(X.shape[1]),
                   **{k: v for k, v in r.items() if not k.startswith("_")})
        rows.append(row)
        per_image.append(dict(
            key=f"{enc}[{geom[0]}]|{mk}|{pl}|{hd}",
            oof_pred=r["_oof_pred"].astype(int).tolist(),   # all dev images --
            y_dev=yd.astype(int).tolist(),                  # comparisons use
            pred_ensemble=r["_pred_ensemble"].astype(int).tolist(),
            pred_single=r["_pred_single"].astype(int).tolist(),
            pred_refit=r["_pred_refit"].astype(int).tolist(),
            y_test=yt.astype(int).tolist(),
            meta=meta))
    prog.close()
    write_csv(out_csv, rows)
    (out_csv.parent / "per_image_probe.json").write_text(json.dumps(per_image))
    return rows


# ---------------------------------------------------------------------------
def write_csv(path, rows):
    import csv
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})
    log(f"  wrote {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
def stage_finetune(args, device, models, inputs, out_csv):
    """End-to-end fine-tuned baselines, under the identical protocol."""
    import finetune_baselines as FB
    from finetune_fit import make_finetune_fit, EXTRA_MODELS
    FB.PREP_MAX_SIDE = args.prep_max_side or None
    FB.PREP_DISK_CACHE = not args.no_prep_cache

    roster = dict(FB.MODELS); roster.update(EXTRA_MODELS)
    yolo_keys = [m for m in models if m in roster and roster[m]["kind"] == "yolo"]
    if yolo_keys:
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            raise SystemExit(
                "ABORT: ultralytics is not installed but YOLO models were "
                "requested.\n"
                "  Install WITHOUT letting pip touch torch (it will happily\n"
                "  downgrade your sm_120 build and silently drop the 5080):\n"
                "      pip install ultralytics --no-deps\n"
                "      pip install ultralytics-thop py-cpuinfo\n"
                "  or drop them:  --ft-models resnet50 vit_b16 convnextv2")
        # Resolve every checkpoint NOW.  A name that 404s is worth finding in
        # ten seconds, not three hours into the overnight run.
        cache_setup.prefetch_amp_model(verbose=True)
        from ultralytics import YOLO
        for k in yolo_keys:
            cid = cache_setup.yolo_checkpoint(roster[k]["id"])
            try:
                YOLO(cid)
                log(f"  [preflight] {k}: {cid} resolved")
            except Exception as e:
                raise SystemExit(
                    f"ABORT: cannot resolve YOLO checkpoint {cid!r} for {k}.\n"
                    f"  {type(e).__name__}: {str(e)[:160]}\n"
                    f"  Drop it:  --ft-models "
                    + " ".join(m for m in models if m != k))
    unknown = [m for m in models if m not in roster]
    if unknown:
        raise SystemExit(f"ABORT: unknown fine-tune model(s): {unknown}\n"
                         f"  available: {sorted(roster)}")

    from encoder_bench import load_splits
    sp = load_splits(args.splits, args.root, args.root_remap)
    dev_rows = sp["train"] + sp["val"]      # 1,606 for dataset_splits.pkl
    test_rows = sp["test"]                  # 179
    log(f"  dev {len(dev_rows)}  test {len(test_rows)}"
        f"   (load_splits also verifies no image appears in two splits)")

    import gpu_boost
    cfg = FB.Cfg(epochs=args.ft_epochs, patience=args.ft_patience,
                 batch=args.ft_batch, folds=args.folds,
                 device=str(device), out=args.out,
                 workers=gpu_boost.workers())

    rows, per_image = [], []
    prog = Progress(len(models) * len(inputs) * P.n_fits(args.folds), "fine-tunes")
    for mk in models:
        spec = roster[mk]
        for inp in inputs:
            use_mask = (inp == "masked")
            items_dev = FB.prepare(dev_rows, use_mask, f"dev_{inp}")
            items_test = FB.prepare(test_rows, use_mask, f"test_{inp}")
            yd = np.array([y for _, y in items_dev])
            yt = np.array([y for _, y in items_test])
            fit = make_finetune_fit(items_dev, yd, items_test, mk, cfg, spec,
                                    f"{mk}_{inp}")
            r = P.run(fit, yd, yt, n_folds=args.folds, seed=args.seed,
                      label=f"{mk}/{inp}", log=log, tick=prog.tick)
            rows.append(dict(stage="finetune", method=mk, geometry="n/a",
                             family="finetuned",
                             mask="masked" if use_mask else "nomask",
                             pooling="n/a", classifier="end-to-end",
                             timm_id=spec["id"],
                             **{k: v for k, v in r.items() if not k.startswith("_")}))
            per_image.append(dict(key=f"{mk}|{inp}",
                                  oof_pred=r["_oof_pred"].astype(int).tolist(),
                                  y_dev=yd.astype(int).tolist(),
                                  pred_ensemble=r["_pred_ensemble"].astype(int).tolist(),
                                  pred_single=r["_pred_single"].astype(int).tolist(),
                                  y_test=yt.astype(int).tolist()))
    prog.close()
    write_csv(out_csv, rows)
    (out_csv.parent / "per_image_finetune.json").write_text(json.dumps(per_image))
    return rows


# ---------------------------------------------------------------------------
def stage_analyse(args, out_dir):
    """Every comparison the manuscript makes, with the test it actually needs."""
    import csv
    from analyse import mask_effect, headline, compare_all, wilson_ci

    rows, per_image = [], {}
    for f in sorted(out_dir.glob("results_*.csv")):
        if ("smoke" in f.name) != bool(getattr(args, "smoke", False)):
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                for k in ("oof_acc", "cv_acc", "cv_std", "test_ensemble_acc",
                          "test_single_acc", "test_refit_acc"):
                    if r.get(k) not in (None, ""):
                        r[k] = float(r[k])
                # rows written before the flag existed: derive it with the same rule
                if r.get("refit_unreliable") in (None, "") and r.get("epoch_budget"):
                    r["refit_unreliable"] = P.refit_unreliable(int(r["epoch_budget"]))
                rows.append(r)
    for f in sorted(out_dir.glob("per_image_*.json")):
        for e in json.loads(f.read_text()):
            per_image[e["key"]] = e
    if not rows:
        raise SystemExit("ABORT: no results to analyse.")

    banner("ANALYSIS")
    log(f"  {len(rows)} configurations, {len(per_image)} with per-image predictions")

    hl = headline(rows, P.PRIMARY)
    # must match the per_image key format, which carries the geometry
    geo = hl.get("geometry", "n/a")
    pk = (f"{hl['method']}[{geo}]|{hl['mask']}|{hl['pooling']}|{hl['classifier']}"
          if hl.get("stage") == "probe"
          else f"{hl['method']}|{hl['mask']}")
    log(f"\n  HEADLINE (pre-declared, not selected on results)")
    log(f"    {pk}")
    log(f"    geometry       {geo}"
        + ("   (518x518, taps [1,3,6,9,11], 3840-D -- the manuscript's "
           "configuration)" if geo == "g37_L1-3-6-9-11" else ""))
    n_oof, n_test = int(hl["oof_n"]), int(hl["test_n"])
    log(f"    OOF accuracy   {hl['oof_acc']:.2f}%  over {n_oof:,} predictions   <- PRIMARY METRIC")
    k = int(round(hl["test_ensemble_acc"] / 100 * n_test))
    lo, hi = wilson_ci(k, n_test)
    log(f"    held-out test  {hl['test_ensemble_acc']:.2f}%  ({k}/{n_test})  95% CI [{lo:.1f}, {hi:.1f}]")
    log(f"    (the test set is reported once, for this configuration only)")

    me = mask_effect(rows)
    LBL = {"frozen": ("FROZEN ENCODERS -- mask-guided token pooling",
                      "the image is untouched; only the pooling support changes"),
           "finetuned": ("FINE-TUNED -- crop to mask bbox, background zeroed",
                         "a DIFFERENT intervention: also changes per-image "
                         "magnification, so masking is confounded with zoom")}
    for fam in ("frozen", "finetuned"):
        s = me.get(fam)
        if not s:
            continue
        title, note = LBL[fam]
        log(f"\n  {title}   (paired, on OOF accuracy)")
        log(f"    {note}")
        log(f"    {s['n_favouring_masked']}/{s['n_pairs']} matched pairs favour mask-guided")
        log(f"    mean {s['mean_delta']:+.2f} pp   range {s['min_delta']:+.2f} to "
            f"{s['max_delta']:+.2f}   sign test p = {s['sign_test_p']:.4g}"
            f"   paired t = {s['paired_t']:.2f} (df={s['df']})")
        for k, v in sorted(s["per_encoder"].items(), key=lambda x: -x[1]):
            log(f"      {k:18s} {v:+.2f} pp")

    if pk in per_image:
        y_dev = np.array(per_image[pk]["y_dev"])
        oof = {k2: np.array(v["oof_pred"]) for k2, v in per_image.items()}
        comps, fam = compare_all(oof, y_dev, pk)
        log(f"\n  EVERY METHOD vs PRIMARY  (exact McNemar on {len(y_dev):,} OOF predictions,")
        log(f"  Holm-corrected over a family of {fam})")
        log(f"    {'method':38s} {'OOF%':>7s} {'delta':>7s} {'b':>4s} {'c':>4s} "
            f"{'p':>10s} {'p_holm':>9s}")
        for c in comps:
            log(f"    {c['method']:38s} {c['acc']:7.2f} {c['delta']:+7.2f} "
                f"{c['b']:4d} {c['c']:4d} {c['p_raw']:10.3g} {c['p_holm']:9.3g}")
        json.dump(dict(headline=hl, mask_effect=me, comparisons=comps,
                       family_size=fam, primary=P.PRIMARY),
                  open(out_dir / "analysis.json", "w"), indent=2, default=str)
        log(f"\n  wrote {out_dir/'analysis.json'}")
    else:
        log(f"\n  !! no per-image predictions for PRIMARY ({pk}); "
            f"McNemar comparisons skipped")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="a short end-to-end check on one configuration; writes "
                         "to results/smoke/ unless --out is given")
    # The full run is the default; --full is accepted so older commands work.
    ap.add_argument("--full", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--stages", nargs="+", choices=["probes", "finetune", "luo", "analyse"],
                    default=["probes", "finetune", "analyse"])
    ap.add_argument("--out", type=Path, default=None,
                    help="results directory (default <repo>/results, or "
                         "<repo>/results/smoke with --smoke)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    # Nothing here skips completed work, so there is nothing to force;
    # reproduce.sh / reproduce.bat take --force.  Accepted so older commands work.
    ap.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--prep-max-side", type=int, default=384,
                    help="cache decoded images downscaled to this long side "
                         "(never upscaled). 0 keeps full 1350x1040 resolution.")
    ap.add_argument("--no-prep-cache", action="store_true",
                    help="always re-decode the TIFFs instead of using "
                         "<CNT_BASE>/_cache/prep")
    ap.add_argument("--root", type=Path, default=None,
                    help="data root; default is the CNT_BASE environment variable")
    ap.add_argument("--root-remap", default=None, help=argparse.SUPPRESS)  # obsolete
    ap.add_argument("--splits", type=Path, default=cnt_paths.splits_file(),
                    help="split pickle. Switching this changes the dataset the whole "
                         "paper is about; every stage must be re-run, not just one.")
    ap.add_argument("--encoders", nargs="+", default=None,
                    help="override the default roster (e.g. to add usam_b, "
                         "which additionally needs USAM_CHECKPOINT set)")
    ap.add_argument("--ft-models", nargs="+", default=FINETUNE)
    ap.add_argument("--ft-epochs", type=int, default=30)
    ap.add_argument("--ft-patience", type=int, default=8)
    ap.add_argument("--ft-batch", type=int, default=32)
    a = ap.parse_args(argv)

    if "luo" in a.stages:
        # Refused before anything runs, not after hours of probes.
        raise SystemExit(
            "The Luo stage is not run from here: baseline/run_all.py runs it, and\n"
            "  baseline/results/luo_baseline_summary.csv already holds both split "
            "files x clean/leaky.\n  Drop 'luo' from --stages.")
    if a.out is None:
        # A smoke run's per-image predictions and analysis.json have the same
        # names as the shipped ones, so it gets a directory of its own.
        a.out = cnt_paths.results_dir() / "smoke" if a.smoke else cnt_paths.results_dir()

    import torch
    dev = torch.device("cuda" if (a.device in ("auto", "cuda")
                                  and torch.cuda.is_available()) else "cpu")
    a.out.mkdir(parents=True, exist_ok=True)
    global SPLIT_STEM, BASE, BENCH
    SPLIT_STEM = Path(a.splits).stem
    if a.root is not None:
        os.environ["CNT_BASE"] = str(a.root)
    needs_data = any(st in a.stages for st in ("probes", "finetune")) or a.plan
    if needs_data or cnt_paths.base(required=False) is not None:
        BASE = cnt_paths.base()
        BENCH = cnt_paths.bench_dir()
        cdir = cache_setup.setup(BASE, verbose=False)
    else:
        cdir = "(not needed for --stages analyse)"

    encoders = a.encoders or ENCODERS
    pools, heads, masks = POOLS, HEADS, MASKS
    ft_models, ft_inputs = a.ft_models, INPUTS
    if a.smoke:
        # Derived from PRIMARY, not hard-coded: the smoke must produce the
        # headline row, or `analyse` aborts on a missing PRIMARY and a healthy
        # smoke reads as a failure.  Both mask settings are kept so the paired
        # mask-effect test has one matched pair to work on.
        encoders = [P.PRIMARY["encoder"]]
        pools = [P.PRIMARY["pooling"]]
        heads = [P.PRIMARY["classifier"]]
        masks = MASKS
        a.epochs, a.patience = 12, 4
        ft_models, ft_inputs = ["resnet50", "yolo11"], ["masked"]
        a.ft_epochs, a.ft_patience, a.folds = 2, 1, 2

    banner("PAPER RESULTS -- unified protocol")
    log(f"  device        {dev}"
        + (f"  ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    log(f"  primary       {P.PRIMARY}   <- declared before any results exist")
    log(f"  protocol      {a.folds}-fold CV on dev; test scored once per variant")
    log(f"  data root     {BASE}")
    log(f"  out           {a.out}")
    log(f"  caches        {cdir}   (weights, HF hub, decoded images)")
    log(f"  splits        {a.splits.name}")

    if a.plan:
        n = (sum(len(geometries_for(e)) for e in encoders)
             * len(masks) * len(pools) * len(heads))
        log(f"\n  probes    {n} configs x {P.n_fits(a.folds)} fits = {n*P.n_fits(a.folds)} trainings")
        log(f"  finetune  {len(FINETUNE)*len(INPUTS)} configs x {P.n_fits()} fits "
            f"= {len(FINETUNE)*len(INPUTS)*P.n_fits()} fine-tunes")
        log(f"  luo       1 config x {P.n_fits()} fits")
        return 0

    t0 = time.time()
    if "probes" in a.stages:
        banner("STAGE: frozen-encoder probes")
        stage_probes(a, dev, encoders, pools, heads, masks,
                     a.out / ("results_probes_smoke.csv" if a.smoke
                              else "results_probes.csv"))
    if "finetune" in a.stages:
        banner("STAGE: fine-tuned baselines")
        stage_finetune(a, dev, ft_models, ft_inputs,
                       a.out / ("results_finetune_smoke.csv" if a.smoke
                                else "results_finetune.csv"))
    if "analyse" in a.stages:
        stage_analyse(a, a.out)
    log(f"\n  total {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
