#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 Fine-tuned classification baselines  --  ResNet-50, YOLO11-cls, YOLO26-cls,
 supervised ViT-B/16  vs  our frozen DINOv2 + mask-guided probe
================================================================================

Conventional transfer-learning controls: a fine-tuned ResNet-50, YOLO11 or
supervised ViT, to test the claim that foundation-model features outperform
transfer learning. The encoder benchmark already covers ResNet-50 and a
supervised ViT as FROZEN feature extractors, which is a different thing. This
script fine-tunes the whole network end to end, which is the direct form of
that comparison and the harder test for us.

GIVING THE BASELINES A FAIR SHOT
--------------------------------
A baseline that loses because it was set up badly proves nothing, so each model
gets a genuinely modern recipe:

  * ImageNet-pretrained initialisation
  * discriminative learning rates (backbone 1e-4, head 1e-3), AdamW,
    cosine schedule with warmup
  * augmentation chosen for this modality: random resized crop, horizontal AND
    vertical flips, and 90-degree rotations. Electron micrographs have no
    canonical orientation, so these are label-preserving here in a way they
    would not be for natural images -- this is a real advantage handed to the
    baselines, not a token one
  * label smoothing 0.1, mixed precision, early stopping on an inner split
  * BOTH masked and raw inputs, with the better of the two BY CROSS-VALIDATED
    ACCURACY reported per model (make_tables.py); the test set plays no part

Every model sees the same images, the same folds, the same seeds and the same
held-out test set as the DINOv2 pipeline. The test set is scored once.

WHAT IS NOT MATCHED, AND WE SAY SO IN THE PAPER
-----------------------------------------------
These baselines update every weight; our DINOv2 pipeline trains only a small
head on frozen features. That asymmetry favours the baselines on capacity and
disfavours them on data efficiency. It is the comparison that matters here --
frozen foundation features against a properly fine-tuned conventional network --
so we report it rather than trying to equalise it away.

USAGE
    This file is a library.  The baselines are run ONLY through the shared
    protocol, so the same rules apply to them as to every other method:

        python paper_results.py --stages finetune

    (An earlier stand-alone driver in this file could pick masked-vs-raw on
     the test set when cross-validation was disabled; it has been removed.)
================================================================================
"""
from __future__ import annotations

import math, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import gpu_boost

CLASSES = ["Fiber", "Cluster", "Matrix", "MatrixSurface"]
CID = {c: i for i, c in enumerate(CLASSES)}
SEED = 42

MODELS = {
    "resnet50":  dict(kind="timm", id="resnet50.a1_in1k",
                      note="conventional transfer-learning control"),
    "vit_b16":   dict(kind="timm", id="vit_base_patch16_224.augreg2_in21k_ft_in1k",
                      note="supervised ViT (transfer-learning control)"),
    "convnextv2":dict(kind="timm", id="convnextv2_base.fcmae_ft_in1k",
                      note="modern CNN control"),
    "yolo11":    dict(kind="yolo", id="yolo11s-cls.pt",
                      note="YOLOv11 classification variant"),
    "yolo26":    dict(kind="yolo", id="yolo26s-cls.pt",
                      note="current YOLO generation"),
}


class Cfg:
    """Fine-tuning hyperparameters.  Defaults match the CLI defaults, so a run
    driven by paper_results.py is identical to one driven from this file."""
    def __init__(self, **kw):
        self.epochs, self.batch, self.folds = 30, 32, 5
        self.seeds, self.patience, self.yolo_imgsz = 3, 8, 224
        self.lr_backbone, self.lr_head = 1e-4, 1e-3
        self.workers, self.device, self.out = 4, "cuda", None
        self.work = None
        for k, v in kw.items():
            setattr(self, k, v)
        if self.work is None and self.out is not None:
            from pathlib import Path
            self.work = Path(self.out) / "_work"
        if self.work is not None:
            from pathlib import Path
            Path(self.work).mkdir(parents=True, exist_ok=True)


def log(m=""): print(m, flush=True)
def banner(m): log("\n" + "=" * 78); log(m); log("=" * 78)


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, c - h), 100 * min(1, c + h)


# ---------------------------------------------------------------- data
def load_gray(p):
    from PIL import Image
    with Image.open(p) as im:
        if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
            a = np.asarray(im, dtype=np.float32)
            lo, hi = float(a.min()), float(a.max())
            a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
            return (a * 255).astype(np.uint8)
        return np.asarray(im.convert("L"), dtype=np.uint8)


def masked_crop(img, mask, margin=0.10):
    """Crop to the mask bounding box with a margin, then zero the background.
    This is the fine-tuning analogue of mask-guided pooling: the network sees
    the particle and not the field around it."""
    m = mask > 127
    if not m.any():
        return img
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = img.shape
    py, px = int((y1 - y0) * margin) + 1, int((x1 - x0) * margin) + 1
    y0, y1 = max(0, y0 - py), min(h, y1 + py + 1)
    x0, x1 = max(0, x0 - px), min(w, x1 + px + 1)
    out = img.copy()
    out[~m] = 0
    return out[y0:y1, x0:x1]


_PREP_CACHE = {}


PREP_MAX_SIDE = 384      # see the note in prepare()
PREP_DISK_CACHE = True


def _shrink(img, max_side):
    """Downscale so the long side is at most `max_side`.  NEVER upscales, so a
    small masked crop is kept at its native resolution."""
    if max_side is None:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    from PIL import Image as _I
    s = max_side / m
    return np.asarray(_I.fromarray(img).resize(
        (max(1, int(round(w * s))), max(1, int(round(h * s)))), _I.BILINEAR))


def prepare(rows, use_mask, tag=""):
    """Decode and (optionally) mask a split, cached in memory AND on disk.

    The source images are up to 2048x2048 TIFFs, ~1.4 GB per pass, and this ran once
    per (split, variant) on every process start -- which dominated the fine-tune
    stage far more than training did.  Results are now cached to
    <CNT_BASE>/_cache/prep/ under a content hash of the exact file list, so a
    changed split cannot silently reuse a stale cache.

    Images are stored downscaled to a long side of PREP_MAX_SIDE (default 384).
    Training augments with RandomResizedCrop(224, scale=(0.7, 1.0)), whose
    largest possible source region is 224/0.7 = 320 px, so 384 keeps full
    augmentation headroom while cutting the cache to ~250 MB per variant.
    Nothing is ever upscaled.
    """
    key = (tag, use_mask, len(rows), PREP_MAX_SIDE,
           str(rows[0]["image"]) if rows else "",
           str(rows[-1]["image"]) if rows else "")
    if key in _PREP_CACHE:
        return _PREP_CACHE[key]

    path = None
    if PREP_DISK_CACHE and rows:
        import cache_setup
        path = cache_setup.prep_cache_path(tag or "split", use_mask, rows,
                                           PREP_MAX_SIDE)
        if path.exists():
            try:
                z = np.load(path, allow_pickle=True)
                ys = z["y"]
                out = [(z[f"i{i}"], int(ys[i])) for i in range(len(ys))]
                if len(out) == len(rows):
                    log(f"  [cache] {tag}: {len(out)} images from {path.name}")
                    _PREP_CACHE[key] = out
                    return out
                log(f"  [cache] {path.name} has {len(out)} rows, expected "
                    f"{len(rows)}; rebuilding")
            except Exception as e:
                log(f"  [cache] could not read {path.name} "
                    f"({type(e).__name__}); rebuilding")

    def one(r):
        img = load_gray(r["image"])
        if use_mask:
            img = masked_crop(img, load_gray(r["mask"]))
        return (_shrink(img, PREP_MAX_SIDE), r["y"])

    t0 = time.time()
    out = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=gpu_boost.workers()) as ex:
        for i, v in enumerate(ex.map(one, rows)):
            out[i] = v
    log(f"  [cache] {tag}: decoded {len(out)} images in {time.time()-t0:.0f}s")

    if path is not None:
        try:
            tmp = path.with_suffix(".part.npz")
            np.savez(tmp, y=np.array([y for _, y in out], dtype=np.int16),
                     **{f"i{i}": im for i, (im, _) in enumerate(out)})
            tmp.replace(path)
            mb = path.stat().st_size / 1e6
            log(f"  [cache] wrote {path.name} ({mb:.0f} MB) -- "
                f"later runs skip the decode")
        except Exception as e:
            log(f"  [cache] could not write {path}: {type(e).__name__}: {e}")

    _PREP_CACHE[key] = out
    return out


class ArrDataset:
    def __init__(self, items, size, train, mean, std):
        self.items, self.size, self.train = items, size, train
        self.mean, self.std = mean, std

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        import torchvision.transforms.v2 as T
        img, y = self.items[i]
        pil = Image.fromarray(img).convert("RGB")
        if self.train:
            tf = T.Compose([
                T.RandomResizedCrop(self.size, scale=(0.7, 1.0), antialias=True),
                T.RandomHorizontalFlip(),          # micrographs have no canonical
                T.RandomVerticalFlip(),            # orientation, so flips and 90-degree
                T.RandomChoice([T.RandomRotation((0, 0)),   # rotations are label-preserving
                                T.RandomRotation((90, 90)),
                                T.RandomRotation((180, 180)),
                                T.RandomRotation((270, 270))]),
                T.ToImage(), T.ToDtype(torch.float32, scale=True),
                T.Normalize(self.mean, self.std)])
        else:
            tf = T.Compose([T.Resize(int(self.size * 1.14), antialias=True),
                            T.CenterCrop(self.size),
                            T.ToImage(), T.ToDtype(torch.float32, scale=True),
                            T.Normalize(self.mean, self.std)])
        return tf(pil), y


# ---------------------------------------------------------------- timm path
def finetune_timm(model_id, tr_items, es_items, ev_items, cfg, seed,
                  fixed_epochs=None):
    import torch, timm
    import torch.nn as nn
    from torch.utils.data import DataLoader

    torch.manual_seed(seed); np.random.seed(seed)
    dev = cfg.device
    import os
    pre = os.environ.get("FT_NO_PRETRAINED") != "1"
    model = timm.create_model(model_id, pretrained=pre, num_classes=4).to(dev)
    is_conv = not hasattr(model, "patch_embed")
    if is_conv and dev == "cuda":
        model = gpu_boost.to_channels_last(model)
    dc = timm.data.resolve_data_config({}, model=model)
    size = dc["input_size"][-1]
    mean, std = list(dc["mean"]), list(dc["std"])

    dl = lambda items, train: DataLoader(
        ArrDataset(items, size, train, mean, std), batch_size=cfg.batch,
        shuffle=train, num_workers=cfg.workers, drop_last=train and len(items) > cfg.batch,
        pin_memory=(dev == "cuda"), persistent_workers=(cfg.workers > 0),
        prefetch_factor=(4 if cfg.workers > 0 else None))
    tr_dl = dl(tr_items, True)
    es_dl = dl(es_items, False) if es_items else None
    # ev_items is a LIST of item-lists, one per prediction target
    ev_dls = [dl(t, False) for t in ev_items]

    # discriminative learning rates: gently on the pretrained trunk, faster on the head
    head_names = ("head", "fc", "classifier")
    head, trunk = [], []
    for n, p in model.named_parameters():
        (head if any(h in n for h in head_names) else trunk).append(p)
    opt = torch.optim.AdamW([{"params": trunk, "lr": cfg.lr_backbone},
                             {"params": head,  "lr": cfg.lr_head}], weight_decay=0.05)
    use_es = fixed_epochs is None
    n_ep = cfg.epochs if use_es else int(fixed_epochs)

    # The schedule must span the epochs ACTUALLY run.  Building it for
    # cfg.epochs while the refit runs `fixed_epochs` leaves the model stranded
    # mid-cosine at a high learning rate: a 13/30 budget stops at 70% of peak
    # LR, which took ConvNeXt-V2 to 25.00% (chance) and ViT-B/16 down 4-6 pp.
    steps = max(1, len(tr_dl)) * n_ep
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))

    best, wait, best_state, best_ep = float("inf"), 0, None, n_ep
    for ep in range(n_ep):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if is_conv and dev == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if not use_es:
            continue
        model.eval(); tot = n = 0.0
        with torch.no_grad():
            for x, y in es_dl:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                    tot += crit(model(x), y).item() * len(y)
                n += len(y)
        vl = tot / max(n, 1)
        if vl < best - 1e-4:
            best, wait, best_ep = vl, 0, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience: break
    if use_es and best_state: model.load_state_dict(best_state)

    # One trained model scores EVERY target -- a fold model is never trained
    # twice, and the ensemble members are exactly the CV models.
    model.eval(); out = []
    for _dl in ev_dls:
        probs = []
        with torch.no_grad():
            for x, _ in _dl:
                with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                    probs.append(torch.softmax(model(x.to(dev)).float(), 1).cpu().numpy())
        out.append(np.concatenate(probs))
    del model
    if dev == "cuda":
        import torch as _t; _t.cuda.empty_cache()
    return out, best_ep


# ---------------------------------------------------------------- yolo path
_YOLO_RUN = 0


def finetune_yolo(model_id, tr_items, es_items, ev_items, cfg, seed, tag,
                  fixed_epochs=None):
    """Ultralytics needs a folder dataset, so we materialise one per run.

    Contract matches finetune_timm: `ev_items` is a LIST of item-lists (one per
    prediction target), the return is (list of probability arrays, best_epoch),
    and `fixed_epochs` trains for an exact budget with early stopping disabled.
    """
    from PIL import Image
    from ultralytics import YOLO

    use_es = fixed_epochs is None
    n_ep = cfg.epochs if use_es else int(fixed_epochs)

    import cache_setup
    model_id = cache_setup.yolo_checkpoint(model_id, verbose=False) \
               if str(model_id).endswith(".pt") and not Path(model_id).is_absolute() \
               else model_id

    global _YOLO_RUN
    _YOLO_RUN += 1
    run = f"{tag}_{seed}_{_YOLO_RUN:03d}"      # unique per call: folds must not
    root = cfg.work / f"yolo_{run}"            # share a run directory
    if root.exists(): shutil.rmtree(root)
    for i, (img, y) in enumerate(tr_items):
        d = root / "train" / CLASSES[y]; d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).convert("RGB").save(d / f"{i:05d}.png")

    if use_es:
        for i, (img, y) in enumerate(es_items):
            d = root / "val" / CLASSES[y]; d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).convert("RGB").save(d / f"{i:05d}.png")
    else:
        # Refit variant: no held-out set exists by construction -- the epoch
        # budget came from the folds and early stopping is off.  Ultralytics'
        # classification trainer builds its test loader from val/ regardless of
        # val=False and raises on an empty directory, so val/ gets a small
        # stratified COPY of training images purely to satisfy the API.  Those
        # images are already in train/, so nothing is held out and all
        # len(tr_items) images are still trained on.  The resulting val metrics
        # are meaningless, which is why `last.pt` is loaded below instead of the
        # val-selected `best.pt`.
        by_cls = {}
        for i, (img, y) in enumerate(tr_items):
            by_cls.setdefault(y, []).append((i, img))
        for y, group in by_cls.items():
            d = root / "val" / CLASSES[y]; d.mkdir(parents=True, exist_ok=True)
            for i, img in group[:8]:
                Image.fromarray(img).convert("RGB").save(d / f"{i:05d}.png")
    m = YOLO(model_id)
    m.train(data=str(root), epochs=n_ep, imgsz=cfg.yolo_imgsz,
            batch=cfg.batch,
            patience=(cfg.patience if use_es else n_ep + 1),
            seed=seed,
            device=(0 if cfg.device == "cuda" else "cpu"),
            project=str(cfg.work / "runs"), name=run, exist_ok=True,
            verbose=False, val=use_es, plots=False,
            fliplr=0.5, flipud=0.5, degrees=90.0, scale=0.3,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, erasing=0.0, auto_augment=None)

    best_ep = n_ep
    sd = getattr(getattr(m, "trainer", None), "save_dir", None)
    if use_es and sd:
        # Ultralytics exposes no best_epoch attribute; every fold silently
        # reported the cap.  Recover it from the per-epoch results.csv instead.
        import csv as _csv
        rp = Path(sd) / "results.csv"
        if rp.exists():
            try:
                rws = list(_csv.DictReader(open(rp)))
                col = next((c for c in rws[0]
                            if "accuracy_top1" in c or "top1" in c), None)
                if col and rws:
                    best_ep = 1 + max(range(len(rws)),
                                      key=lambda i: float(rws[i][col]))
            except Exception as e:
                log(f"    [warn] could not read {rp.name} ({type(e).__name__}); "
                    f"reporting the epoch cap")
    if not use_es:
        last = Path(sd) / "weights" / "last.pt" if sd else None
        if last and last.exists():
            m = YOLO(str(last))          # NOT best.pt -- see the note above
        else:
            log(f"    [warn] {run}: last.pt missing; falling back to the "
                f"in-memory model")

    outs = []
    for t_i, items in enumerate(ev_items):
        ev_dir = cfg.work / f"yolo_{run}_eval{t_i}"
        if ev_dir.exists(): shutil.rmtree(ev_dir)
        ev_dir.mkdir(parents=True)
        paths = []
        for i, (img, _y) in enumerate(items):
            p = ev_dir / f"{i:05d}.png"
            Image.fromarray(img).convert("RGB").save(p); paths.append(p)
        probs = np.zeros((len(paths), 4), dtype=np.float64)
        k = 0
        for i in range(0, len(paths), 64):
            for r in m.predict([str(p) for p in paths[i:i + 64]], verbose=False,
                               device=(0 if cfg.device == "cuda" else "cpu")):
                # YOLO orders classes by sorted folder name; remap into OUR
                # canonical class order so every method's columns line up.
                pr = r.probs.data.detach().cpu().numpy().astype(np.float64)
                for j, nm in r.names.items():
                    probs[k, CID[nm]] = pr[int(j)]
                k += 1
        probs /= probs.sum(1, keepdims=True).clip(min=1e-12)
        outs.append(probs)
        shutil.rmtree(ev_dir, ignore_errors=True)
    shutil.rmtree(root, ignore_errors=True)
    return outs, best_ep




def inner_split(idx, y, seed, n_classes=4):
    """Carve an early-stopping split that is guaranteed to contain every class.
    A flat 10 % can drop below n_classes on small folds, which sklearn rejects."""
    from sklearn.model_selection import train_test_split
    n_es = min(max(int(0.1 * len(idx)), 2 * n_classes), max(len(idx) - 2 * n_classes, 1))
    return train_test_split(idx, test_size=n_es, stratify=y[idx], random_state=seed)

# ---------------------------------------------------------------- driver
# There is deliberately NO stand-alone driver in this file.  An earlier
# version had one (`run_model`) whose `report()` fell back to test accuracy when
# cross-validation was skipped with `--folds 0`, so the masked-vs-raw choice
# could be made on the held-out test set -- directly under a comment claiming
# selection on CV.  That path was never used for a reported number, and it has
# been removed rather than left in place.  The only way to run these baselines
# is through the shared protocol:
#
#     python paper_results.py --stages finetune
#
# which applies paper_protocol.run() to every model and never sees the test
# set before scoring it once.


def main():
    raise SystemExit(
        "finetune_baselines.py is a library used by paper_results.py; it has no\n"
        "driver of its own.  Run:\n"
        "    python paper_results.py --stages finetune\n"
        "(The old stand-alone driver could select masked-vs-raw on the test set\n"
        " when run with --folds 0, and was removed for that reason.)")


if __name__ == "__main__":
    main()
