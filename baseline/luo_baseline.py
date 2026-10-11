#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 Luo et al. (2021) baseline, re-implemented on OUR splits
 "A transfer learning approach for improved classification of carbon
  nanomaterials from TEM images", Nanoscale Adv. 3, 206-213 (2021)
================================================================================

WHAT THIS REPRODUCES (all values taken verbatim from the paper)
---------------------------------------------------------------
  * greyscale TEM images normalised to 224 x 224
  * VGG-16 pretrained on ImageNet, frozen
  * hypercolumn descriptors concatenating layers b1c2, b2c2, b3c2, b4c3, b5c3
        -> 64+128+256+512+512 = 1472 features per pixel  (paper states 1472)
  * hypercolumn DENSITY = 10%  (fraction of pixels sampled at random per image)
        -> 0.10 * 224 * 224 = 5017 descriptors per image
        (paper: "a full density hypercolumn of a 224x224 image can cover over
         73 million features" -> 50176 * 1472 = 73.9e6, confirming the reading)
  * K-means visual dictionary, K = 50   (paper: "we chose gradient boosting
        with K = 50 to build our classifier")
  * VLAD encoding of residuals to the nearest centroid -> 50 * 1472 = 73,600-D
  * softmax classifier trained with gradient boosting

DELIBERATE DEVIATIONS, ALL DEFENSIBLE AND ALL LOGGED AT RUNTIME
---------------------------------------------------------------
 1. Gradient boosting implementation: XGBoost (hist / GPU) rather than the
    unspecified 2020 implementation. Same algorithm family, ~100x faster.
    A scikit-learn HistGradientBoosting path is provided as a CPU fallback.
 2. Descriptor normalisation before K-means/VLAD is not specified in the paper.
    Raw VGG activations have different scales per block, which lets block5
    dominate the dictionary. Three options are provided -- 'none' (literal
    reading of the paper), 'l2' (standard VLAD practice) and 'blockl2' (per-
    block then global, the only one that actually equalises block scales).
    run_all.py sweeps all three; the reported variant is the one with the best
    CROSS-VALIDATED accuracy under the clean protocol (make_tables.py), never
    the best test accuracy. Being generous to the baseline is the point: if it
    loses, it must lose honestly.
 3. VLAD post-processing (signed square root + global L2) follows standard
    VLAD practice. Toggle with --no-vlad-postnorm.
 4. NO data augmentation. Luo augmented minority classes to fix a 8%/32%/24%
    class imbalance. Our split (dataset_splits.pkl) is close to balanced --
    430 / 434 / 459 / 462 images per class, a 7.4% spread -- so the
    augmentation step is not needed and is deliberately omitted. This is
    stated explicitly rather than silently dropped.

EVALUATION PROTOCOL (matched to our own pipeline, exactly)
---------------------------------------------------------------
  * identical images, identical splits, loaded from the SAME pkl our DINOv2
    models use, so no re-splitting can drift.
  * PRIMARY: 5-fold StratifiedKFold(shuffle=True, random_state=42) on
    train+val. This mirrors our Supplementary S3 CV *and* Luo's own protocol
    (they report a 5-fold CV mean and never held out a test set at all).
  * SECONDARY: single held-out test evaluation, model fit on train+val.

  --earlystop clean  (DEFAULT)  early stopping on an inner split carved out of
                                train+val. The test set is touched exactly once.
  --earlystop leaky             passes the TEST set as the early-stopping
                                monitor. Provided ONLY so the size of that bias
                                can be measured and reported. Never use it for
                                a published number.

USAGE
---------------------------------------------------------------
  set CNT_BASE=<data root>          (see classification/cnt_paths.py)
  python luo_baseline.py --stage all --desc-norm l2
  python luo_baseline.py --stage all --desc-norm blockl2
  python luo_baseline.py --stage classify --desc-norm l2 --earlystop leaky
  python luo_baseline.py --stage features        # just cache descriptors/VLAD
  python luo_baseline.py --stage classify        # re-use cached VLAD
  python run_all.py                              # all three normalisations, clean + leaky
================================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "classification"))   # gpu_boost, cnt_paths
import gpu_boost
import cnt_paths

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- configuration
# Feature caches (the 73,600-D VLAD matrices, ~0.5 GB each) go under CNT_BASE;
# the small result JSONs go to <repo>/baseline/results by default.
DEFAULTS = dict(
    splits=cnt_paths.splits_file(),
    out_dir=None,                          # -> <CNT_BASE>/Luo Baseline
    results_dir=HERE / "results",
    img_size=224,
    vgg_relu_indices=(3, 8, 13, 22, 29),   # b1c2, b2c2, b3c2, b4c3, b5c3 (post-ReLU)
    hypercolumn_dim=1472,
    density=0.10,
    n_clusters=50,
    dict_sample_per_image=250,             # descriptors/image used to fit K-means
    n_folds=5,
    seed=42,
    batch_size=32,
)

CATEGORY_TO_ID = {"Fiber": 0, "Cluster": 1, "Matrix": 2, "MatrixSurface": 3}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}
CLASS_NAMES = [ID_TO_CATEGORY[i] for i in range(4)]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def banner(msg: str) -> None:
    log("\n" + "=" * 78)
    log(msg)
    log("=" * 78)


# ---------------------------------------------------------------- splits
def load_splits(pkl_path: Path, root_remap: str | None = None) -> dict:
    """Load the exact same split file our DINOv2 runs use.

    Image paths are rebuilt from `category` + `filename` under CNT_BASE (the
    absolute paths stored in the pickle belong to the machine it was made on
    and are ignored).  `root_remap` is accepted and ignored.
    `category_id` is stored as a STRING in some files; cast defensively.
    """
    with open(pkl_path, "rb") as fh:
        raw = pickle.load(fh)

    out = {}
    for key in ("train_df", "val_df", "test_df"):
        if key not in raw:
            raise KeyError(f"{pkl_path} is missing '{key}'")
        recs = raw[key]
        if hasattr(recs, "to_dict"):          # a real DataFrame
            recs = recs.to_dict("records")
        rows = []
        for r in recs:
            cat = r["category"]
            cid = r.get("category_id", CATEGORY_TO_ID[cat])
            cid = int(cid)
            if CATEGORY_TO_ID[cat] != cid:
                raise ValueError(
                    f"label mismatch: category={cat} but category_id={cid}. "
                    "CATEGORY_TO_ID in this script must match the notebook."
                )
            fn = r.get("filename") or Path(str(r["image_path"])).name
            p = cnt_paths.image_path(cat, fn)
            if not p.exists():
                raise SystemExit(f"image not found: {p}\n  set CNT_BASE to the data root "
                                 f"(see classification/cnt_paths.py)")
            rows.append({"image_path": str(p), "y": cid, "base_name": Path(fn).stem})
        out[key.replace("_df", "")] = rows

    # integrity: no image may appear in more than one split
    seen = {}
    for split, rows in out.items():
        for r in rows:
            prev = seen.get(r["image_path"])
            if prev is not None:
                raise ValueError(f"LEAK: {r['image_path']} in both {prev} and {split}")
            seen[r["image_path"]] = split
    return out


def describe_splits(splits: dict) -> None:
    total = sum(len(v) for v in splits.values())
    log(f"  total images: {total}")
    for name in ("train", "val", "test"):
        rows = splits[name]
        if not rows:
            log(f"  {name:5s} n=0   (EMPTY)")
            continue
        counts = np.bincount([r["y"] for r in rows], minlength=4)
        dist = "  ".join(f"{ID_TO_CATEGORY[i]}={counts[i]}" for i in range(4))
        log(f"  {name:5s} n={len(rows):5d}   {dist}")




# ---------------------------------------------------------------- progress
class Progress:
    """Writes a machine-readable progress file plus a live console line.

    Each job (one invocation of this script) occupies [offset, offset+weight]
    of the overall plan, so run_all.py can drive a single global percentage
    across many invocations without any read-modify-write races.
    """

    # within-job stage weights, calibrated from measured component costs
    STAGE_W = {"passA": 0.22, "kmeans": 0.16, "passB": 0.32,
               "cv": 0.24, "final": 0.06}

    def __init__(self, path, label, offset=0.0, weight=1.0, stages=None):
        self.path = Path(path) if path else None
        self.label = label
        self.offset = float(offset)
        self.weight = float(weight)
        act = stages or list(self.STAGE_W)
        tot = sum(self.STAGE_W[s] for s in act)
        self.w = {s: self.STAGE_W[s] / tot for s in act}
        self.done = {s: 0.0 for s in act}
        self.stage = None
        self.n = 0
        self.k = 0
        self.t0 = time.time()
        self.t_stage = self.t0
        self._last = 0.0

    def stage_start(self, stage, n_units, note=""):
        self.stage = stage
        self.n = max(int(n_units), 1)
        self.k = 0
        self.t_stage = time.time()
        self.note = note
        self._flush(force=True)

    def tick(self, k=1):
        self.k = min(self.k + k, self.n)
        if self.stage:
            self.done[self.stage] = self.k / self.n
        self._flush()

    def stage_end(self):
        if self.stage:
            self.done[self.stage] = 1.0
            self.k = self.n
        self._flush(force=True)

    @property
    def job_frac(self):
        return sum(self.w[s] * self.done[s] for s in self.w)

    def _flush(self, force=False):
        now = time.time()
        if not force and now - self._last < 1.0:
            return
        self._last = now
        jf = self.job_frac
        overall = self.offset + self.weight * jf
        el = now - self.t0
        eta_job = (el / jf - el) if jf > 1e-6 else float("nan")
        # stage-local rate gives a much better short-horizon estimate
        el_s = now - self.t_stage
        eta_stage = (el_s / self.k * (self.n - self.k)) if self.k > 0 else float("nan")
        pct = 100 * overall
        line = (f"\r    [{self.stage or '-':6s}] {self.k}/{self.n}  "
                f"job {100*jf:5.1f}%  overall {pct:5.1f}%  "
                f"stage ETA {self._fmt(eta_stage)}  job ETA {self._fmt(eta_job)}   ")
        sys.stdout.write(line)
        sys.stdout.flush()
        if self.path:
            payload = dict(
                label=self.label, stage=self.stage, note=getattr(self, "note", ""),
                units_done=self.k, units_total=self.n,
                job_percent=round(100 * jf, 2),
                overall_percent=round(pct, 2),
                elapsed_s=round(el, 1),
                stage_eta_s=None if eta_stage != eta_stage else round(eta_stage, 1),
                job_eta_s=None if eta_job != eta_job else round(eta_job, 1),
                stages_done={k: round(v, 3) for k, v in self.done.items()},
                updated=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            tmp = self.path.with_suffix(".tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "w") as fh:
                    json.dump(payload, fh, indent=2)
                tmp.replace(self.path)          # atomic
            except Exception:
                pass

    @staticmethod
    def _fmt(x):
        if x != x or x in (float("inf"),):
            return "  --  "
        x = int(x)
        return f"{x//60:3d}m{x%60:02d}s"

    def close(self, status="done"):
        for s in self.done:
            self.done[s] = 1.0
        self._flush(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()


# ---------------------------------------------------------------- feature stage
def build_vgg(device, relu_indices):
    import torch
    import torchvision as tv

    import os
    if os.environ.get("LUO_TEST_RANDOM_VGG") == "1":
        log("  !! LUO_TEST_RANDOM_VGG=1 -> RANDOM weights. Smoke test only.")
        model = tv.models.vgg16(weights=None).features.to(device).eval()
    else:
        weights = tv.models.VGG16_Weights.IMAGENET1K_V1
        model = tv.models.vgg16(weights=weights).features.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    chans = [model[i - 1].out_channels for i in relu_indices]
    assert sum(chans) == DEFAULTS["hypercolumn_dim"], (
        f"channel sum {sum(chans)} != 1472; relu_indices are wrong"
    )
    log(f"  VGG-16 taps {list(relu_indices)} -> channels {chans} (sum {sum(chans)})")
    return model, chans


BLOCK_BOUNDS = (0, 64, 192, 448, 960, 1472)   # cumulative channels of the 5 taps


def apply_desc_norm(d, mode):
    """d: (B, N, 1472). Normalisation of local descriptors before K-means/VLAD.

    'none'    raw VGG activations, as literally described by Luo et al.
    'l2'      L2-normalise the whole 1472-D descriptor (standard VLAD practice).
              NOTE: this does NOT equalise the per-block contributions -- a
              single scalar leaves the block ratios untouched.
    'blockl2' L2-normalise each of the 5 layer blocks, then the concatenation.
              This is what actually equalises block scales, so that block5 does
              not dominate the dictionary simply by having larger activations.
    """
    import torch
    import torch.nn.functional as F

    if mode == "none":
        return d
    if mode == "l2":
        return F.normalize(d, dim=-1)
    if mode == "blockl2":
        parts = []
        for a, b in zip(BLOCK_BOUNDS[:-1], BLOCK_BOUNDS[1:]):
            parts.append(F.normalize(d[..., a:b], dim=-1))
        return F.normalize(torch.cat(parts, dim=-1), dim=-1)
    raise ValueError(f"unknown desc_norm {mode!r}")


def load_batch(paths, img_size, device):
    """Greyscale -> 224x224 -> 3-channel -> ImageNet normalisation."""
    import torch
    from PIL import Image

    arr = np.empty((len(paths), img_size, img_size), dtype=np.float32)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            # 16-bit / float TIFFs must not go through convert("L"), which
            # truncates rather than rescales. Normalise by the actual range.
            if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
                a = np.asarray(im, dtype=np.float32)
                lo, hi = float(a.min()), float(a.max())
                a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
                im = Image.fromarray((a * 255).astype(np.uint8), mode="L")
            elif im.mode != "L":
                im = im.convert("L")
            im = im.resize((img_size, img_size), Image.BILINEAR)
            arr[i] = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).to(device)              # (B, H, W)
    t = t.unsqueeze(1).repeat(1, 3, 1, 1)             # (B, 3, H, W)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (t - mean) / std


def sample_coords(n_images, n_pixels, img_size, seed, offset):
    """Random pixel coordinates per image -> grid_sample grid in [-1, 1].

    align_corners=False convention: input pixel centre x maps to
        gx = 2*(x + 0.5)/img_size - 1
    The SAME grid is valid for every feature map regardless of its resolution,
    which is what makes this equivalent to (but far cheaper than) upsampling
    every map to 224x224 and indexing.
    """
    import torch

    rng = np.random.default_rng(seed + offset)
    total = img_size * img_size
    flat = np.empty((n_images, n_pixels), dtype=np.int64)
    for i in range(n_images):
        flat[i] = rng.choice(total, size=n_pixels, replace=False)
    ys, xs = np.divmod(flat, img_size)
    gx = 2.0 * (xs + 0.5) / img_size - 1.0
    gy = 2.0 * (ys + 0.5) / img_size - 1.0
    grid = np.stack([gx, gy], axis=-1).astype(np.float32)   # (B, N, 2)
    return torch.from_numpy(grid).unsqueeze(2)              # (B, N, 1, 2)


def extract_descriptors(model, relu_indices, imgs, grid, use_amp):
    """Return (B, N, 1472) descriptors sampled at `grid`, without ever
    materialising a full-resolution hypercolumn tensor."""
    import torch
    import torch.nn.functional as F

    taps = set(relu_indices)
    last = max(relu_indices)
    feats = []
    h = imgs
    dev_type = "cuda" if imgs.is_cuda else "cpu"
    with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_amp):
        for i, layer in enumerate(model):
            h = layer(h)
            if i in taps:
                # D9: grid stays fp32. Downcasting to fp16 costs 0.02 px of
                # coordinate precision and buys nothing; grid_sample is not
                # the bottleneck and fp32 keeps CPU and GPU runs identical.
                s = F.grid_sample(h.float(), grid, mode="bilinear",
                                  padding_mode="border", align_corners=False)
                feats.append(s.squeeze(-1))               # (B, C, N)
            if i >= last:
                break
    d = torch.cat(feats, dim=1)                           # (B, 1472, N)
    return d.permute(0, 2, 1).contiguous().float()        # (B, N, 1472)


def _dist2_to(X, v):
    """||X - v||^2 without materialising an (n, D) temporary."""
    import torch
    return (X * X).sum(1) - 2.0 * (X @ v) + (v * v).sum()


def _kmeans_once(X, k, g, iters, tol_rel):
    import torch

    n = X.shape[0]
    # --- k-means++ seeding (D^2 sampling), fully seeded via generator `g`
    idx = torch.randperm(n, generator=g)[:1].to(X.device)
    C = X[idx].clone()
    d2 = _dist2_to(X, C[0])
    for _ in range(1, k):
        probs = torch.clamp(d2, min=0)
        tot = probs.sum()
        probs = probs / tot if tot > 0 else torch.full_like(probs, 1.0 / n)
        nxt = torch.multinomial(probs.cpu(), 1, generator=g).to(X.device)  # D1: seeded
        C = torch.cat([C, X[nxt]], 0)
        d2 = torch.minimum(d2, _dist2_to(X, X[nxt][0]))

    # --- Lloyd
    scale = X.norm(dim=1).mean().item()
    tol = tol_rel * max(scale, 1e-12)
    converged = False
    for it in range(iters):
        assign = torch.empty(n, dtype=torch.long, device=X.device)
        for s0 in range(0, n, 65536):
            e0 = min(s0 + 65536, n)
            assign[s0:e0] = torch.cdist(X[s0:e0], C).argmin(1)
        newC = torch.zeros_like(C)
        newC.index_add_(0, assign, X)
        cnt = torch.bincount(assign, minlength=k)
        newC = newC / cnt.clamp(min=1).unsqueeze(1).to(X.dtype)
        empty = cnt == 0
        if empty.any():
            newC[empty] = X[torch.randperm(n, generator=g)[: int(empty.sum())].to(X.device)]
        shift = (newC - C).norm(dim=1).max().item()
        C = newC
        if shift < tol:
            converged = True
            break
    # inertia, for best-of-n_init selection
    inertia = 0.0
    for s0 in range(0, n, 65536):
        e0 = min(s0 + 65536, n)
        inertia += torch.cdist(X[s0:e0], C).min(1).values.pow(2).sum().item()
    return C, inertia, converged, it + 1


def gpu_kmeans(X, k, iters=100, seed=42, n_init=10, tol_rel=1e-4, prog=None):
    """K-means with k-means++ seeding, n_init restarts, best-inertia selection.

    D1: every RNG draw goes through the seeded generator `g`.  The seeding is
        exact, but the descriptors it clusters come from a cuDNN-autotuned VGG
        forward pass, so repeated GPU runs differ at the last bits and the
        dictionary is reproducible in distribution, not bit for bit (about
        half a point of CV accuracy; see README.md, "Run-to-run variation").
    D2: n_init restarts (sklearn's default is 10) and an explicit warning if a
        restart exhausts `iters` without converging. A single unlucky init used
        to be able to hand the baseline a bad dictionary.
    """
    import torch

    best = None
    for r in range(n_init):
        g = torch.Generator(device="cpu").manual_seed(seed + 1000 * r)
        C, inertia, conv, nit = _kmeans_once(X, k, g, iters, tol_rel)
        flag = "converged" if conv else "NOT CONVERGED"
        log(f"    init {r + 1}/{n_init}: inertia {inertia:.4e}  {flag} after {nit} iters")
        if not conv:
            log(f"      WARNING: restart {r + 1} hit the {iters}-iteration cap.")
        if best is None or inertia < best[1]:
            best = (C, inertia)
        if prog: prog.tick()
    log(f"    best inertia {best[1]:.4e}")
    return best[0]


def vlad_encode(desc, C, post_norm=True):
    """desc (B, N, D), C (K, D) -> (B, K*D), computed for the WHOLE batch at once.

    This was a Python loop over images, each issuing its own cdist and
    index_add_. At batch 32 that is 32 round trips per batch where one will do.

    V_k = sum_{i : a(i)=k} (x_i - c_k) = (sum of assigned x) - n_k * c_k
    """
    import torch

    B, N, D = desc.shape
    K = C.shape[0]
    flat = desc.reshape(-1, D)                          # (B*N, D)
    a = torch.cdist(flat, C).argmin(1)                  # (B*N,)
    img = torch.arange(B, device=desc.device).repeat_interleave(N)
    lin = img * K + a                                   # unique bin per (image, centroid)
    V = torch.zeros(B * K, D, device=desc.device, dtype=flat.dtype)
    V.index_add_(0, lin, flat)
    cnt = torch.bincount(lin, minlength=B * K).to(flat.dtype)
    V = V.view(B, K, D) - cnt.view(B, K, 1) * C.unsqueeze(0)
    v = V.reshape(B, K * D)
    if post_norm:
        v = torch.sign(v) * torch.sqrt(torch.abs(v))
        n = v.norm(dim=1, keepdim=True).clamp(min=1e-12)
        v = v / n
    return v.float()


def run_feature_stage(cfg, splits, device, prog=None):
    import torch

    banner("STAGE 1-3  |  VGG-16 hypercolumns -> K-means dictionary -> VLAD")
    model, _ = build_vgg(device, cfg.vgg_relu_indices)
    n_pix = int(round(cfg.density * cfg.img_size * cfg.img_size))
    if cfg.dict_sample_per_image > n_pix:
        raise ValueError(
            f"dict_sample_per_image ({cfg.dict_sample_per_image}) > descriptors "
            f"per image ({n_pix}); lower it or raise --density")
    log(f"  density {cfg.density:.0%} -> {n_pix} descriptors/image "
        f"({cfg.img_size}x{cfg.img_size} = {cfg.img_size**2} px)")

    order = ["train", "val", "test"]
    all_rows, split_of = [], []
    for s in order:
        all_rows += splits[s]
        split_of += [s] * len(splits[s])
    split_of = np.array(split_of)
    paths = [r["image_path"] for r in all_rows]
    y = np.array([r["y"] for r in all_rows], dtype=np.int64)

    use_amp = device.type == "cuda"

    # ---- PASS A: subsample descriptors from TRAIN ONLY, to fit the dictionary
    log("\n  Pass A: collecting dictionary-fitting descriptors (train split only)")
    if prog: prog.stage_start("passA", len(np.where(split_of == "train")[0]), "VGG on train")
    tr_idx = np.where(split_of == "train")[0]
    keep = cfg.dict_sample_per_image
    pool = torch.empty(len(tr_idx) * keep, cfg.hypercolumn_dim,
                       device=device, dtype=torch.float16 if use_amp else torch.float32)
    rng = np.random.default_rng(cfg.seed)
    t0, w = time.time(), 0
    with torch.no_grad():
        for s in range(0, len(tr_idx), cfg.batch_size):
            bidx = tr_idx[s:s + cfg.batch_size]
            imgs = load_batch([paths[i] for i in bidx], cfg.img_size, device)
            grid = sample_coords(len(bidx), n_pix, cfg.img_size, cfg.seed, s).to(device)
            d = extract_descriptors(model, cfg.vgg_relu_indices, imgs, grid, use_amp)
            d = apply_desc_norm(d, cfg.desc_norm)
            sel = torch.from_numpy(
                np.stack([rng.choice(n_pix, keep, replace=False) for _ in range(len(bidx))])
            ).to(device)
            picked = torch.gather(d, 1, sel.unsqueeze(-1).expand(-1, -1, d.shape[-1]))
            picked = picked.reshape(-1, d.shape[-1]).to(pool.dtype)
            pool[w:w + picked.shape[0]] = picked
            w += picked.shape[0]
            if prog: prog.tick(len(bidx))
    pool = pool[:w]
    log(f"  dictionary pool: {tuple(pool.shape)} {pool.dtype}  ({time.time() - t0:.1f}s)")
    if pool.dtype != torch.float32:
        # D6: convert in place-ish, chunk by chunk, so the fp16 and fp32 copies
        # are never both fully resident (this OOMed a 16 GB card at 5,323 images)
        out = torch.empty(pool.shape, device=pool.device, dtype=torch.float32)
        for s0 in range(0, pool.shape[0], 65536):
            out[s0:s0 + 65536] = pool[s0:s0 + 65536].float()
        del pool
        if use_amp:
            torch.cuda.empty_cache()
        pool = out

    if prog: prog.stage_end()
    log(f"\n  Fitting K-means (K={cfg.n_clusters}) on TRAIN descriptors only")
    if prog: prog.stage_start("kmeans", cfg.n_init, f"K={cfg.n_clusters}, n_init={cfg.n_init}")
    t0 = time.time()
    C = gpu_kmeans(pool, cfg.n_clusters, seed=cfg.seed, n_init=cfg.n_init, prog=prog)
    log(f"  centroids {tuple(C.shape)}  ({time.time() - t0:.1f}s)")
    del pool
    if use_amp:
        torch.cuda.empty_cache()

    # ---- PASS B: VLAD-encode every image
    if prog: prog.stage_end()
    log("\n  Pass B: VLAD encoding all images")
    if prog: prog.stage_start("passB", len(paths), "VGG + VLAD, all images")
    dim = cfg.n_clusters * cfg.hypercolumn_dim
    X = np.empty((len(paths), dim), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(paths), cfg.batch_size):
            e = min(s + cfg.batch_size, len(paths))
            imgs = load_batch(paths[s:e], cfg.img_size, device)
            grid = sample_coords(e - s, n_pix, cfg.img_size, cfg.seed, 10_000 + s).to(device)
            d = extract_descriptors(model, cfg.vgg_relu_indices, imgs, grid, use_amp)
            d = apply_desc_norm(d, cfg.desc_norm)
            X[s:e] = vlad_encode(d, C, cfg.vlad_postnorm).cpu().numpy()
            if prog: prog.tick(e - s)
    if prog: prog.stage_end()
    log(f"  VLAD matrix {X.shape}  {X.nbytes / 1e6:.0f} MB  ({time.time() - t0:.1f}s)")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_dir / f"featkey_{cfg.tag}.txt", "w") as fh:
        fh.write(cfg.feat_key)
    np.save(cfg.cache_x, X)
    np.save(cfg.cache_y, y)
    np.save(cfg.cache_s, split_of)
    with open(cfg.out_dir / f"centroids_{cfg.tag}.npy", "wb") as fh:
        np.save(fh, C.cpu().numpy())
    log(f"  cached -> {cfg.cache_x.name}")
    return X, y, split_of


# ---------------------------------------------------------------- classifier
def make_booster(cfg):
    import xgboost as xgb

    params = dict(
        objective="multi:softprob",
        num_class=4,
        tree_method="hist",
        max_bin=cfg.max_bin,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=cfg.colsample,
        min_child_weight=1.0,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=cfg.seed,
    )
    if cfg.device == "cuda":
        params = gpu_boost.xgb_params(params, "cuda",
                                      verbose=not getattr(make_booster, "_said", False))
        make_booster._said = True
    return params


def fit_eval_xgb(cfg, Xfit, yfit, Xes, yes_, Xev, X_refit=None, y_refit=None):
    """Two-stage fit.

    Stage 1 selects the number of boosting rounds using (Xes, yes_).
    Stage 2 (clean mode only) REFITS on the full training data for exactly that
    many rounds, with early stopping off.

    D3: without stage 2, the clean protocol trained on 10% less data than the
    leaky one, so the clean-vs-leaky difference measured the leak *plus* a
    training-set-size penalty, and the publishable number was handicapped
    against a comparator fit on all of train+val.
    """
    import xgboost as xgb

    params = make_booster(cfg)
    dfit = xgb.QuantileDMatrix(Xfit, label=yfit, max_bin=cfg.max_bin)
    des = xgb.QuantileDMatrix(Xes, label=yes_, ref=dfit, max_bin=cfg.max_bin)
    bst = xgb.train(
        params, dfit,
        num_boost_round=cfg.n_rounds,
        evals=[(des, "es")],
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose_eval=False,
    )
    n_best = bst.best_iteration + 1

    if X_refit is not None:
        dfull = xgb.QuantileDMatrix(X_refit, label=y_refit, max_bin=cfg.max_bin)
        bst = xgb.train(params, dfull, num_boost_round=n_best, verbose_eval=False)
        proba = bst.predict(xgb.DMatrix(Xev))
    else:
        proba = bst.predict(xgb.DMatrix(Xev), iteration_range=(0, n_best))
    return proba.argmax(1), n_best


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def run_classify_stage(cfg, X, y, split_of, prog=None):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

    banner("STAGE 4  |  Gradient-boosted softmax on VLAD features")
    log(f"  feature matrix: {X.shape}   ({X.nbytes / 1e6:.0f} MB)")
    log(f"  early stopping: {cfg.earlystop.upper()}"
        + ("   <-- scored set used as the stopping monitor, NOT for publication"
           if cfg.earlystop == "leaky" else "   (test set touched once)"))

    tr = np.where(split_of == "train")[0]
    va = np.where(split_of == "val")[0]
    te = np.where(split_of == "test")[0]
    trval = np.concatenate([tr, va])
    X_tv, y_tv = X[trval], y[trval]
    X_te, y_te = X[te], y[te]

    results = {}

    # ---------- PRIMARY: 5-fold CV on train+val (matches S3 and Luo's protocol)
    log(f"\n  {cfg.n_folds}-fold StratifiedKFold on train+val "
        f"(n={len(y_tv)}, shuffle=True, random_state={cfg.seed})")
    log("  NOTE: the K-means dictionary is fitted ONCE on the train split, so the")
    log("        CV folds are scored under a dictionary that has seen most of their")
    log("        own images. This is an unsupervised representation fitted without")
    log("        labels, and the bias runs IN FAVOUR of the baseline, but the CV")
    log("        number is therefore not fully nested. The held-out test number")
    log("        below is clean: no test image enters the dictionary or the fit.")
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    accs, f1s, rounds = [], [], []
    if prog: prog.stage_start("cv", cfg.n_folds, f"{cfg.n_folds}-fold CV")
    t0 = time.time()
    for k, (i_tr, i_va) in enumerate(skf.split(X_tv, y_tv), 1):
        t1 = time.time()
        if cfg.earlystop == "leaky":
            # reproduce the published protocol exactly: the fold that is scored
            # is also the fold used to pick the stopping round. No refit.
            pred, nr = fit_eval_xgb(cfg, X_tv[i_tr], y_tv[i_tr],
                                    X_tv[i_va], y_tv[i_va], X_tv[i_va])
        else:
            # inner split picks the round count; then refit on ALL of i_tr
            i_fit, i_es = train_test_split(
                i_tr, test_size=0.1, stratify=y_tv[i_tr], random_state=cfg.seed
            )
            pred, nr = fit_eval_xgb(cfg, X_tv[i_fit], y_tv[i_fit],
                                    X_tv[i_es], y_tv[i_es], X_tv[i_va],
                                    X_refit=X_tv[i_tr], y_refit=y_tv[i_tr])
        a = 100 * accuracy_score(y_tv[i_va], pred)
        f = f1_score(y_tv[i_va], pred, average="macro")
        accs.append(a); f1s.append(f); rounds.append(nr)
        if prog:
            sys.stdout.write("\r" + " " * 110 + "\r")
        log(f"    fold {k}/{cfg.n_folds}: acc {a:6.2f}%  F1 {f:.4f}  "
            f"({nr} rounds, {time.time() - t1:.1f}s)")
        if prog: prog.tick()
    results["cv_mean_acc"] = float(np.mean(accs))
    results["cv_std_acc"] = float(np.std(accs))
    results["cv_mean_f1"] = float(np.mean(f1s))
    results["cv_std_f1"] = float(np.std(f1s))
    results["cv_time_min"] = (time.time() - t0) / 60
    log(f"\n  CV accuracy : {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
    log(f"  CV macro F1 : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    log(f"  CV wall time: {results['cv_time_min']:.1f} min")

    # ---------- SECONDARY: held-out test
    if prog: prog.stage_end()
    log("\n  Final model: fit on train+val -> evaluate held-out test")
    if prog: prog.stage_start("final", 1, "train+val -> test")
    t0 = time.time()
    if cfg.earlystop == "leaky":
        # the bug, reproduced on purpose: the test set picks the stopping round
        pred, nr = fit_eval_xgb(cfg, X_tv, y_tv, X_te, y_te, X_te)
    else:
        i_fit, i_es = train_test_split(
            np.arange(len(y_tv)), test_size=0.1, stratify=y_tv, random_state=cfg.seed
        )
        pred, nr = fit_eval_xgb(cfg, X_tv[i_fit], y_tv[i_fit],
                                X_tv[i_es], y_tv[i_es], X_te,
                                X_refit=X_tv, y_refit=y_tv)
    acc = 100 * accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    n_correct = int(round(acc / 100 * len(y_te)))
    lo, hi = wilson_ci(n_correct, len(y_te))
    results.update(
        test_acc=float(acc), test_f1=float(f1),
        test_n=int(len(y_te)), test_correct=n_correct,
        test_ci_low=lo, test_ci_high=hi,
        final_rounds=int(nr), final_time_min=(time.time() - t0) / 60,
    )
    if prog:
        prog.stage_end()
        sys.stdout.write("\r" + " " * 110 + "\r")
    log(f"    test accuracy: {acc:.2f}%  ({n_correct}/{len(y_te)})  "
        f"95% CI [{lo:.1f}, {hi:.1f}]")
    log(f"    test macro F1: {f1:.4f}   ({nr} rounds, {results['final_time_min']*60:.1f}s)")
    log("\n  Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_te, pred, labels=list(range(4)))
    log("            " + "".join(f"{c[:6]:>9}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        log(f"    {CLASS_NAMES[i][:10]:<10}" + "".join(f"{v:>9}" for v in row))
    log("\n" + classification_report(y_te, pred, target_names=CLASS_NAMES, digits=3))
    results["confusion_matrix"] = cm.tolist()
    return results


# ---------------------------------------------------------------- main
@dataclass
class Cfg:
    feat_key: str
    splits: Path
    out_dir: Path
    tag: str
    img_size: int
    vgg_relu_indices: tuple
    hypercolumn_dim: int
    density: float
    n_clusters: int
    dict_sample_per_image: int
    n_folds: int
    n_init: int
    seed: int
    batch_size: int
    desc_norm: str
    vlad_postnorm: bool
    earlystop: str
    device: str
    n_rounds: int
    early_stopping_rounds: int
    max_bin: int
    colsample: float

    @property
    def cache_x(self): return self.out_dir / f"vlad_X_{self.tag}.npy"
    @property
    def cache_y(self): return self.out_dir / f"vlad_y_{self.tag}.npy"
    @property
    def cache_s(self): return self.out_dir / f"vlad_split_{self.tag}.npy"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Luo et al. (2021) baseline on our splits")
    ap.add_argument("--stage", choices=["all", "features", "classify"], default="all")
    ap.add_argument("--splits", type=Path, default=DEFAULTS["splits"])
    ap.add_argument("--out-dir", type=Path, default=DEFAULTS["out_dir"],
                    help="VLAD feature caches (default <CNT_BASE>/Luo Baseline)")
    ap.add_argument("--results-dir", type=Path, default=DEFAULTS["results_dir"],
                    help="where luo_results_*.json go (default <repo>/baseline/results)")
    ap.add_argument("--tag", default=None, help="cache/result suffix (default: from options)")
    ap.add_argument("--desc-norm", choices=["none", "l2", "blockl2"], default="l2")
    ap.add_argument("--no-vlad-postnorm", action="store_true")
    ap.add_argument("--earlystop", choices=["clean", "leaky"], default="clean")
    ap.add_argument("--density", type=float, default=DEFAULTS["density"])
    ap.add_argument("--n-clusters", type=int, default=DEFAULTS["n_clusters"])
    ap.add_argument("--n-init", type=int, default=10,
                    help="K-means restarts; best inertia wins (sklearn default is 10)")
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"],
                    help="32 needs ~2 GB VRAM for descriptors; drop to 16 if OOM")
    ap.add_argument("--n-rounds", type=int, default=400)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    ap.add_argument("--max-bin", type=int, default=64)
    ap.add_argument("--colsample", type=float, default=0.3)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--progress-file", default=None,
                    help="JSON file continuously updated with %% complete and ETA")
    ap.add_argument("--job-offset", type=float, default=0.0)
    ap.add_argument("--job-weight", type=float, default=1.0)
    ap.add_argument("--job-label", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: keep only N images per split")
    ap.add_argument("--root-remap", default=None, help=argparse.SUPPRESS)  # obsolete
    args = ap.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = cnt_paths.base() / "Luo Baseline"

    import torch

    torch.manual_seed(DEFAULTS["seed"])
    np.random.seed(DEFAULTS["seed"])
    gpu_boost.setup()
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    feat_key = (f"{Path(args.splits).stem}|{args.desc_norm}|dens{args.density}"
                f"|K{args.n_clusters}|post{not args.no_vlad_postnorm}|lim{args.limit}")
    feat_hash = hashlib.sha1(feat_key.encode()).hexdigest()[:8]
    tag = args.tag or f"{Path(args.splits).stem}_{args.desc_norm}_{feat_hash}"
    cfg = Cfg(
        feat_key=feat_key,
        splits=args.splits, out_dir=args.out_dir, tag=tag,
        img_size=DEFAULTS["img_size"], vgg_relu_indices=DEFAULTS["vgg_relu_indices"],
        hypercolumn_dim=DEFAULTS["hypercolumn_dim"], density=args.density,
        n_clusters=args.n_clusters,
        dict_sample_per_image=DEFAULTS["dict_sample_per_image"],
        n_folds=DEFAULTS["n_folds"], n_init=args.n_init, seed=DEFAULTS["seed"],
        batch_size=args.batch_size, desc_norm=args.desc_norm,
        vlad_postnorm=not args.no_vlad_postnorm, earlystop=args.earlystop,
        device=dev, n_rounds=args.n_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        max_bin=args.max_bin, colsample=args.colsample,
    )

    banner("LUO ET AL. (2021) BASELINE  |  matched splits, matched CV protocol")
    log(f"  device              : {dev}"
        + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    log(f"  splits file         : {cfg.splits}")
    log(f"  VLAD dimensionality : {cfg.n_clusters} x {cfg.hypercolumn_dim} "
        f"= {cfg.n_clusters * cfg.hypercolumn_dim:,}")
    log(f"  descriptor norm     : {cfg.desc_norm}    VLAD post-norm: {cfg.vlad_postnorm}")
    log(f"  augmentation        : NONE (classes within 7.4% of each other; Luo "
        f"augmented only to fix a large imbalance)")
    log(f"  tag                 : {cfg.tag}")

    banner("SPLITS")
    splits = load_splits(cfg.splits, args.root_remap)
    if args.limit:
        log(f"  !! --limit {args.limit}: truncating each split (DEBUG ONLY)")
        rs = np.random.default_rng(cfg.seed)
        for k in splits:
            rows = splits[k]
            idx = rs.permutation(len(rows))[: args.limit]
            splits[k] = [rows[i] for i in sorted(idx)]
    describe_splits(splits)

    stages = (["passA", "kmeans", "passB"] if args.stage in ("all", "features") else []) \
             + (["cv", "final"] if args.stage in ("all", "classify") else [])
    prog = Progress(args.progress_file, args.job_label or cfg.tag,
                    args.job_offset, args.job_weight, stages) if args.progress_file else None

    if args.stage in ("all", "features"):
        X, y, split_of = run_feature_stage(cfg, splits, device, prog)
    else:
        keyf = cfg.out_dir / f"featkey_{cfg.tag}.txt"
        if keyf.exists():
            cached = keyf.read_text().strip()
            if cached != cfg.feat_key:
                raise SystemExit(
                    f"cached features were built with a DIFFERENT config:\n"
                    f"  cached: {cached}\n  now   : {cfg.feat_key}\n"
                    f"re-run with --stage all")
        X = np.load(cfg.cache_x); y = np.load(cfg.cache_y)
        split_of = np.load(cfg.cache_s, allow_pickle=True)
        log(f"\n  loaded cached VLAD: {X.shape}")

    if args.stage == "features":
        if prog: prog.close()
        log("\nfeature stage complete.")
        return

    res = run_classify_stage(cfg, X, y, split_of, prog)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: (Path(v).name if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    payload.update(res)
    rtag = cfg.tag if cfg.earlystop in cfg.tag else f"{cfg.tag}_{cfg.earlystop}"
    out = args.results_dir / f"luo_results_{rtag}.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)

    if prog: prog.close()
    banner("SUMMARY")
    log(f"  Luo et al. re-implementation, {Path(cfg.splits).stem}")
    log(f"    5-fold CV accuracy (train+val) : {res['cv_mean_acc']:.2f}% "
        f"+/- {res['cv_std_acc']:.2f}%")
    log(f"    Held-out test accuracy         : {res['test_acc']:.2f}% "
        f"({res['test_correct']}/{res['test_n']}, 95% CI "
        f"[{res['test_ci_low']:.1f}, {res['test_ci_high']:.1f}])")
    log(f"    Held-out test macro F1         : {res['test_f1']:.4f}")
    log(f"  Luo's own published figure       : 90.9% (5-fold CV mean, augmented, "
        f"5,323-image 4-class set)")
    log(f"\n  results -> {out}")


if __name__ == "__main__":
    main()
