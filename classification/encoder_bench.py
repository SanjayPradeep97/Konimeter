#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 Multi-encoder benchmark for mask-guided CNT morphology classification
================================================================================

Covers, in one table:
  newer encoders   -- newer foundation models
                      (DINOv3, DINOv2+registers, SAM 2.1)
  layer sampling   -- one fixed layer-sampling STRATEGY for every encoder,
                      rather than a layer set tuned for DINOv2
  controls         -- conventional transfer-learning controls
                      (ResNet-50, supervised ViT-B/16, ConvNeXt-V2)
  protocol         -- selection on CV, held-out test reported once, with
                      Wilson confidence intervals

Also defined, but NOT part of the reported benchmark: micro-SAM (`usam_b`,
microscopy-specific; needs USAM_CHECKPOINT) and I-JEPA (`ijepa_h14`, ViT-H
only, so not capacity-matched). Run them explicitly with --encoders.

THE POINT OF THIS SCRIPT
------------------------
Not "which encoder wins". Every encoder is run BOTH mask-guided and unmasked,
from the same forward pass. If mask-guided pooling helps across the board, the
contribution stops being "DINOv2 is good at CNTs" -- fragile, and overturned the
moment a better backbone appears -- and becomes "background suppression before
pooling is an architecture-independent gain". That claim survives whatever the
new numbers say.

FAIRNESS, DELIBERATELY
----------------------
1. MATCHED FEATURE GRIDS. At native resolutions the token grids span 196
   (CLIP/SigLIP/MAE at 224) to 4096 (SAM at 1024) -- a 20x range. Comparing
   there would measure input resolution, not encoders. Every ViT is therefore
   run at patch_size * TARGET_GRID so all produce the same token grid.
   Pyramid models (ResNet, ConvNeXt, Hiera) get a fixed input and their actual
   grids are printed in the results table.
2. NO PER-MODEL LAYER TUNING. Taps are evenly spaced across each model's depth.
   [1,3,6,9,11] was tuned for DINOv2 and reusing it elsewhere would hand DINOv2
   an advantage that has nothing to do with its features.
3. IDENTICAL SPLITS, FOLDS, SEEDS and classifier hyperparameters, loaded from
   the same pickle the DINOv2 runs use.
4. NO TEST-SET MODEL SELECTION. Early stopping uses an inner split; the test
   set is scored once per configuration. `--earlystop leaky` uses the scored
   set as the stopping monitor, only so that bias can be measured.

USAGE
-----
  python encoder_bench.py --list
  python encoder_bench.py --encoders dinov2_b14 dinov3_b16 resnet50
  python encoder_bench.py --group benchmark      # the eight-encoder benchmark
  python encoder_bench.py --group all
  python encoder_bench.py --stage classify       # reuse cached features
================================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import gpu_boost
import cnt_paths

# Pin the HF / torch download caches before timm is imported anywhere.  The
# data root comes from the CNT_BASE environment variable (see cnt_paths.py);
# it is only required once something actually touches images or caches.
import cache_setup as _cache_setup
if cnt_paths.base(required=False) is not None:
    _cache_setup.setup(cnt_paths.base(), verbose=False)
CATEGORY_TO_ID = {"Fiber": 0, "Cluster": 1, "Matrix": 2, "MatrixSurface": 3}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}
CLASS_NAMES = [ID_TO_CATEGORY[i] for i in range(4)]

# ---------------------------------------------------------------- roster
# group: 'paper'     = what the manuscript already uses (reference rows)
#        'reviewer'  = newer foundation models and transfer-learning controls
#                      (the label is stored in result rows, so it is kept)
#        'objective' = same ViT-B/16 capacity, different pretraining objective
ENCODERS = {
    # --- reference: what the paper uses today -------------------------------
    "dinov2_b14":  dict(timm="vit_base_patch14_dinov2.lvd142m",       group="paper",
                        note="paper's encoder"),
    "sam_b16":     dict(timm="samvit_base_patch16.sa1b",              group="paper",
                        note="paper's SAM encoder"),
    # --- newer foundation models and transfer-learning controls -------------
    "dinov3_b16":  dict(timm="vit_base_patch16_dinov3.lvd1689m",      group="reviewer",
                        note="newer self-supervised encoder (DINOv3)"),
    "dinov2_reg":  dict(timm="vit_base_patch14_reg4_dinov2.lvd142m",  group="reviewer",
                        note="registers remove the high-norm artifact tokens in Fig 6"),
    "sam2_hiera":  dict(timm="sam2_hiera_base_plus.fb_r896_2pt1",     group="reviewer",
                        note="modern segmentation encoder; successor to the 2023 SAM"),
    "resnet50":    dict(timm="resnet50.a1_in1k",                      group="reviewer",
                        note="conventional transfer-learning control"),
    "supvit_b16":  dict(timm="vit_base_patch16_224.augreg2_in21k_ft_in1k", group="reviewer",
                        note="supervised ViT (transfer-learning control)"),
    "convnextv2_b": dict(timm="convnextv2_base.fcmae_ft_in1k",        group="reviewer",
                        note="modern CNN control"),
    # --- same capacity, different pretraining objective ---------------------
    "clip_b16":    dict(timm="vit_base_patch16_clip_224.laion2b",     group="objective",
                        note="language-supervised contrastive"),
    "siglip2_b16": dict(timm="vit_base_patch16_siglip_224.v2_webli",  group="objective",
                        note="SigLIP 2, sigmoid language-image"),
    "mae_b16":     dict(timm="vit_base_patch16_224.mae",              group="objective",
                        note="masked-autoencoder reconstruction"),
    "usam_b":      dict(timm="samvit_base_patch16.sa1b",              group="reviewer",
                        note="microscopy-specific SAM (micro-SAM); not in the "
                             "reported benchmark. NOTE - set USAM_CHECKPOINT to the "
                             "micro-sam vit_b_em_organelles weights; its EM training "
                             "data (MitoLab/MitoEM/PlatyEM) contains no materials "
                             "imagery and its own docs send non-cellular tasks back "
                             "to stock SAM"),
    # --- capacity NOT matched: include only with the caveat printed ---------
    "ijepa_h14":   dict(timm="vit_huge_patch14_gap_224.in1k_ijepa",   group="unmatched",
                        note="I-JEPA; not in the reported benchmark. ViT-H/630M -- "
                             "no ViT-B release exists, "
                             "so this is NOT capacity-matched to the ViT-B rows"),
}

GROUPS = {
    "paper": ["dinov2_b14", "sam_b16"],
    "benchmark": ["dinov2_b14", "dinov3_b16", "dinov2_reg", "sam_b16", "sam2_hiera",
                  "resnet50", "supvit_b16", "convnextv2_b"],
    "usam": ["usam_b"],
    "objective": ["dinov2_b14", "clip_b16", "siglip2_b16", "mae_b16", "supvit_b16"],
    "all": [k for k in ENCODERS if k != "ijepa_h14"],
    "everything": list(ENCODERS),
}
# Earlier name for "benchmark", kept so existing commands still run.
GROUPS["reviewers"] = GROUPS["benchmark"]

POOLINGS = ("avg", "max", "avg+max")
CLASSIFIERS = ("linear", "mlp")
MASK_MODES = ("masked", "nomask")


def log(m=""):
    print(m, flush=True)


def banner(m):
    log("\n" + "=" * 78); log(m); log("=" * 78)


# ---------------------------------------------------------------- splits
def load_splits(pkl_path: Path, root: Path | None = None, root_remap=None):
    """Load the split file and resolve every image and mask on THIS machine.

    The pickle stores the absolute paths of the machine it was created on
    (`image_path`, `mask_path`).  Those are ignored: paths are rebuilt from
    `category` and `filename` under CNT_BASE (or `root`), so the same split
    file works anywhere.  Masks are looked up with cnt_paths.mask_path(), i.e.
    <CNT_BASE>/NIOSH Dataset/Masks/masks/<base>_mask.png unless CNT_MASKS is set.

    `root_remap` is accepted for backward compatibility and ignored.
    """
    import os
    with open(pkl_path, "rb") as fh:
        raw = pickle.load(fh)
    if root is not None:
        os.environ.setdefault("CNT_BASE", str(root))
    out, missing_img, missing_mask = {}, [], []
    for key in ("train_df", "val_df", "test_df"):
        recs = raw[key]
        if hasattr(recs, "to_dict"):
            recs = recs.to_dict("records")
        rows = []
        for r in recs:
            cat = r["category"]
            cid = int(r.get("category_id", CATEGORY_TO_ID[cat]))
            if CATEGORY_TO_ID[cat] != cid:
                raise ValueError(f"label mismatch: {cat} vs id {cid}")
            fn = r.get("filename") or Path(str(r["image_path"])).name
            ip = cnt_paths.image_path(cat, fn)
            mp = cnt_paths.mask_path(fn)
            if not ip.exists():
                missing_img.append(fn)
            if not mp.exists():
                missing_mask.append(fn)
            rows.append({"image": ip, "mask": mp, "y": cid})
        out[key.replace("_df", "")] = rows
    if missing_img:
        raise SystemExit(
            f"could not locate {len(missing_img)} images, e.g. {missing_img[:5]}.\n"
            f"  looked under {cnt_paths.images_dir()}/CNT-<class>/\n"
            f"  Download the dataset from https://doi.org/10.7910/DVN/5O0SF7 and set CNT_BASE.")
    if missing_mask:
        raise SystemExit(
            f"could not locate masks for {len(missing_mask)} images, e.g. {missing_mask[:5]}.\n"
            f"  looked under {cnt_paths.masks_dir()}\n"
            f"  Masks are distributed with the paper's data archive (see classification/REPRODUCE.md); set CNT_MASKS\n"
            f"  if they live elsewhere.")
    seen = {}
    for s_, rows in out.items():
        for r in rows:
            if r["image"] in seen:
                raise ValueError(f"LEAK: {r['image']} in {seen[r['image']]} and {s_}")
            seen[r["image"]] = s_
    return out




# ---------------------------------------------------------------- progress
class Progress:
    """Live console line + optional JSON file, always on.

    Encoder i of N occupies [i/N, (i+1)/N] of the overall bar, so the overall
    percentage and ETA span the whole run rather than the current encoder.
    """

    STAGE_W = {"extract": 0.55, "probe": 0.45}

    def __init__(self, path, n_jobs):
        self.path = Path(path) if path else None
        self.n_jobs = max(int(n_jobs), 1)
        self.job_i = 0
        self.label = ""
        self.done = {k: 0.0 for k in self.STAGE_W}
        self.stage = None
        self.n = 1
        self.k = 0
        self.t0 = time.time()
        self.t_stage = self.t0
        self._last = 0.0

    def job_start(self, i, label):
        self.job_i = i
        self.label = label
        self.done = {k: 0.0 for k in self.STAGE_W}

    def stage_start(self, stage, n_units, note=""):
        self.stage = stage
        self.n = max(int(n_units), 1)
        self.k = 0
        self.note = note
        self.t_stage = time.time()
        self._flush(True)

    def tick(self, k=1):
        self.k = min(self.k + k, self.n)
        if self.stage:
            self.done[self.stage] = self.k / self.n
        self._flush()

    def stage_end(self):
        if self.stage:
            self.done[self.stage] = 1.0
            self.k = self.n
        self._flush(True)

    @property
    def job_frac(self):
        tot = sum(self.STAGE_W.values())
        return sum(self.STAGE_W[s] * self.done[s] for s in self.done) / tot

    @property
    def overall(self):
        return (self.job_i - 1 + self.job_frac) / self.n_jobs if self.job_i else 0.0

    def _flush(self, force=False):
        now = time.time()
        if not force and now - self._last < 1.0:
            return
        self._last = now
        ov = self.overall
        el = now - self.t0
        eta_all = (el / ov - el) if ov > 1e-6 else float("nan")
        els = now - self.t_stage
        eta_stage = (els / self.k * (self.n - self.k)) if self.k > 0 else float("nan")
        bar_n = 28
        filled = int(round(bar_n * ov))
        bar = "#" * filled + "-" * (bar_n - filled)
        sys.stdout.write(
            f"\r  [{bar}] {100*ov:5.1f}%  job {self.job_i}/{self.n_jobs} "
            f"{self.label[:14]:14} {self.stage or '-':7} {self.k}/{self.n}  "
            f"stage {self._fmt(eta_stage)}  total left {self._fmt(eta_all)}  ")
        sys.stdout.flush()
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                with open(tmp, "w") as fh:
                    json.dump(dict(
                        encoder=self.label, job=self.job_i, jobs_total=self.n_jobs,
                        stage=self.stage, note=getattr(self, "note", ""),
                        units_done=self.k, units_total=self.n,
                        job_percent=round(100 * self.job_frac, 2),
                        overall_percent=round(100 * ov, 2),
                        elapsed_s=round(el, 1),
                        stage_eta_s=None if eta_stage != eta_stage else round(eta_stage, 1),
                        total_eta_s=None if eta_all != eta_all else round(eta_all, 1),
                        updated=time.strftime("%Y-%m-%d %H:%M:%S")), fh, indent=2)
                tmp.replace(self.path)
            except Exception:
                pass

    @staticmethod
    def _fmt(x):
        if x != x:
            return "  --  "
        x = int(x)
        h, m, s = x // 3600, (x % 3600) // 60, x % 60
        return f"{h}h{m:02d}m" if h else f"{m:2d}m{s:02d}s"

    def newline(self):
        sys.stdout.write("\r" + " " * 128 + "\r")
        sys.stdout.flush()

    def close(self):
        self.job_i = self.n_jobs
        self.done = {k: 1.0 for k in self.done}
        self._flush(True)
        sys.stdout.write("\n")


# ---------------------------------------------------------------- encoder
def build_encoder(key, target_grid, pyramid_img, device):
    import timm
    import torch

    spec = ENCODERS[key]
    cfgm = timm.create_model(spec["timm"], pretrained=False, num_classes=0)
    patch = getattr(getattr(cfgm, "patch_embed", None), "patch_size", None)
    native = cfgm.pretrained_cfg.get("input_size", (3, 224, 224))[-1]
    del cfgm

    if patch is not None:                       # plain ViT: exact grid control
        ps = patch[0] if isinstance(patch, (tuple, list)) else patch
        img_size = int(ps * target_grid)
        kw = dict(img_size=img_size)
    else:                                       # pyramid: fixed input
        img_size = int(pyramid_img)
        kw = {}

    import os
    pre = os.environ.get("ENCBENCH_NO_PRETRAINED") != "1"
    if not pre:
        log("    !! ENCBENCH_NO_PRETRAINED=1 -> RANDOM weights. Smoke test only.")
    ckpt = os.environ.get("USAM_CHECKPOINT") if key == "usam_b" else None
    try:
        m = timm.create_model(spec["timm"], pretrained=pre and not ckpt,
                              num_classes=0, **kw)
        if ckpt:
            import torch as _t
            sd = _t.load(ckpt, map_location="cpu")
            sd = sd.get("model_state", sd.get("state_dict", sd))
            missing, unexpected = m.load_state_dict(
                {k.replace("image_encoder.", ""): v for k, v in sd.items()}, strict=False)
            log(f"    loaded micro-sam weights from {Path(ckpt).name} "
                f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
    except Exception as e:
        log(f"    [warn] {key}: could not set img_size ({str(e)[:70]}); using native {native}")
        if native >= 768:
            log(f"    [warn] {key} will run at {native}px. That is a large attention "
                f"footprint; the batch size will be reduced automatically.")
        m = timm.create_model(spec["timm"], pretrained=True, num_classes=0)
        img_size = native
    m = m.to(device).eval()
    if patch is None:                 # convolutional / pyramid backbone
        m = gpu_boost.to_channels_last(m)
    for p in m.parameters():
        p.requires_grad_(False)

    dc = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(dc["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(dc["std"], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        probe = m.forward_intermediates(
            torch.zeros(1, 3, img_size, img_size, device=device),
            intermediates_only=True, output_fmt="NCHW")
        torch.cuda.empty_cache() if device.type == "cuda" else None
    grids = [tuple(f.shape[-2:]) for f in probe]
    chans = [int(f.shape[1]) for f in probe]
    is_attn = patch is not None
    heads = 12
    try:
        heads = int(m.blocks[0].attn.num_heads)
    except Exception:
        pass
    return m, mean, std, img_size, len(probe), grids, chans, is_attn, heads


def even_taps(n_avail, n_taps):
    """Evenly spaced across depth -- the same STRATEGY for every model, so no
    encoder benefits from a layer set that was tuned for it."""
    if n_avail <= n_taps:
        return list(range(n_avail))
    return sorted(set(np.linspace(0, n_avail - 1, n_taps).round().astype(int).tolist()))


def load_images_np(rows, img_size):
    """Pure-CPU decode. Safe to call from worker threads: PIL releases the GIL."""
    from PIL import Image
    B = len(rows)
    ims = np.empty((B, img_size, img_size), dtype=np.float32)
    mks = np.empty((B, img_size, img_size), dtype=np.float32)
    for i, r in enumerate(rows):
        with Image.open(r["image"]) as im:
            if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
                a = np.asarray(im, dtype=np.float32)
                lo, hi = float(a.min()), float(a.max())
                a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
                im = Image.fromarray((a * 255).astype(np.uint8), mode="L")
            elif im.mode != "L":
                im = im.convert("L")
            ims[i] = np.asarray(im.resize((img_size, img_size), Image.BILINEAR),
                                dtype=np.float32) / 255.0
        with Image.open(r["mask"]) as mk:
            mk = mk.convert("L").resize((img_size, img_size), Image.BILINEAR)
            mks[i] = np.asarray(mk, dtype=np.float32) / 255.0
    return ims, mks


def load_images(rows, img_size, device, mean, std):
    import torch
    from PIL import Image

    B = len(rows)
    ims = np.empty((B, img_size, img_size), dtype=np.float32)
    mks = np.empty((B, img_size, img_size), dtype=np.float32)
    for i, r in enumerate(rows):
        with Image.open(r["image"]) as im:
            if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
                a = np.asarray(im, dtype=np.float32)
                lo, hi = float(a.min()), float(a.max())
                a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
                im = Image.fromarray((a * 255).astype(np.uint8), mode="L")
            elif im.mode != "L":
                im = im.convert("L")
            ims[i] = np.asarray(im.resize((img_size, img_size), Image.BILINEAR),
                                dtype=np.float32) / 255.0
        with Image.open(r["mask"]) as mk:
            mk = mk.convert("L").resize((img_size, img_size), Image.BILINEAR)
            mks[i] = (np.asarray(mk, dtype=np.float32) / 255.0)
    x = torch.from_numpy(ims).to(device).unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - mean) / std
    mk = torch.from_numpy(mks).to(device).unsqueeze(1)
    return x, mk


def pool_maps(feats, mask, taps):
    """For each tap: masked avg/max and unmasked avg/max, concatenated.

    The mask is resized to each tap's own grid, so pyramid encoders are handled
    identically to ViTs. If a particle is too small to cover any cell at a
    coarse grid, the single highest-overlap cell is used instead of silently
    pooling over nothing -- that failure mode is real at 14x14.
    """
    import torch
    import torch.nn.functional as F

    out = {}
    for mode in MASK_MODES:
        avg_parts, max_parts = [], []
        for ti in taps:
            f = feats[ti].float()                          # (B, C, h, w)
            if mode == "nomask":
                avg_parts.append(f.mean(dim=(2, 3)))
                max_parts.append(f.amax(dim=(2, 3)))
                continue
            h, w = f.shape[-2:]
            mr = F.interpolate(mask, size=(h, w), mode="area")   # (B,1,h,w)
            wgt = (mr >= 0.5).float()
            empty = wgt.sum(dim=(2, 3)).squeeze(1) == 0
            if empty.any():                                # rescue tiny particles
                flat = mr.flatten(2).argmax(dim=2)         # (B,1)
                rescue = torch.zeros_like(wgt).flatten(2)
                rescue.scatter_(2, flat.unsqueeze(-1), 1.0)
                rescue = rescue.view_as(wgt)
                wgt = torch.where(empty.view(-1, 1, 1, 1), rescue, wgt)
            denom = wgt.sum(dim=(2, 3)).clamp(min=1.0)     # (B,1)
            avg_parts.append((f * wgt).sum(dim=(2, 3)) / denom)
            neg = torch.finfo(f.dtype).min
            max_parts.append(f.masked_fill(wgt == 0, neg).amax(dim=(2, 3)))
        a = torch.cat(avg_parts, dim=1)
        m = torch.cat(max_parts, dim=1)
        out[(mode, "avg")] = a
        out[(mode, "max")] = m
        out[(mode, "avg+max")] = torch.cat([a, m], dim=1)
    return out


def extract(key, splits, cfg, device, prog=None):
    import torch

    banner(f"FEATURES  |  {key}  ({ENCODERS[key]['timm']})")
    log(f"  {ENCODERS[key]['note']}")
    m, mean, std, img_size, n_avail, grids, chans, is_attn, heads = build_encoder(
        key, cfg.target_grid, cfg.pyramid_img, device)
    if getattr(cfg, "taps", None):
        bad = [t for t in cfg.taps if not (0 <= t < n_avail)]
        if bad:
            raise SystemExit(f"--taps out of range for {key}: {bad} "
                             f"(0..{n_avail-1} available)")
        taps = sorted(set(cfg.taps))
        log(f"  [explicit] layer taps {taps} (manuscript geometry, not even spacing)")
    else:
        taps = even_taps(n_avail, cfg.n_taps)
    tap_grids = [grids[t] for t in taps]
    tap_chans = [chans[t] for t in taps]
    # ---- activation-aware batch sizing -------------------------------------
    # A ResNet's 256x256 conv map is CHEAP (memory linear in positions); a ViT's
    # 64x64 token grid is EXPENSIVE (attention is quadratic). Sizing on token
    # count alone conflates the two and throttles CNNs for no reason, while
    # under-protecting against the case that actually hurts: SAM ViT-B at its
    # native 1024px needs ~6.4 GB for a SINGLE global-attention layer at batch 16.
    linear = sum(int(g[0]) * int(g[1]) * int(c) for g, c in zip(tap_grids, tap_chans))
    max_tok = max(int(g[0]) * int(g[1]) for g in tap_grids)
    quad = heads * max_tok * max_tok if is_attn else 0
    per_item = linear + quad
    safe_bs = max(1, min(cfg.batch_size, int(cfg.act_budget // max(per_item, 1))))
    log(f"  activation/item ~{per_item/1e6:.1f}M elements "
        f"({'attention-bound' if quad > linear else 'feature-map-bound'})")
    if safe_bs < cfg.batch_size:
        log(f"  [auto] batch {cfg.batch_size} -> {safe_bs} to stay under the "
            f"{cfg.act_budget/1e6:.0f}M-element budget")
    cfg.effective_batch = safe_bs
    log(f"  input {img_size}x{img_size} | {n_avail} taps available | using {taps}")
    log(f"  tap grids  {tap_grids}")
    log(f"  tap chans  {tap_chans}  -> hypercolumn {sum(tap_chans)}-D")

    order = ["train", "val", "test"]
    rows, split_of = [], []
    for s in order:
        rows += splits[s]
        split_of += [s] * len(splits[s])
    y = np.array([r["y"] for r in rows], dtype=np.int64)
    split_of = np.array(split_of)

    store, N = {}, len(rows)
    prog.stage_start("extract", N, key)
    t0 = time.time()
    bs = getattr(cfg, "effective_batch", cfg.batch_size)

    def _decode(batch_rows):
        """CPU-side decode, run on a thread pool ahead of the GPU."""
        return load_images_np(batch_rows, img_size)

    s = 0
    with torch.no_grad():
        for batch_rows, (ims, mks) in gpu_boost.prefetch(rows, bs, _decode):
            e = s + len(batch_rows)
            x = torch.from_numpy(ims).to(device, non_blocking=True)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
            x = (x - mean) / std
            mk = torch.from_numpy(mks).to(device, non_blocking=True).unsqueeze(1)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                # indices=taps: materialise ONLY the taps we pool. Without this,
                # all 12 block outputs are kept and 7 of them thrown away.
                feats = m.forward_intermediates(x, indices=taps,
                                                intermediates_only=True,
                                                output_fmt="NCHW")
            pooled = pool_maps(feats, mk, list(range(len(feats))))
            for k, v in pooled.items():
                if k not in store:
                    store[k] = np.empty((N, v.shape[1]), dtype=np.float32)
                store[k][s:e] = v.float().cpu().numpy()
            prog.tick(e - s)
            s = e
            del feats, pooled, x, mk
            if cfg.throttle_ms:
                time.sleep(cfg.throttle_ms / 1000.0)
    prog.stage_end()
    prog.newline()
    log(f"  extracted in {(time.time() - t0)/60:.1f} min")

    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    meta = dict(encoder=key, timm=ENCODERS[key]["timm"], group=ENCODERS[key]["group"],
                pretrained=cfg.pretrained, checkpoint=cfg.checkpoint(key),
                pyramid_img=cfg.pyramid_img, split_file=Path(cfg.splits).name,
                img_size=img_size, taps=taps, tap_grids=[list(g) for g in tap_grids],
                tap_chans=tap_chans, hypercolumn_dim=int(sum(tap_chans)),
                grid_str="/".join(f"{g[0]}" for g in tap_grids),
                n_tokens=int(np.prod(tap_grids[-1])) if tap_grids else 0)
    np.savez_compressed(cfg.cache(key), y=y, split=split_of,
                        meta=json.dumps(meta),
                        **{f"{a}|{b}": v for (a, b), v in store.items()})
    return store, y, split_of, meta


# ---------------------------------------------------------------- classifier
def train_probe(Xtr, ytr, Xes, yes_, Xev, kind, cfg, device):
    import torch
    import torch.nn as nn

    d = Xtr.shape[1]
    if kind == "linear":
        model = nn.Linear(d, 4)
    else:
        model = nn.Sequential(
            nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    Xes_t = torch.tensor(Xes, device=device); yes_t = torch.tensor(yes_, device=device)
    Xev_t = torch.tensor(Xev, device=device)

    n = len(ytr); best, wait, best_state = float("inf"), 0, None
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xes_t), yes_t).item()
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(Xev_t).argmax(1).cpu().numpy()


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def evaluate(X, y, split_of, cfg, device, label, prog=None):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    tr = np.where(split_of == "train")[0]; va = np.where(split_of == "val")[0]
    te = np.where(split_of == "test")[0]
    tv = np.concatenate([tr, va])
    Xtv, ytv, Xte, yte = X[tv], y[tv], X[te], y[te]

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    accs, f1s = [], []
    for i_tr, i_va in skf.split(Xtv, ytv):
        if cfg.earlystop == "leaky":
            i_fit, i_es = i_tr, i_va
        else:
            i_fit, i_es = train_test_split(i_tr, test_size=0.1,
                                           stratify=ytv[i_tr], random_state=cfg.seed)
        sc = StandardScaler().fit(Xtv[i_fit])
        pred = train_probe(sc.transform(Xtv[i_fit]), ytv[i_fit],
                           sc.transform(Xtv[i_es]), ytv[i_es],
                           sc.transform(Xtv[i_va]), cfg.kind, cfg, device)
        accs.append(100 * accuracy_score(ytv[i_va], pred))
        f1s.append(f1_score(ytv[i_va], pred, average="macro"))
        if prog:
            prog.tick()

    if cfg.earlystop == "leaky":
        i_fit = np.arange(len(ytv)); Xes, yes_ = Xte, yte
    else:
        i_fit, i_es = train_test_split(np.arange(len(ytv)), test_size=0.1,
                                       stratify=ytv, random_state=cfg.seed)
        Xes, yes_ = Xtv[i_es], ytv[i_es]
    sc = StandardScaler().fit(Xtv[i_fit])
    pred = train_probe(sc.transform(Xtv[i_fit]), ytv[i_fit],
                       sc.transform(Xes), yes_, sc.transform(Xte), cfg.kind, cfg, device)
    acc = 100 * accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    if prog:
        prog.tick()
    k = int(round(acc / 100 * len(yte)))
    lo, hi = wilson_ci(k, len(yte))
    return dict(cv_acc=float(np.mean(accs)), cv_std=float(np.std(accs)),
                cv_f1=float(np.mean(f1s)), test_acc=float(acc), test_f1=float(f1),
                test_correct=k, test_n=int(len(yte)), test_ci_low=lo, test_ci_high=hi)




def preflight(out_dir, device, seconds=15):
    """Escalating GPU load with a crash-persistent log.

    A hard reboot leaves no traceback, so this writes the last SURVIVED level to
    disk after each step. If the machine dies, the file says exactly which load
    level killed it -- which is the difference between guessing and knowing.
    """
    import torch
    path = Path(out_dir) / "preflight_log.json"
    rec = {"levels": []}
    if path.exists():
        try:
            rec = json.load(open(path))
        except Exception:
            pass
    banner("PREFLIGHT  |  escalating GPU load with a crash-persistent log")
    if device.type != "cuda":
        log("  device is CPU; nothing to test."); return
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"  {name}  {total:.1f} GB  torch {torch.__version__}  cuda {torch.version.cuda}")
    if rec["levels"]:
        log(f"  previous run survived up to level {max(l['level'] for l in rec['levels'])}")
    log(f"  each level runs ~{seconds}s. Ctrl-C at any point is safe.\n")

    for lvl, n in enumerate([2048, 4096, 6144, 8192, 10240], 1):
        log(f"  level {lvl}: {n}x{n} fp16 matmul, sustained {seconds}s ...")
        try:
            a = torch.randn(n, n, device="cuda", dtype=torch.float16)
            b = torch.randn(n, n, device="cuda", dtype=torch.float16)
            t0 = time.time(); it = 0
            while time.time() - t0 < seconds:
                c = a @ b; it += 1
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            tflops = 2 * n**3 * it / (time.time() - t0) / 1e12
            log(f"    survived. {it} iterations, {tflops:.1f} TFLOP/s, peak {peak:.1f} GB")
            rec["levels"].append(dict(level=lvl, n=n, tflops=round(tflops, 1),
                                      peak_gb=round(peak, 2),
                                      when=time.strftime("%Y-%m-%d %H:%M:%S")))
            with open(path, "w") as fh:
                json.dump(rec, fh, indent=2)
            del a, b, c
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            time.sleep(2)
        except RuntimeError as e:
            log(f"    stopped: {str(e)[:100]}")
            break
    log(f"\n  log -> {path}")
    log("  If the machine reboots during this, that is a POWER or THERMAL fault,")
    log("  not a bug in this script -- user code cannot reset a PC.")


# ---------------------------------------------------------------- main
class Cfg:
    def __init__(self, a, out_dir):
        self.out_dir = out_dir
        self.target_grid = a.target_grid
        self.pyramid_img = a.pyramid_img
        self.n_taps = a.n_taps
        self.taps = ([int(x) for x in str(a.taps).replace(' ', '').split(',')]
                     if a.taps else None)
        self.batch_size = a.batch_size
        self.n_folds = 5
        self.seed = 42
        self.epochs = a.epochs
        self.batch = 32
        self.patience = 15
        self.earlystop = a.earlystop
        self.kind = "mlp"
        self.act_budget = a.act_budget
        self.throttle_ms = a.throttle_ms
        self.effective_batch = a.batch_size
        # Everything below is part of the cache identity -- see cache_name().
        self.splits = a.splits
        import os
        self.pretrained = os.environ.get("ENCBENCH_NO_PRETRAINED") != "1"

    @staticmethod
    def checkpoint(key):
        """Checkpoint override for this encoder, or None.  Only usam_b has one."""
        import os
        return os.environ.get("USAM_CHECKPOINT") if key == "usam_b" else None

    def cache(self, key):
        return self.out_dir / cache_name(
            key, self.target_grid, self.taps, self.n_taps, Path(self.splits).stem,
            pyramid_img=self.pyramid_img, pretrained=self.pretrained,
            checkpoint=self.checkpoint(key))


def cache_name(key, target_grid, taps, n_taps, split_stem, *, pyramid_img=512,
               pretrained=True, checkpoint=None):
    """The feature-cache filename.  EVERYTHING that changes the features is in
    the name, so a cache can never be silently reused for a different
    configuration:

      geometry     g<grid> and either t<n> (even taps) or L<a-b-c> (explicit taps)
      split file   the pickle's stem; the two split files that have existed
                   contain 1,785 and 1,800 images and are not interchangeable
      pyramid_img  input size of the pyramid encoders (ResNet/ConvNeXt/Hiera);
                   ViTs derive their input from the grid, so the default 512
                   adds no suffix and the historical names stay valid
      pretrained   ENCBENCH_NO_PRETRAINED=1 (random weights, smoke tests only)
                   gets a `_RANDOMWEIGHTS` suffix, so a smoke-test cache can
                   never be picked up by a real run
      checkpoint   a USAM_CHECKPOINT override is hashed into the name

    paper_results.cache_path() builds the same name independently and
    check_cache_agreement.py asserts the two agree.
    """
    tag = ("L" + "-".join(map(str, sorted(set(taps))))) if taps else f"t{n_taps}"
    suffix = "" if split_stem == "balanced_dataset_splits" else f"_{split_stem}"
    if int(pyramid_img) != 512:
        suffix += f"_p{int(pyramid_img)}"
    if checkpoint:
        suffix += "_ckpt" + checkpoint_digest(checkpoint)
    if not pretrained:
        suffix += "_RANDOMWEIGHTS"
    return f"feats_{key}_g{target_grid}_{tag}{suffix}.npz"


def checkpoint_digest(checkpoint):
    """8 hex chars identifying a checkpoint override: the file's SHA-1 when it
    exists, otherwise the SHA-1 of the string (so a dry run still resolves)."""
    cp = Path(checkpoint)
    h = hashlib.sha1()
    if cp.is_file():
        with open(cp, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    else:
        h.update(str(checkpoint).encode())
    return h.hexdigest()[:8]


def summarise(out_dir, splits_stem):
    import csv
    rows = []
    for f in sorted(out_dir.glob(f"results_{splits_stem}_*.json")):
        rows += json.load(open(f))
    if not rows:
        log("no results yet."); return
    out = out_dir / f"encoder_benchmark_{splits_stem}.csv"
    keys = ["encoder", "group", "mask", "pooling", "classifier", "grids", "n_tokens",
            "hypercolumn_dim", "cv_acc", "cv_std", "test_acc", "test_correct",
            "test_ci", "test_f1"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    banner("ENCODER BENCHMARK")
    # ---- best configuration per encoder, selected on CV, never on test
    log("\n  BEST PER ENCODER  (configuration chosen on CV accuracy)\n")
    log(f"  {'encoder':13} {'group':10} {'mask':7} {'pool':8} {'clf':7} "
        f"{'grids':>14} {'CV':>14} {'test':>7} {'95% CI':>14}")
    log("  " + "-" * 96)
    per_enc = {}
    for r in rows:
        per_enc.setdefault(r["encoder"], []).append(r)
    ranked = []
    for enc, rs in per_enc.items():
        b = max(rs, key=lambda r: r["cv_acc"])
        ranked.append(b)
    for b in sorted(ranked, key=lambda r: -r["cv_acc"]):
        log(f"  {b['encoder']:13} {b['group']:10} {b['mask']:7} {b['pooling']:8} "
            f"{b['classifier']:7} {str(b.get('grids','')):>14} "
            f"{b['cv_acc']:6.2f} +/-{b['cv_std']:5.2f} {b['test_acc']:7.2f} "
            f"{b['test_ci']:>14}")

    # ---- the architecture-independent claim
    log("\n  MASK-GUIDED vs UNMASKED  (same encoder, same pooling, same classifier)\n")
    log(f"  {'encoder':13} {'masked CV':>11} {'unmasked CV':>13} {'delta':>8}   "
        f"{'masked test':>12} {'unmasked test':>14} {'delta':>8}")
    log("  " + "-" * 88)
    deltas_cv, deltas_te = [], []
    for enc, rs in per_enc.items():
        m = [r for r in rs if r["mask"] == "masked"]
        u = [r for r in rs if r["mask"] == "nomask"]
        if not m or not u:
            continue
        bm = max(m, key=lambda r: r["cv_acc"])
        bu = [r for r in u if r["pooling"] == bm["pooling"]
              and r["classifier"] == bm["classifier"]]
        if not bu:
            continue
        bu = bu[0]
        dcv = bm["cv_acc"] - bu["cv_acc"]; dte = bm["test_acc"] - bu["test_acc"]
        deltas_cv.append(dcv); deltas_te.append(dte)
        log(f"  {enc:13} {bm['cv_acc']:11.2f} {bu['cv_acc']:13.2f} {dcv:+8.2f}   "
            f"{bm['test_acc']:12.2f} {bu['test_acc']:14.2f} {dte:+8.2f}")
    if deltas_cv:
        n_pos = sum(1 for d in deltas_cv if d > 0)
        log("  " + "-" * 88)
        log(f"  mask-guided pooling helps on CV for {n_pos}/{len(deltas_cv)} encoders; "
            f"mean {np.mean(deltas_cv):+.2f} pp (test {np.mean(deltas_te):+.2f} pp)")
        if n_pos == len(deltas_cv):
            log("  -> the gain is ARCHITECTURE-INDEPENDENT, which is the claim that")
            log("     survives a better backbone appearing next month.")
    log(f"\n  -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the roster and exit")
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--group", choices=list(GROUPS), default="benchmark")
    ap.add_argument("--splits", type=Path, default=cnt_paths.splits_file(),
                    help="split pickle (default: <repo>/splits/dataset_splits.pkl)")
    ap.add_argument("--root", type=Path, default=None,
                    help="data root; default is the CNT_BASE environment variable")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="feature-cache directory (default <CNT_BASE>/Encoder Benchmark)")
    ap.add_argument("--stage", choices=["all", "features", "classify"], default="all")
    ap.add_argument("--target-grid", type=int, default=32,
                    help="token grid every ViT is resized to, for a fair comparison")
    ap.add_argument("--pyramid-img", type=int, default=512,
                    help="input size for pyramid encoders (ResNet/ConvNeXt/Hiera)")
    ap.add_argument("--taps", default=None,
                    help="explicit hypercolumn layer indices, e.g. 1,3,6,9,11. "
                         "Overrides the even-spacing strategy. Use this to "
                         "reproduce the geometry described in the manuscript "
                         "(DINOv2 at 518px, layers [1,3,6,9,11]); leave unset "
                         "for the cross-encoder benchmark, where even spacing "
                         "keeps the layer strategy identical across models.")
    ap.add_argument("--n-taps", type=int, default=5,
                    help="hypercolumn taps, evenly spaced across depth")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="upper bound; automatically reduced for high-token models")
    ap.add_argument("--act-budget", type=float, default=200e6,
                    help="peak activation elements per batch. Lower it to reduce "
                         "peak GPU load; batch size is derived from this per model")
    ap.add_argument("--throttle-ms", type=int, default=0,
                    help="pause between batches (ms). Reduces SUSTAINED draw, which "
                         "is what trips a marginal PSU")
    ap.add_argument("--max-vram-frac", type=float, default=0.0,
                    help="cap this process's VRAM (0-1). Fails fast instead of thrashing")
    ap.add_argument("--safe", action="store_true",
                    help="conservative preset: token-budget 2048, throttle 50 ms, "
                         "VRAM cap 0.6")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--earlystop", choices=["clean", "leaky"], default="clean")
    ap.add_argument("--progress-file", default=None)
    ap.add_argument("--root-remap", default=None, help=argparse.SUPPRESS)  # obsolete
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="escalating GPU stress test with a crash-persistent log")
    ap.add_argument("--dry-run", action="store_true",
                    help="build Cfg, resolve every cache path and load the splits, "
                         "then exit 0 without extracting anything")
    ap.add_argument("--force", action="store_true",
                    help="redo encoders whose results already exist")
    a = ap.parse_args(argv)

    if a.list:
        log(f"{'key':14} {'group':10} timm id")
        log("-" * 84)
        for k, v in ENCODERS.items():
            log(f"{k:14} {v['group']:10} {v['timm']}")
            log(f"{'':25} {v['note']}")
        log("\ngroups: " + ", ".join(f"{g} ({len(v)})" for g, v in GROUPS.items()
                                     if g != "reviewers"))
        return

    if a.root is not None:
        import os
        os.environ["CNT_BASE"] = str(a.root)
    if a.out_dir is None:
        a.out_dir = cnt_paths.bench_dir()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.splits).stem
    if a.summarise_only:
        summarise(a.out_dir, stem); return

    if a.safe:
        a.act_budget = min(a.act_budget, 50e6)
        a.throttle_ms = max(a.throttle_ms, 50)
        a.max_vram_frac = a.max_vram_frac or 0.6

    import torch
    torch.manual_seed(42); np.random.seed(42)
    gpu_boost.setup()
    dev = a.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    if dev == "cuda" and a.max_vram_frac:
        torch.cuda.set_per_process_memory_fraction(float(a.max_vram_frac), 0)

    if a.preflight:
        preflight(a.out_dir, device); return

    keys = a.encoders or GROUPS[a.group]
    bad = [k for k in keys if k not in ENCODERS]
    if bad:
        raise SystemExit(f"unknown encoder(s): {bad}. Use --list.")

    cfg = Cfg(a, a.out_dir)

    banner("MULTI-ENCODER BENCHMARK")
    log(f"  device      : {dev}" +
        (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    log(f"  splits      : {a.splits}")
    log(f"  encoders    : {len(keys)} -> {', '.join(keys)}")
    log(f"  target grid : {a.target_grid}x{a.target_grid} tokens for every ViT "
        f"(pyramid models at {a.pyramid_img}px)")
    log(f"  taps        : " + (f"explicit {cfg.taps} (manuscript geometry)" if cfg.taps
        else f"{a.n_taps}, evenly spaced -- SAME strategy for every model"))
    if not cfg.pretrained:
        log("  WEIGHTS     : RANDOM (ENCBENCH_NO_PRETRAINED=1) -- caches carry _RANDOMWEIGHTS")
    log(f"  early stop  : {a.earlystop}")
    log(f"  act budget  : {a.act_budget/1e6:.0f}M elements/batch "
        f"(batch auto-derived per model)")
    if a.throttle_ms:
        log(f"  throttle    : {a.throttle_ms} ms between batches")
    if a.max_vram_frac:
        log(f"  VRAM cap    : {a.max_vram_frac:.0%} of the card")
    if a.safe:
        log("  SAFE PRESET : conservative load. Use this if the machine is unstable.")
    log(f"  configs     : {len(keys)} x {len(MASK_MODES)} x {len(POOLINGS)} x "
        f"{len(CLASSIFIERS)} = {len(keys)*len(MASK_MODES)*len(POOLINGS)*len(CLASSIFIERS)}")

    banner("SPLITS")
    splits = load_splits(a.splits, a.root, a.root_remap)
    for s in ("train", "val", "test"):
        c = np.bincount([r["y"] for r in splits[s]], minlength=4)
        log(f"  {s:5} n={len(splits[s]):5d}   " +
            "  ".join(f"{ID_TO_CATEGORY[i]}={c[i]}" for i in range(4)))

    if a.dry_run:
        banner("DRY RUN  |  resolved cache paths")
        for k in keys:
            c = cfg.cache(k)
            log(f"  {k:14} -> {c.name}   {'EXISTS' if c.exists() else 'to build'}")
        log("\n  splits, Cfg and every cache path resolved. Exiting 0.")
        return

    prog_path = Path(a.progress_file) if a.progress_file else (a.out_dir / "progress.json")
    prog = Progress(prog_path, len(keys))
    log(f"\n  live progress -> {prog_path}")

    for ei, key in enumerate(keys, 1):
        prog.job_start(ei, key)
        res_path = a.out_dir / f"results_{stem}_{key}.json"
        if res_path.exists() and not a.force and a.stage != "features":
            log(f"\n# ENCODER {ei}/{len(keys)}: {key}  -- already done, skipping "
                f"(--force to redo)")
            continue
        log(f"\n{'#'*78}\n# ENCODER {ei}/{len(keys)}: {key}\n{'#'*78}")
        cache = cfg.cache(key)
        if a.stage in ("all", "features") and not cache.exists():
            store, y, split_of, meta = extract(key, splits, cfg, device, prog)
        else:
            if not cache.exists():
                log(f"  no cache for {key}; run --stage all first."); continue
            z = np.load(cache, allow_pickle=True)
            y = z["y"]; split_of = z["split"]; meta = json.loads(str(z["meta"]))
            store = {tuple(k.split("|")): z[k] for k in z.files
                     if "|" in k}
            log(f"  loaded cached features ({meta['hypercolumn_dim']}-D, "
                f"{meta['n_tokens']} tokens)")
        if a.stage == "features":
            continue

        results = []
        combos = [(mm, pl, cl) for mm in MASK_MODES for pl in POOLINGS
                  for cl in CLASSIFIERS]
        prog.stage_start("probe", len(combos) * (cfg.n_folds + 1), key)
        for mm, pl, cl in combos:
            cfg.kind = cl
            X = store[(mm, pl)]
            t0 = time.time()
            r = evaluate(X, y, split_of, cfg, device, f"{key}/{mm}/{pl}/{cl}", prog)
            r.update(encoder=key, group=ENCODERS[key]["group"], mask=mm, pooling=pl,
                     classifier=cl, n_tokens=meta["n_tokens"],
                     grids=meta.get("grid_str", ""),
                     hypercolumn_dim=meta["hypercolumn_dim"],
                     test_ci=f"[{r['test_ci_low']:.1f},{r['test_ci_high']:.1f}]",
                     seconds=round(time.time() - t0, 1))
            results.append(r)
            prog.newline()
            log(f"    {mm:7} {pl:8} {cl:7}  CV {r['cv_acc']:6.2f} +/-{r['cv_std']:5.2f}"
                f"   test {r['test_acc']:6.2f}  ({r['test_correct']}/{r['test_n']})"
                f"  [{time.time()-t0:.0f}s]")
        prog.stage_end()
        with open(a.out_dir / f"results_{stem}_{key}.json", "w") as fh:
            json.dump(results, fh, indent=2)

    prog.close()
    if a.stage != "features":
        summarise(a.out_dir, stem)


if __name__ == "__main__":
    main()
