"""
finetune_fit.py -- the end-to-end fine-tuning adapter for paper_protocol.run().

The frozen probes and the fine-tuned CNNs go through the SAME protocol object,
so the frozen-vs-fine-tuned comparison is like-for-like: same 1,606
development images, same five folds, same held-out 179, same rule for when the
test set may be touched.  Only the model differs.

`convnextv2` was listed in the plan but is not a key in finetune_baselines.MODELS;
it is added here rather than silently dropped.
"""
from __future__ import annotations
import numpy as np


def make_finetune_fit(items_dev, y_dev, items_test, model_key, cfg, spec, tag):
    """items_* are lists of (image_array, label) as produced by prepare()."""
    import finetune_baselines as FB

    def fit(train_idx, stop_idx, targets, fixed_epochs, seed):
        tr = [items_dev[i] for i in np.asarray(train_idx)]
        es = [items_dev[i] for i in np.asarray(stop_idx)] if stop_idx is not None else []
        ev = [items_test if isinstance(t, str) else [items_dev[i] for i in np.asarray(t)]
              for t in targets]
        if spec["kind"] == "timm":
            return FB.finetune_timm(spec["id"], tr, es, ev, cfg, seed,
                                    fixed_epochs=fixed_epochs)
        return FB.finetune_yolo(spec["id"], tr, es, ev, cfg, seed, tag,
                                fixed_epochs=fixed_epochs)

    return fit


EXTRA_MODELS = {
    "convnextv2": dict(kind="timm", id="convnextv2_base.fcmae_ft_in1k",
                       note="modern CNN control (fine-tuned)"),
}
