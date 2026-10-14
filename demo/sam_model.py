"""sam_model.py -- Segment Anything (ViT-B), as the demo uses it.

The image is encoded once when it is opened; each click then runs only SAM's
light mask decoder on that embedding, which is why a click returns a mask in a
fraction of a second while opening an image takes longer.

    sam = SAMModel(checkpoint)               # sam_weights/sam_vit_b_01ec64.pth
    sam.set_image(rgb_uint8_HxWx3)
    masks, scores, _ = sam.predict(point_coords, point_labels)
"""
from __future__ import annotations
import os
from pathlib import Path

# PyTorch and NumPy each ship an OpenMP runtime; on Windows they otherwise
# refuse to share a process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def best_device():
    """An NVIDIA GPU, else Apple silicon's GPU, else the CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:          # a PyTorch built without MPS
        pass
    return torch.device("cpu")


class SAMModel:
    def __init__(self, checkpoint, model_type="vit_b", device=None):
        from segment_anything import SamPredictor, sam_model_registry
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.device = device or best_device()
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def set_image(self, rgb):
        self.predictor.set_image(rgb)

    def predict(self, point_coords, point_labels, multimask_output=True):
        return self.predictor.predict(point_coords=point_coords,
                                      point_labels=point_labels,
                                      multimask_output=multimask_output)
