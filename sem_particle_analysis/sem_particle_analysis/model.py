"""
SAM Model Initialization and Management
"""

import os
from pathlib import Path

import torch
from segment_anything import sam_model_registry, SamPredictor

# Fix OpenMP library conflict (common with Intel MKL and PyTorch)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Canonical filenames published by Meta, used to infer the architecture from a
# checkpoint name so it doesn't have to be selected by hand.
CHECKPOINT_FILENAMES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}

# Searched in order when no explicit path is given.
def _default_search_dirs():
    repo_root = Path(__file__).resolve().parents[2]
    return [repo_root / "sam_weights", Path.home() / "segment-anything" / "models"]


def infer_model_type(path, default="vit_h"):
    """
    Guess the SAM architecture from a checkpoint filename.

    Loading a vit_b checkpoint as vit_h fails deep inside torch with an opaque
    shape-mismatch error, so it is worth getting right without asking.

    Args:
        path: Checkpoint path or filename.
        default (str): Returned when the name carries no hint.

    Returns:
        str: 'vit_h', 'vit_l' or 'vit_b'.
    """
    name = Path(path).name.lower()
    return next(
        (m for m in ("vit_h", "vit_l", "vit_b")
         if m in name or m.replace("_", "") in name),
        default,
    )


def is_full_checkpoint(path):
    """
    True if the filename looks like a complete SAM model rather than a fragment.

    A partial checkpoint — one holding only a mask decoder, say — cannot be
    loaded by ``sam_model_registry`` and fails with an opaque key error, so such
    files are offered last rather than being picked as a default.
    """
    return infer_model_type(path, default=None) is not None


def discover_checkpoints(extra_dirs=None):
    """
    Find SAM checkpoints on disk, best default first.

    Ordered so that the first entry is a sensible default: complete checkpoints
    before partial ones, then ViT-B before the larger models.

    ViT-B is the default rather than ViT-H because of how the app is used. The
    analyst judges and corrects every mask, so the model produces a starting
    point rather than an answer, and ViT-B gets there several times faster from
    a 375 MB checkpoint instead of a 2.4 GB one. Over a folder of hundreds, the
    wait for the larger encoder costs more than its extra accuracy returns —
    and it is one click away in the same list when a frame deserves it.

    Args:
        extra_dirs: Additional directories to search before the defaults.

    Returns:
        list[Path]: Existing ``.pth`` files, de-duplicated by resolved target so
        a symlink and the file it points at are not offered twice.
    """
    found, seen = [], set()
    for directory in [Path(d) for d in (extra_dirs or [])] + _default_search_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.pth")):
            target = path.resolve()
            if target in seen:
                continue
            seen.add(target)
            found.append(path)

    preference = {"vit_b": 0, "vit_l": 1, "vit_h": 2}
    found.sort(key=lambda p: (
        not is_full_checkpoint(p),
        preference.get(infer_model_type(p, default=None), 3),
        p.name,
    ))
    return found


class SAMModel:
    """
    Manages the Segment Anything Model (SAM) initialization and device setup.

    Attributes:
        model_type (str): SAM model variant ('vit_h', 'vit_l', or 'vit_b')
        checkpoint (str): Path to SAM checkpoint file
        device (torch.device): Computing device (GPU/CPU)
        predictor (SamPredictor): SAM predictor instance
    """

    def __init__(self, checkpoint_path, model_type="vit_h", device=None):
        """
        Initialize the SAM model.

        Args:
            checkpoint_path (str): Path to SAM checkpoint file
            model_type (str): Model variant - 'vit_h' (best), 'vit_l', or 'vit_b' (fastest)
            device (str or torch.device, optional): Computing device. Auto-detected if None.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If model_type is invalid
        """
        self.checkpoint = checkpoint_path
        self.model_type = model_type

        # Verify checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}\n"
                f"Download from: https://github.com/facebookresearch/segment-anything"
            )

        # Verify model type
        if model_type not in sam_model_registry:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                f"Choose from: {list(sam_model_registry.keys())}"
            )

        # Setup device
        self.device = self._setup_device(device)

        # Load model
        print(f"Loading SAM model ({model_type})...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device)

        # Create predictor
        self.predictor = SamPredictor(self.sam)
        print(f"SAM model loaded successfully on {self.device}")

    def _setup_device(self, device=None):
        """
        Determine the best available computing device.

        Priority: Apple Silicon GPU (MPS) > CUDA GPU > CPU

        Handles compatibility across platforms:
        - macOS: Uses MPS (Metal Performance Shaders) for Apple Silicon
        - Windows/Linux: Uses CUDA for NVIDIA GPUs
        - Fallback: CPU if no GPU available

        Args:
            device (str or torch.device, optional): Requested device

        Returns:
            torch.device: Selected device
        """
        if device is not None:
            requested_device = torch.device(device)
            print(f"Using requested device: {requested_device}")
            return requested_device

        # Auto-detect best device
        # Priority 1: Apple Silicon GPU (MPS) - for macOS with M1/M2/M3/M4
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
                device_name = "Apple Silicon GPU (MPS)"
                print(f"Using device: {device_name}")
                return device
        except Exception as e:
            print(f"MPS detection error: {e}")

        # Priority 2: NVIDIA CUDA GPU - for Windows/Linux with NVIDIA GPUs
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                device = torch.device("cuda")
                device_name = f"NVIDIA GPU (CUDA) - {gpu_name}"
                print(f"Using device: {device_name}")
                return device
        except Exception as e:
            print(f"CUDA detection error: {e}")

        # Priority 3: CPU fallback
        device = torch.device("cpu")
        device_name = "CPU (no GPU detected)"
        print(f"Using device: {device_name}")
        print("No GPU acceleration available. Processing will be slower.")
        return device

    def set_image(self, image):
        """
        Set the image for SAM prediction (runs encoder once).

        Args:
            image (np.ndarray): RGB image array (H, W, 3)
        """
        self.predictor.set_image(image)

    def predict(self, point_coords=None, point_labels=None, box=None,
                multimask_output=True):
        """
        Generate segmentation masks using SAM.

        Args:
            point_coords (np.ndarray, optional): Nx2 array of point prompts
            point_labels (np.ndarray, optional): N array of labels (1=positive, 0=negative)
            box (np.ndarray, optional): Bounding box [x0, y0, x1, y1]
            multimask_output (bool): Whether to return multiple mask candidates

        Returns:
            tuple: (masks, scores, logits)
                - masks: Boolean arrays of shape (num_masks, H, W)
                - scores: Confidence scores for each mask
                - logits: Raw mask logits
        """
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )
