"""
Downloading SAM checkpoints.

Used by the app's Setup tab and by download_sam_weights.py. Checkpoints go to
``model.weights_dir()``, which is also where the app looks for them, so a model
downloaded once is found every time the app starts.
"""

import urllib.request
from pathlib import Path

from .model import weights_dir

MODELS = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size": "358 MB",
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size": "1.2 GB",
    },
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "size": "2.4 GB",
    },
}


def download(model_type="vit_b", directory=None, progress=None):
    """
    Download one SAM checkpoint, unless it is already there.

    The file is written as ``<name>.part`` and renamed only once complete, and
    urllib raises if the server sent less than it announced, so an interrupted
    download never leaves a truncated checkpoint that later fails to load.

    Args:
        model_type: 'vit_b', 'vit_l' or 'vit_h'.
        directory: Where to save it; ``weights_dir()`` if None.
        progress: Called as ``progress(done_bytes, total_bytes)`` as data
            arrives; total_bytes is 0 if the server did not say.

    Returns:
        Path: The checkpoint.
    """
    model = MODELS[model_type]
    directory = Path(directory) if directory is not None else weights_dir()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / model["filename"]
    if destination.exists():
        return destination

    partial = destination.with_name(destination.name + ".part")

    def hook(blocks, block_size, total):
        if progress is not None:
            done = blocks * block_size
            progress(min(done, total) if total > 0 else done, max(total, 0))

    try:
        urllib.request.urlretrieve(model["url"], partial, hook)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return destination
