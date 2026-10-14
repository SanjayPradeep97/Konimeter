#!/usr/bin/env python3
"""
Download Segment Anything (SAM) checkpoints into sam_weights/.

The demo app offers to download ViT-B itself, with a button; this script does
the same from a terminal, and reproduce.bat / reproduce.sh call it for
Figure 7. Weights go to sam_weights/ in this folder, or to the directory named
by SAM_WEIGHTS_DIR.

    python download_sam_weights.py --model vit_b
    python download_sam_weights.py --model vit_b vit_h

With no arguments an interactive menu is shown.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

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


def weights_dir():
    """SAM_WEIGHTS_DIR if set, otherwise sam_weights/ beside this script."""
    if os.environ.get("SAM_WEIGHTS_DIR"):
        return Path(os.environ["SAM_WEIGHTS_DIR"]).expanduser()
    return Path(__file__).resolve().parent / "sam_weights"


def checkpoint_path(model_type="vit_b"):
    """Where a checkpoint is, or would be once downloaded."""
    return weights_dir() / MODELS[model_type]["filename"]


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


def print_progress(done, total):
    if total > 0:
        percent = 100 * done / total
        filled = int(percent / 2)
        print(f"\r[{'=' * filled}{'-' * (50 - filled)}] {percent:.1f}% "
              f"({done / 2**20:.1f}/{total / 2**20:.1f} MB)", end="", flush=True)


def choose_interactively():
    print("\nWhich model(s) would you like to download?")
    print("  1. ViT-B (358 MB; the one the demo and Figure 7 use, recommended)")
    print("  2. ViT-L (1.2 GB)")
    print("  3. ViT-H (2.4 GB)")
    print("  4. All models")
    choice = input("\nEnter choice (1-4): ").strip()
    return {"1": ["vit_b"], "2": ["vit_l"], "3": ["vit_h"],
            "4": ["vit_b", "vit_l", "vit_h"]}.get(choice, [])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download SAM checkpoints into sam_weights/.")
    ap.add_argument("--model", nargs="+", choices=list(MODELS), default=None,
                    help="checkpoint(s) to download without asking, e.g. --model vit_b")
    args = ap.parse_args(argv)

    models = list(args.model) if args.model else choose_interactively()
    if not models:
        print("Invalid choice. Exiting.")
        return 1

    failed = []
    for model_type in models:
        model = MODELS[model_type]
        if checkpoint_path(model_type).exists():
            print(f"{model['filename']} already exists. Skipping.")
            continue
        print(f"Downloading {model['filename']} ({model['size']})...")
        try:
            download(model_type, progress=print_progress)
            print()
        except Exception as problem:
            print(f"\nError downloading {model['filename']}: {problem}")
            failed.append(model_type)

    if failed:
        print(f"FAILED: {', '.join(failed)} (check the connection and re-run)")
        return 1
    print(f"Weights are in {weights_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
