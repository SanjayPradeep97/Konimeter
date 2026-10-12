#!/usr/bin/env python3
"""
Download SAM Model Weights

The app offers to download ViT-B itself, on its Setup tab. This script does the
same from a terminal, for scripts and installers, and fetches the larger models
too. Weights go to the same place either way: sam_weights/ in this folder, or
the directory named by SAM_WEIGHTS_DIR.

Available models:
- ViT-B (vit_b): Fastest, smallest model (358 MB); what the app loads first
- ViT-L (vit_l): Good balance of quality and speed (1.2 GB)
- ViT-H (vit_h): Best quality, largest model (2.4 GB)

    python download_sam_weights.py --model vit_b
    python download_sam_weights.py --model vit_b vit_h

With no arguments an interactive menu is shown.
"""

import argparse
import sys

from sem_particle_analysis.model import weights_dir
from sem_particle_analysis.weights import MODELS, download


def print_progress(done, total):
    if total > 0:
        percent = 100 * done / total
        filled = int(percent / 2)
        print(f"\r[{'=' * filled}{'-' * (50 - filled)}] {percent:.1f}% "
              f"({done / 2**20:.1f}/{total / 2**20:.1f} MB)", end="", flush=True)


def choose_interactively():
    print("\nWhich model(s) would you like to download?")
    print("  1. ViT-B (fastest, 358 MB; what the app loads first, recommended)")
    print("  2. ViT-L (1.2 GB)")
    print("  3. ViT-H (best quality, 2.4 GB)")
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
        if (weights_dir() / model["filename"]).exists():
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
