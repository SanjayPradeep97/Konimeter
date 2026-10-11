#!/usr/bin/env python3
"""
Download SAM Model Weights

Downloads the Segment Anything Model (SAM) weights from Meta AI's repository.
Weights are saved to the sam_weights/ directory next to this script.

Available models:
- ViT-H (vit_h): Best quality, largest model (2.4 GB)
- ViT-L (vit_l): Good balance of quality and speed (1.2 GB)
- ViT-B (vit_b): Fastest, smallest model (358 MB)

Non-interactive use (what the demo's launch scripts do):

    python download_sam_weights.py --model vit_b
    python download_sam_weights.py --model vit_b vit_h

With no arguments an interactive menu is shown.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

# Model download URLs
MODELS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "size": "2.4 GB",
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size": "1.2 GB",
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size": "358 MB",
    },
}


def download_file(url, destination):
    """Download `url` to `destination` with a progress bar, via a .part file so
    an interrupted download never leaves a truncated checkpoint behind."""
    destination = Path(destination)
    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading to: {destination}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded / total_size * 100, 100)
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = "=" * filled + "-" * (bar_length - filled)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                  end="", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp, progress_hook)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    print()  # New line after progress bar
    tmp.replace(destination)


def choose_interactively():
    print("\nWhich model(s) would you like to download?")
    print("  1. ViT-H (best quality, recommended)")
    print("  2. ViT-L (good balance)")
    print("  3. ViT-B (fastest)")
    print("  4. All models")
    print("  5. ViT-H and ViT-B (recommended for most users)")
    choice = input("\nEnter choice (1-5): ").strip()
    return {"1": ["vit_h"], "2": ["vit_l"], "3": ["vit_b"],
            "4": ["vit_h", "vit_l", "vit_b"], "5": ["vit_h", "vit_b"]}.get(choice, [])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download SAM checkpoints into sam_weights/.")
    ap.add_argument("--model", nargs="+", choices=list(MODELS), default=None,
                    help="checkpoint(s) to download without asking, e.g. --model vit_b")
    args = ap.parse_args(argv)

    weights_dir = Path(__file__).resolve().parent / "sam_weights"
    weights_dir.mkdir(exist_ok=True)

    print("SAM Model Weights Downloader")
    print("=" * 60)
    print("\nAvailable models:")
    for key, model in MODELS.items():
        print(f"  {key}: {model['filename']} ({model['size']})")

    models_to_download = list(args.model) if args.model else choose_interactively()
    if not models_to_download:
        print("Invalid choice. Exiting.")
        return 1

    print(f"\nDownloading {len(models_to_download)} model(s)...")
    failed = []
    for model_key in models_to_download:
        model = MODELS[model_key]
        destination = weights_dir / model["filename"]
        if destination.exists():
            print(f"\n{model['filename']} already exists. Skipping.")
            continue
        print(f"\nDownloading {model['filename']} ({model['size']})...")
        try:
            download_file(model["url"], destination)
            print("Downloaded successfully.")
        except Exception as e:
            print(f"Error downloading {model['filename']}: {e}")
            failed.append(model_key)

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {', '.join(failed)} (check the connection and re-run)")
        return 1
    print("Download complete!")
    print(f"Weights saved to: {weights_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
