"""
Headless batch analysis.

Runs the same pipeline as the web app over a folder of micrographs without any
clicking, and records enough about each run to reproduce it later: the exact
inputs, the model, the scale and where it came from, every parameter, and the
library versions. Interactive refinement is deliberately not available here —
anything this tool reports came out of the automatic pipeline alone.

    sem-analyze path/to/images -o path/to/output
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import modality, region
from .analysis import ParticleAnalyzer
from .model import SAMModel, discover_checkpoints, infer_model_type
from .scale_detection import ScaleDetector
from .segmentation import ParticleSegmenter
from .utils import find_images_in_folder, load_image

# Used only when the databar can't be measured.
DEFAULT_CROP_PERCENT = 7.0

# --particles, as the ``dark_features`` argument the segmenter takes.
POLARITY_CHOICES = {"auto": None, "bright": False, "dark": True}


class FrameSkipped(Exception):
    """
    Raised when a frame is deliberately not analysed.

    Distinct from a failure: the run is fine, this frame simply does not belong
    in the measurement. Recorded as such so a skipped navigation shot is never
    mistaken for an image the pipeline choked on.
    """


def _out_of_scale_range(nm_per_px, args):
    """Why this frame's magnification excludes it, or None to analyse it."""
    if nm_per_px is None:
        return None
    ceiling = getattr(args, "max_nm_per_px", None)
    floor = getattr(args, "min_nm_per_px", None)
    if ceiling is not None and nm_per_px > ceiling:
        return (f"{nm_per_px:.4g} nm/px is coarser than the --max-nm-per-px limit "
                f"of {ceiling:g}; magnification too low to resolve particles")
    if floor is not None and nm_per_px < floor:
        return (f"{nm_per_px:.4g} nm/px is finer than the --min-nm-per-px limit "
                f"of {floor:g}")
    return None


def _polarity(args, kind):
    """
    Which side of the frame holds the particles, as ``dark_features``.

    An explicit --particles wins; otherwise the modality decides, because it
    knows: material on an SEM filter substrate reads brighter than the membrane,
    and electron-dense material in TEM reads darker than the support film. Only
    when the modality is genuinely unknown is this left to contrast.
    """
    chosen = getattr(args, "particles", "auto") or "auto"
    if chosen != "auto":
        return POLARITY_CHOICES[chosen]
    return kind.dark_particles

def _repo_root():
    # cli.py -> sem_particle_analysis -> sem_particle_analysis -> repo root
    return Path(__file__).resolve().parents[2]


def find_checkpoint(model_type):
    """Locate SAM weights for ``model_type``, or return None."""
    return next((p for p in discover_checkpoints()
                 if infer_model_type(p, default=None) == model_type), None)


def sha256_file(path, chunk_size=1 << 20):
    """SHA-256 of a file, read in chunks so multi-GB checkpoints don't blow up RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision():
    try:
        out = subprocess.run(
            ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _versions():
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "torch", "skimage", "cv2", "easyocr"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    try:
        from . import __version__

        versions["sem_particle_analysis"] = __version__
    except Exception:
        versions["sem_particle_analysis"] = None
    return versions


def resolve_scale(detector, image, image_path, args):
    """
    Determine nm/pixel for one image.

    Returns:
        tuple: (nm_per_px, provenance_dict). nm_per_px is None when scale could
        not be established, in which case measurements stay in pixels.
    """
    if args.scale_nm_per_px is not None:
        return args.scale_nm_per_px, {"method": "manual", "nm_per_px": args.scale_nm_per_px}

    try:
        result = detector.detect_scale(image, file_path=str(image_path), method=args.scale_method)
        bar_region = result.get("region")
        return result["conversion"], {
            "method": result.get("method", args.scale_method),
            "nm_per_px": result["conversion"],
            "scale_nm": result.get("scale_nm"),
            "pixel_length": result.get("pixel_length"),
            "ocr_text": result.get("ocr_text"),
            "warning": result.get("warning"),
            # Where the bar was read, so a bar printed inside the micrograph can
            # be excluded from measurement instead of counted as a particle.
            "region": list(bar_region) if bar_region else None,
        }
    except Exception as exc:
        return None, {"method": "failed", "nm_per_px": None, "error": str(exc)}


def read_metadata(image_path):
    """
    Raw TIFF tags for ``image_path``, or None if there are none to read.

    Only used to hand the databar detector the instrument's own answer; a
    failure here is never fatal.
    """
    try:
        from .utils import extract_tiff_metadata

        return extract_tiff_metadata(str(image_path))
    except Exception:
        return None


def crop_databar(detector, image, args, metadata=None):
    """
    Remove the instrument databar from the bottom of the frame.

    With --crop-percent left at auto, the databar's height is measured. A fixed
    percentage is fragile in both directions: too small leaves a strip whose text
    and borders segment into spurious particles, too large eats real image area.

    Args:
        metadata: Raw TIFF metadata, when available. FEI records the scan height
            in tag 34682, which gives the databar height exactly — worth far more
            than measuring it off the pixels.

    Returns:
        tuple: (cropped_image, info_dict)
    """
    height = image.shape[0]

    if args.crop_percent is not None:
        if args.crop_percent <= 0:
            return image, {"method": "none", "rows_removed": 0}
        cropped = detector.crop_scale_bar(image, crop_percent=args.crop_percent)
        return cropped, {"method": "fixed-percent", "percent": args.crop_percent,
                         "rows_removed": height - cropped.shape[0]}

    try:
        databar = detector.detect_databar(image, metadata=metadata)
    except Exception:
        databar = None

    if databar and databar.get("has_databar") and databar.get("databar_height"):
        keep = height - int(databar["databar_height"])
        if 0 < keep < height:
            return image[:keep].copy(), {
                "method": "detected",
                "rows_removed": height - keep,
                "fraction": round(databar.get("databar_fraction", 0.0), 4),
            }

    if databar is not None:
        # Detection ran and found no databar. That is the normal case for TEM
        # frames, where the scale bar is burned into the micrograph itself and
        # there is no strip below it — trimming a fixed percentage would throw
        # away real image. Keep the frame; the bar is excluded from segmentation
        # rather than cropped.
        return image, {"method": "none", "rows_removed": 0}

    # Detection itself failed, so nothing is known either way; fall back to the
    # historical fixed percentage rather than risk leaving a databar in frame,
    # whose text and rules segment into spurious particles.
    cropped = detector.crop_scale_bar(image, crop_percent=DEFAULT_CROP_PERCENT)
    return cropped, {"method": "fallback-percent", "percent": DEFAULT_CROP_PERCENT,
                     "rows_removed": height - cropped.shape[0]}


def analyze_image(image_path, sam_model, detector, args):
    """Run scale detection, segmentation and measurement for a single image."""
    image = load_image(str(image_path))
    metadata = read_metadata(image_path)

    nm_per_px, scale_info = resolve_scale(detector, image, image_path, args)

    # Magnification gate, before anything expensive runs. A low-magnification
    # overview is a navigation frame: its particles are a few pixels across, so
    # counting them adds noise to a distribution rather than information.
    reason = _out_of_scale_range(nm_per_px, args)
    if reason:
        raise FrameSkipped(reason)

    # Trim the databar so it can't be segmented as a particle. Measuring its
    # height beats a fixed percentage, which either leaves a strip behind (and
    # the leftover text fragments into "particles") or eats into the micrograph.
    # The instrument's own scan height, when it recorded one, beats measuring.
    working, crop_info = crop_databar(detector, image, args, metadata=metadata)

    # SEM and TEM need opposite handling. Which one this is decides whether
    # particles are the brighter or the darker side of the frame, and guessing
    # that from contrast alone picks the aperture vignette or the grid bar
    # instead of the sample.
    kind = modality.resolve(getattr(args, "modality", None)) or modality.detect(
        image, metadata=metadata, databar_height=crop_info.get("rows_removed", 0))

    # Everything in the frame that is not specimen: beam-blocked area, and the
    # scale bar when it is printed inside the image rather than in a databar.
    exclude_boxes = []
    bar_region = (scale_info or {}).get("region")
    if bar_region and crop_info.get("rows_removed", 0) == 0:
        exclude_boxes.append(bar_region)
    analysable, region_info = region.analysable_region(working, exclude_boxes=exclude_boxes)
    region_info["modality"] = kind.to_dict()

    if not region.usable(analysable):
        raise ValueError(
            f"Only {100 * region_info['analysable_fraction']:.0f}% of the frame is "
            f"specimen; the rest is beam-blocked. Too little to count."
        )

    segmenter = ParticleSegmenter(sam_model)
    masks, scores = segmenter.segment_image(working, multimask_output=True)

    # Same ranking the app shows the analyst, so a batch run and an interactive
    # one agree about which mask is the right one.
    dark_particles = _polarity(args, kind)
    candidates = segmenter.rank_candidates(
        working, masks, top_k=3, dark_features=dark_particles,
        exclude=~analysable,
    )
    if not candidates:
        raise ValueError(
            "No mask candidate isolated anything convincing; the image may need "
            "interactive segmentation."
        )

    chosen = candidates[0]
    analyzer = ParticleAnalyzer(conversion_factor=nm_per_px, min_size=args.min_size)
    # Restrict to specimen before measuring, so nothing outside it is counted
    # even if it survived into the chosen mask.
    analyzer.analyze_mask(
        chosen["mask"] & analysable, min_size=args.min_size,
        remove_border=True, border_buffer=args.border_buffer,
    )
    mask_index, inverted, fraction = (
        chosen["mask_index"], chosen["inverted"], chosen["fraction"])
    rejected = [{k: c[k] for k in ("mask_index", "inverted", "fraction", "contrast")}
                for c in candidates[1:]]

    if args.clear_edges:
        analyzer.clear_edge_particles(buffer_size=0)

    measurements = analyzer.get_measurements(in_nm=nm_per_px is not None)
    stats = analyzer.get_summary_statistics()

    return {
        "measurements": measurements,
        "stats": stats,
        "scale": scale_info,
        "crop": crop_info,
        "modality": kind,
        "region": region_info,
        "dark_particles": dark_particles,
        "mask_index": int(mask_index),
        "mask_inverted": bool(inverted),
        "mask_foreground_fraction": round(fraction, 4),
        "mask_scores": [float(s) for s in scores],
        "other_candidates": rejected,
        "image_shape": list(working.shape[:2]),
    }


def write_plots(per_particle, out_dir, unit):
    """Publication-ready size distributions at 300 dpi, in PNG and PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if per_particle.empty:
        return []

    written = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(per_particle["diameter"], bins=25, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel(f"Equivalent diameter ({unit})")
    axes[0].set_ylabel("Count")
    axes[1].hist(per_particle["area"], bins=25, color="#DD8452", edgecolor="white")
    axes[1].set_xlabel(f"Area ({unit}²)")
    axes[1].set_ylabel("Count")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Particle size distribution (n = {len(per_particle)})")
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        path = out_dir / f"size_distribution.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path.name)
    plt.close(fig)
    return written


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sem-analyze",
        description="Batch particle analysis for SEM/TEM micrographs (no GUI).",
    )
    parser.add_argument("input", type=Path,
                        help="Image file, or folder of images to process")
    parser.add_argument("-o", "--output", type=Path, default=Path("analysis_output"),
                        help="Directory for CSVs, plots and the provenance record")

    model = parser.add_argument_group("model")
    model.add_argument("--model-type", default="vit_h",
                       choices=["vit_b", "vit_h", "vit_l"])
    model.add_argument("--checkpoint", type=Path,
                       help="SAM weights (default: look in sam_weights/)")
    model.add_argument("--device", help="Force a device, e.g. cpu, mps, cuda")

    scale = parser.add_argument_group("scale")
    scale.add_argument("--scale-method", default="auto",
                       choices=["auto", "metadata", "ocr"],
                       help="How to establish nm/pixel (default: auto)")
    scale.add_argument("--scale-nm-per-px", type=float,
                       help="Override detection with a fixed nm/pixel for every image")
    scale.add_argument("--crop-percent", type=float, default=None,
                       help="Percent of image height to trim off the bottom. "
                            "Default is to measure the databar; use 0 to keep the "
                            "full frame")

    frame = parser.add_argument_group("frame")
    frame.add_argument("--modality", default="auto",
                       choices=["auto", "SEM", "TEM"],
                       help="Instrument kind. Decides particle polarity and whether "
                            "a databar is expected (default: read it from the file)")
    frame.add_argument("--particles", default="auto",
                       choices=["auto", "bright", "dark"],
                       help="Whether particles are brighter or darker than their "
                            "surroundings. Default follows the modality: bright for "
                            "SEM, dark for TEM")
    frame.add_argument("--max-nm-per-px", type=float,
                       help="Skip frames coarser than this, i.e. magnifications too "
                            "low to resolve particles. Low-magnification overviews "
                            "are for navigation and counting them adds noise")
    frame.add_argument("--min-nm-per-px", type=float,
                       help="Skip frames finer than this")

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument("--min-size", type=int, default=30,
                          help="Discard particles smaller than this many pixels (default: 30)")
    analysis.add_argument("--border-buffer", type=int, default=4)
    analysis.add_argument("--clear-edges", action="store_true",
                          help="Drop particles touching the frame edge; they are only "
                               "partly imaged, so their size is not a real measurement")

    output = parser.add_argument_group("output")
    output.add_argument("--no-plots", action="store_true")
    output.add_argument("--no-checkpoint-hash", action="store_true",
                        help="Skip hashing the weights file (saves a few seconds)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.input.is_dir():
        image_paths = [Path(p) for p in find_images_in_folder(str(args.input))]
    elif args.input.exists():
        image_paths = [args.input]
    else:
        print(f"error: no such file or folder: {args.input}", file=sys.stderr)
        return 2

    if not image_paths:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2

    checkpoint = args.checkpoint or find_checkpoint(args.model_type)
    if checkpoint is None or not Path(checkpoint).exists():
        print(
            f"error: no {args.model_type} weights found. Pass --checkpoint, or run:\n"
            f"    python download_sam_weights.py",
            file=sys.stderr,
        )
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Analysing {len(image_paths)} image(s) -> {args.output}")

    sam_model = SAMModel(str(checkpoint), model_type=args.model_type, device=args.device)
    detector = ScaleDetector(use_gpu=False)

    per_particle_rows = []
    per_image_rows = []
    image_records = []

    for number, image_path in enumerate(image_paths, start=1):
        print(f"[{number}/{len(image_paths)}] {image_path.name}")
        try:
            result = analyze_image(image_path, sam_model, detector, args)
        except FrameSkipped as skip:
            print(f"    skipped: {skip}")
            image_records.append({"image": str(image_path), "skipped": str(skip)})
            continue
        except Exception as exc:
            print(f"    failed: {exc}", file=sys.stderr)
            image_records.append({"image": str(image_path), "error": str(exc)})
            continue

        measurements = result["measurements"]
        stats = result["stats"]
        unit = measurements["unit"]
        scale_warning = result["scale"].get("warning")

        for index, (area, diameter, centroid) in enumerate(
            zip(measurements["areas"], measurements["diameters"], measurements["centroids"]), start=1
        ):
            per_particle_rows.append({
                "image": image_path.name,
                "particle": index,
                "area": area,
                "diameter": diameter,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "unit": unit,
            })

        per_image_rows.append({
            "image": image_path.name,
            "modality": result["modality"].kind,
            "instrument": result["modality"].instrument,
            "num_particles": stats.get("num_particles", 0),
            "nm_per_px": result["scale"]["nm_per_px"],
            "scale_method": result["scale"]["method"],
            "analysable_fraction": result["region"]["analysable_fraction"],
            "area_mean": stats.get("area_mean"),
            "area_median": stats.get("area_median"),
            "area_std": stats.get("area_std"),
            "diameter_mean": stats.get("diameter_mean"),
            "diameter_median": stats.get("diameter_median"),
            "diameter_std": stats.get("diameter_std"),
            "unit": unit,
        })

        image_records.append({
            "image": str(image_path.resolve()),
            "sha256": sha256_file(image_path),
            "shape": result["image_shape"],
            "scale": result["scale"],
            "crop": result["crop"],
            "modality": result["modality"].to_dict(),
            "region": result["region"],
            "dark_particles": result["dark_particles"],
            "mask_index": result["mask_index"],
            "mask_inverted": result["mask_inverted"],
            "mask_foreground_fraction": result["mask_foreground_fraction"],
            "mask_scores": result["mask_scores"],
            "other_candidates": result["other_candidates"],
            "num_particles": stats.get("num_particles", 0),
        })

        note = f"  {scale_warning}" if scale_warning else ""
        blocked = 1 - result["region"]["analysable_fraction"]
        blocked_note = f", {100 * blocked:.0f}% blocked" if blocked > 0.005 else ""
        print(f"    {result['modality'].kind}: {stats.get('num_particles', 0)} particles, "
              f"scale={result['scale']['method']}{blocked_note}{note}")

    per_particle = pd.DataFrame(per_particle_rows)
    per_image = pd.DataFrame(per_image_rows)
    per_particle.to_csv(args.output / "particles.csv", index=False)
    per_image.to_csv(args.output / "per_image_summary.csv", index=False)

    plots = []
    if not args.no_plots and not per_particle.empty:
        unit = per_particle["unit"].iloc[0]
        plots = write_plots(per_particle, args.output, unit)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_revision": _git_revision(),
        "versions": _versions(),
        "device": str(sam_model.device),
        "model": {
            "type": args.model_type,
            "checkpoint": str(Path(checkpoint).resolve()),
            "sha256": None if args.no_checkpoint_hash else sha256_file(checkpoint),
        },
        "parameters": {
            "min_size": args.min_size,
            "border_buffer": args.border_buffer,
            "clear_edges": args.clear_edges,
            "crop_percent": args.crop_percent,
            "scale_method": args.scale_method,
            "scale_nm_per_px": args.scale_nm_per_px,
            "modality": args.modality,
            "particles": args.particles,
            "max_nm_per_px": args.max_nm_per_px,
            "min_nm_per_px": args.min_nm_per_px,
        },
        "images": image_records,
        "outputs": ["particles.csv", "per_image_summary.csv", *plots],
        "note": "Automatic pipeline only — no interactive refinement was applied.",
    }
    with open(args.output / "run.json", "w") as handle:
        json.dump(provenance, handle, indent=2)

    failed = sum(1 for record in image_records if "error" in record)
    skipped = sum(1 for record in image_records if "skipped" in record)
    kinds = per_image["modality"].value_counts().to_dict() if not per_image.empty else {}
    breakdown = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
    print(f"\nDone. {len(per_image)} image(s) analysed"
          + (f" ({breakdown})" if breakdown else "")
          + (f", {skipped} skipped" if skipped else "")
          + (f", {failed} failed" if failed else "")
          + f", {len(per_particle)} particles total.")
    print(f"Wrote {args.output}/particles.csv, per_image_summary.csv, run.json"
          + (f", {', '.join(plots)}" if plots else ""))
    return 1 if failed and per_image.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
